import argparse
from collections import Counter
import copy
import gc
import json
from pathlib import Path
import time

import torch
from peft import load_peft_weights

from agentic_image.actor.config import adapter_directory
from agentic_image.actor.head import head_log_probs
from agentic_image.actor.records import ActorSnapshot
from agentic_image.config import load_config
from agentic_image.core.protocol import parse_action
from agentic_image.core.runtime import EpisodeEnvironment
from agentic_image.core.types import ContractError, PublicTask
from agentic_image.data.io import read_jsonl, write_jsonl, write_report
from agentic_image.data.paired import actor_snapshot, public_environment
from agentic_image.data.public_protocol import CASES, prepare_recovery, recovery_action, verify_recovery
from agentic_image.learning.sampling import stream_seed
from agentic_image.runtime.components import Components


def restore(components, packet):
    tokenizer = components.actor.processor.tokenizer
    return EpisodeEnvironment.restore(
        copy.deepcopy(packet['environment']), components.artifacts, None, None, None,
        lambda identifier: (_ for _ in ()).throw(ValueError(f'Unexpected bundle: {identifier}')),
        token_counter=lambda text: len(tokenizer.encode(text, add_special_tokens=False)),
        max_input_tokens=packet['view_token_budget'],
    )


@torch.no_grad()
def nll(actor, snapshot, ids):
    with actor.lock:
        actor._set_adapter('policy', training=False)
        tokens = torch.tensor(ids, dtype=torch.long, device=actor.device)
        hidden = actor._hidden(snapshot, tokens)
        values = head_log_probs(
            hidden, actor.base.lm_head, tokens, actor.config.temperature,
            actor.config.head_chunk_size,
        ).float().cpu()
    if not torch.isfinite(values).all():
        raise RuntimeError('Nonfinite teacher-forced target likelihood')
    return {'mean_nll': float(-values.mean()), 'sum_nll': float(-values.sum()), 'tokens': len(ids)}


def execute(components, packet, text, seed):
    environment = restore(components, packet)
    parsed, parse_error = None, None
    try:
        parsed = parse_action(text)
    except ContractError as error:
        parse_error = {'code': error.code, 'message': str(error)}
    if parsed is not None and parsed.kind in {'RENDER', 'APPLY', 'INSPECT', 'DRAFT_REPAIR'}:
        return {
            'status': 'not_executed_out_of_scope', 'action_kind': parsed.kind,
            'parse_error': None, 'recovery_success': False, 'contract_valid': None,
            'reason': 'This public read-recovery evaluation does not execute visual services',
        }
    result = environment.step(text, seed)
    success, reason = True, None
    try:
        verify_recovery(environment, result)
    except RuntimeError as error:
        success, reason = False, str(error)
    return {
        'status': 'executed', 'action_kind': result.action_kind, 'parse_error': parse_error,
        'contract_valid': result.error is None and not result.protocol_failure,
        'recovery_success': success, 'reason': reason,
        'result': result.model_dump(mode='json'),
        'budget_after': environment.state.budget.model_dump(mode='json'),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/pilot-small.yaml')
    parser.add_argument('--tasks', required=True)
    parser.add_argument('--training-tasks', default='data/public/protocol_pilot_tasks.jsonl')
    parser.add_argument('--policy', default='runs/protocol-warmup/sft_protocol/policy/policy')
    parser.add_argument('--output', default='runs/protocol-evaluation')
    parser.add_argument('--seed', type=int, default=101)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'snapshots').mkdir(exist_ok=True)
    tasks = [PublicTask.model_validate(row) for row in read_jsonl(args.tasks)]
    training = list(read_jsonl(args.training_tasks))
    if len(tasks) != 16 or len({task.family_id for task in tasks}) != 16:
        raise ValueError('Evaluation requires exactly 16 distinct held-out families')
    if any(task.split_id != 'official-dev' for task in tasks):
        raise ValueError('Only official-dev tasks are admitted to this evaluation')
    if {task.family_id for task in tasks} & {row['family_id'] for row in training}:
        raise ValueError('Evaluation family overlaps protocol training')
    if {task.user_request.source_image_id for task in tasks} & {
        row['user_request']['source_image_id'] for row in training
    }:
        raise ValueError('Evaluation source image overlaps protocol training')
    policy_path = adapter_directory(args.policy)
    policy_file = policy_path / 'adapter_model.safetensors'
    if not policy_file.is_file():
        raise FileNotFoundError(policy_file)
    config = load_config(args.config)
    config['run_dir'] = str(output)
    config['devices']['actor'] = 'cuda:0'
    config['devices']['judge'] = 'cuda:0'
    config['models']['actor_policy'] = None
    config['models']['actor_reference'] = None
    registration = {
        'scope': 'heldout_public_contract_recovery', 'tasks': [task.model_dump(mode='json') for task in tasks],
        'training_tasks': str(Path(args.training_tasks).resolve()),
        'policy': str(policy_path.resolve()), 'policy_bytes': policy_file.stat().st_size,
        'policy_mtime_ns': policy_file.stat().st_mtime_ns, 'seed': args.seed,
        'actor_model': config['models']['actor'], 'actor_revision': config['models']['actor_revision'],
        'max_action_tokens': config['actor']['max_new_tokens'], 'temperature': config['actor']['temperature'],
        'training_updates': 0, 'private_rewards': 0, 'visual_quality_evaluation': False,
    }
    manifest = output / 'registration.json'
    if manifest.exists() and json.loads(manifest.read_text()) != registration:
        raise ValueError('Existing evaluation registration differs; use a fresh output')
    write_report(manifest, registration)
    started = time.monotonic()
    torch.manual_seed(11)
    components = Components(config)
    actor = components.load_actor()
    if any(torch.count_nonzero(value).item() for name, value in actor.model.named_parameters() if 'lora_B' in name):
        raise RuntimeError('The baseline adapter must have exactly zero LoRA output')
    packet_path = output / 'states.jsonl'
    packets = {row['state_id']: row for row in read_jsonl(packet_path)} if packet_path.exists() else {}
    if len(packets) < len(tasks) * len(CASES):
        components.load_judge(private=False)
        for task_index, task in enumerate(tasks):
            for case in CASES:
                state_id = f'heldout-protocol:{task_index:03d}:{case}'
                if state_id in packets:
                    continue
                environment = public_environment(components, task, state_id)
                setup = prepare_recovery(environment, case)
                snapshot = actor_snapshot(components, environment)
                target = recovery_action(case, snapshot.metadata['view']['payload'])
                raw = json.dumps(environment.public_action(target), ensure_ascii=True, separators=(',', ':'))
                target_ids = actor.processor.tokenizer.encode(raw, add_special_tokens=False) + [248046]
                snapshot_path = output / 'snapshots' / f'{task_index:03d}-{case}.pt'
                snapshot.save(snapshot_path)
                packet = {
                    'state_id': state_id, 'task_id': task.task_id, 'family_id': task.family_id,
                    'case': case, 'environment': environment.snapshot(), 'setup': setup,
                    'view_token_budget': environment.view_builder.max_input_tokens,
                    'snapshot_path': str(snapshot_path.resolve()), 'target_text': raw,
                    'target_ids': target_ids, 'seed': stream_seed(args.seed, task.task_id, case),
                }
                result = environment.step(target)
                verify_recovery(environment, result)
                packet['verified_target_result'] = result.model_dump(mode='json')
                packets[state_id] = packet
                write_jsonl(packet_path, packets.values())
                print(json.dumps({'stage': 'states', 'completed': len(packets), 'total': 32, 'state_id': state_id}), flush=True)
                environment.observer = None
                del environment, snapshot
        components.release_frozen(judge=True)
    result_path = output / 'results.jsonl'
    results = {(row['policy'], row['state_id']): row for row in read_jsonl(result_path)} if result_path.exists() else {}
    baseline_state = actor.adapter_state()['policy']
    for label in ('baseline', 'after'):
        if label == 'after':
            trained = load_peft_weights(policy_path, device='cpu', local_files_only=True)
            if set(trained) != set(baseline_state) or any(trained[key].shape != baseline_state[key].shape for key in trained):
                raise ValueError('Trained adapter does not exactly cover the baseline policy')
            actor.restore_adapter_state({'policy': trained})
            loaded = actor.adapter_state()['policy']
            if any(not torch.equal(loaded[key], trained[key]) for key in trained):
                raise RuntimeError('Trained adapter values were not loaded exactly')
            del trained, loaded
            gc.collect()
        for packet in packets.values():
            if (label, packet['state_id']) in results:
                continue
            snapshot = ActorSnapshot.load(packet['snapshot_path'])
            tick = time.monotonic()
            likelihood = nll(actor, snapshot, packet['target_ids'])
            sample = actor.sample(snapshot, packet['seed'], f'protocol-evaluation:{label}')
            execution = execute(components, packet, sample.text, packet['seed'])
            row = {
                'policy': label, 'state_id': packet['state_id'], 'task_id': packet['task_id'],
                'family_id': packet['family_id'], 'case': packet['case'], 'seed': packet['seed'],
                'target': likelihood, 'sample_text': sample.text, 'sample_token_ids': sample.token_ids.tolist(),
                'sample_log_probs': sample.behavior_log_probs.tolist(), 'finish_reason': sample.finish_reason,
                'execution': execution, 'seconds': time.monotonic() - tick,
            }
            results[label, packet['state_id']] = row
            write_jsonl(result_path, results.values())
            print(json.dumps({'stage': label, 'completed': sum(key[0] == label for key in results), 'total': 32, 'state_id': packet['state_id'], 'mean_nll': likelihood['mean_nll'], 'recovery_success': execution['recovery_success'], 'seconds': row['seconds']}), flush=True)
            del sample, snapshot
    summary = {'status': 'complete', 'scope': registration['scope'], 'families': 16, 'states': 32, 'training_updates': 0, 'visual_quality_evaluation': False, 'policies': {}}
    for label in ('baseline', 'after'):
        rows = [row for row in results.values() if row['policy'] == label]
        summary['policies'][label] = {
            'mean_action_nll': sum(row['target']['mean_nll'] for row in rows) / len(rows),
            'token_weighted_nll': sum(row['target']['sum_nll'] for row in rows) / sum(row['target']['tokens'] for row in rows),
            'recovery_successes': sum(row['execution']['recovery_success'] for row in rows),
            'sample_count': len(rows), 'recovery_success_rate': sum(row['execution']['recovery_success'] for row in rows) / len(rows),
            'execution_statuses': dict(Counter(row['execution']['status'] for row in rows)),
            'action_kinds': dict(Counter(str(row['execution']['action_kind']) for row in rows)),
            'sampling_and_scoring_seconds': sum(row['seconds'] for row in rows),
        }
    summary['paired_nll_change_after_minus_before'] = summary['policies']['after']['mean_action_nll'] - summary['policies']['baseline']['mean_action_nll']
    summary['elapsed_seconds'] = time.monotonic() - started
    summary['peak_gpu0_bytes'] = torch.cuda.max_memory_allocated(0)
    write_report(output / 'summary.json', summary)
    write_report(output / 'completed.json', {'status': 'complete', 'summary': str((output / 'summary.json').resolve())})
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
