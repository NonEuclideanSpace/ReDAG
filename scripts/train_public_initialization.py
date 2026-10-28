"""Teach actual initial PLAN -> RENDER transitions without claiming visual success."""
import argparse
import json
from collections import Counter
from pathlib import Path
import sys

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from collect_cached_public_trajectories import CachedCompiler, CachedRenderBridge, read_json
from agentic_image.actor.records import ActionSample, ActorSnapshot
from agentic_image.config import load_config
from agentic_image.core.runtime import EpisodeEnvironment
from agentic_image.core.types import PlanSpec
from agentic_image.data.io import read_jsonl, write_jsonl
from agentic_image.data.paired import actor_snapshot, save_action_record
from agentic_image.learning.transaction import atomic_json
from agentic_image.render.types import ConditionBundle
from agentic_image.runtime.components import Components
from agentic_image.runtime.prompt import SYSTEM_PROMPT
from agentic_image.runtime.training import train_actor_sft


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--collection', default='runs/public-plan-candidates-after-global')
    p.add_argument('--renders', default='runs/sampled-plan-initial-renders')
    p.add_argument('--retained-records', default='data/public-stage-transitions/actor_records.jsonl')
    p.add_argument('--policy', required=True)
    p.add_argument('--output', default='data/public-initialization')
    p.add_argument('--run-dir', default='runs/public-initialization')
    p.add_argument('--updates', type=int, default=80)
    p.add_argument('--device', default='cuda:1')
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config)
    config['run_dir'] = args.run_dir
    config['models'].update(actor_policy=args.policy, actor_reference=None)
    config['judge']['calibration'] = None
    config['sft'].update(actor_batch=4, protocol_phase_counts=[3, 0, 0, 1], protocol_updates=args.updates)
    config['data']['actor_records'] = str(output / 'actor_records.jsonl')
    for name in ('actor', 'judge', 'encoders', 'renderer'):
        config['devices'][name] = args.device
    metadata = read_json(Path(args.collection) / 'metadata.json')
    task_map = {t['task_id']: t for t in metadata['tasks']}
    rendered = []
    for shard in sorted(Path(args.renders).glob('shard-*')):
        if read_json(shard / 'summary.json')['status'] != 'completed':
            raise ValueError('Original real render shard is incomplete')
        rendered.extend(read_json(shard / 'results.json'))
    rendered.sort(key=lambda x: x['index'])
    if len(rendered) != 32 or len({x['family_id'] for x in rendered}) != 32:
        raise ValueError('Initialization uses the fixed 32 actually rendered training families')
    for row in rendered:
        task = task_map[row['task_id']]
        if task['split_id'] != 'train' or task['family_id'] != row['family_id'] or task['user_request']['source_image_id'] != row['source_image_id'] or task['user_request']['text'] != row['request']:
            raise ValueError('Initialization source/task/family changed')
        if row['plan'] != PlanSpec(global_text=row['request']).model_dump(mode='json'):
            raise ValueError('Initialization plan must match the unchanged original request')
    policy = Path(args.policy).resolve()
    weight = policy / 'adapter_model.safetensors'
    registration = dict(scope='executed_public_initialization_sft_v1', initial_policy=str(policy), initial_policy_bytes=weight.stat().st_size, initial_policy_mtime_ns=weight.stat().st_mtime_ns, families=sorted(x['family_id'] for x in rendered), independent_families=32, updates=args.updates, batch=4, phase_counts=[3,0,0,1], phase_roles={'0':'initial_PLAN_and_initial_RENDER','3':'retained_executed_public_recovery'}, visual_success_label=False, repair_success_label=False, full_four_phase_actor_sft=False, new_renderer_generations=0)
    reg = output / 'registration.json'
    if reg.exists() and read_json(reg) != registration:
        raise ValueError('Initialization registration changed')
    atomic_json(reg, registration)
    (output / 'training.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    components = Components(config)
    actor = components.load_actor()
    components.load_judge(private=False)
    tokenizer = actor.processor.tokenizer
    count = lambda text: len(tokenizer.encode(text, add_special_tokens=False))
    budget = config['actor']['max_input_tokens'] - count(SYSTEM_PROMPT) - 128
    records, evidence = [], []
    if not (output / 'completed.json').exists():
        if (output/'execution_evidence.jsonl').exists():
            evidence = list(read_jsonl(output/'execution_evidence.jsonl'))
            saved_ids = {record_id for item in evidence for record_id in item['record_ids']}
            records = [item for item in read_jsonl(output/'actor_records.jsonl') if item['record_id'] in saved_ids]
            if {item['record_id'] for item in records} != saved_ids:
                raise ValueError('Previously completed initialization evidence lost its records')
        completed_tasks = {item['task_id'] for item in evidence}
        for row in rendered:
            if row['task_id'] in completed_tasks:
                continue
            bundle = ConditionBundle.load(row['bundle_path'])
            compiler = CachedCompiler(components, bundle)
            renderer = CachedRenderBridge(components, row, 0)
            plan_result = read_json(Path(args.collection) / 'tasks' / f"{row['index']:03d}" / 'result.json')
            attempt = next(a for a in plan_result['attempts'] if a['attempt'] == row['sample_attempt'])
            if not attempt['plan_contract_succeeded'] or not attempt['bundle_compiled'] or attempt['internal_plan'] != row['plan']:
                raise ValueError('The retained actor PLAN lacks its actual successful execution')
            def restore(path):
                return EpisodeEnvironment.restore(read_json(path), components.artifacts, compiler, renderer, components.observer, compiler.load, token_counter=count, max_input_tokens=budget)
            env = restore(attempt['before_environment_path'])
            if env.state.plan is not None or env.state.executed_renders:
                raise ValueError('The recorded root PLAN must precede all plans and renders')
            root_was_invalid = env.state.budget.invalid > 0
            sample = ActionSample.from_state_dict(torch.load(attempt['action_sample_path'], map_location='cpu', weights_only=True))
            snapshot = ActorSnapshot.load(attempt['snapshot_path'])
            result = env.step(sample.text, 11)
            if result.error or env.state.plan.model_dump(mode='json') != row['plan']:
                raise ValueError('Original sampled PLAN did not execute successfully')
            provenance = dict(verification='executed_public_plan_recovery' if root_was_invalid else 'executed_public_initial_plan', original_action=attempt['action_sample_path'], actual_actor_sample=True, visual_success_label=False, root_had_actual_error=root_was_invalid)
            plan_record = save_action_record(components, env, sample.text, output/'snapshots', f"initialization:{row['index']:03d}:plan", 3 if root_was_invalid else 0, env.episode_id, provenance, snapshot, sample)
            records.append(plan_record)
            env = restore(row['initial_unrendered_state'])
            if env.state.executed_renders or env.task.task_id != row['task_id']:
                raise ValueError('Initial RENDER must start from its exact unrendered public state')
            snapshot = actor_snapshot(components, env)
            before = env.snapshot()
            action = dict(kind='RENDER', control_set_id=env.state.manifest.active_control_set_id, expected_revision=env.state.manifest.revision_id)
            result = env.step(action, 11)
            if result.error or result.render is None or not result.render.metadata.get('cache_hit') or result.render.image_id != row['result']['image_id'] or env.state.budget.renders != before['budget']['renders'] + 1:
                raise ValueError('The actual initial RENDER did not execute its exact original trace')
            provenance = dict(verification='executed_public_initial_render', teacher_origin='deterministic_public_initialization', actual_actor_sample=False, visual_success_label=False, repair_success_label=False, render_trace=result.render.trajectory_id, original_state=row['initial_unrendered_state'])
            record = save_action_record(components, env, action, output/'snapshots', f"initialization:{row['index']:03d}:render", 0, env.episode_id, provenance, snapshot)
            records.append(record)
            evidence.append(dict(task_id=row['task_id'], family_id=row['family_id'], before=before, after=env.snapshot(), action=action, result=result.model_dump(mode='json'), render_events=renderer.events, record_ids=[plan_record['record_id'],record['record_id']]))
            write_jsonl(output/'actor_records.jsonl', records)
            write_jsonl(output/'execution_evidence.jsonl', evidence)
            print(json.dumps({'built_initialization_families':len(evidence),'initialization_records':len(records)}),flush=True)
        families=set(registration['families'])
        for row in read_jsonl(args.retained_records):
            if row.get('phase') == 3 and row.get('verified') and row['family_id'] in families:
                if not Path(row['snapshot_path']).is_file() or row['target_ids'][-1] != actor.config.action_eos_token_id:
                    raise ValueError('Retained recovery record lost its snapshot/EOS')
                records.append(row)
        if len({r['record_id'] for r in records}) != len(records) or sum(r['record_id'].startswith('initialization:') for r in records)!=64 or sum(r['record_id'].startswith('initialization:') and r['record_id'].endswith(':render') and r['phase']==0 for r in records)!=32 or not any(r['phase']==3 for r in records):
            raise ValueError('Initialization/recovery record pool is incomplete or duplicated')
        write_jsonl(output/'actor_records.jsonl',records)
        atomic_json(output/'completed.json',dict(status='completed',records=len(records),phase_counts=dict(Counter(r['phase'] for r in records)),families=32,actual_initial_render_steps=32,visual_success_label=False))
    components.release_frozen(encoders=True,judge=True)
    result=train_actor_sft(components,stage='protocol',updates=args.updates,resume=args.resume)
    metrics=list(read_jsonl(Path(args.run_dir)/'sft_protocol/metrics.jsonl'))
    if [r['step'] for r in metrics] != list(range(1,args.updates+1)):
        raise ValueError('Requested updates lack complete committed metrics')
    atomic_json(output/'training_completed.json',dict(status='completed',scope=registration['scope'],policy=result,initial_policy=str(policy),updates=args.updates,actually_exposed_families=len({f for row in metrics for f in row['family_ids']}),actual_record_exposures=sum(len(row['record_ids']) for row in metrics),metrics=str((Path(args.run_dir)/'sft_protocol/metrics.jsonl').resolve())))


if __name__=='__main__':
    main()
