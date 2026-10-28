import json
import math
import os
from pathlib import Path
import tempfile

from .fixed_evaluation import read_json


METHODS = ('base', 'sft', 'agent_rl', 'joint_rl')
DEFAULT_SELECTION = 'runs/sampler-fallback-200/sampler_selection.json'


def file_identity(path):
    path = Path(path).resolve()
    stat = path.stat()
    if not path.is_file() or stat.st_size <= 0:
        raise ValueError(f'An actual nonempty weight file is required: {path}')
    return {'path': str(path), 'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns}


def policy_identity(path, role):
    if path is None:
        return None
    path = Path(path)
    if role == 'actor':
        if not (path / 'adapter_config.json').is_file():
            path = path / 'policy'
        if not (path / 'adapter_config.json').is_file():
            raise FileNotFoundError(f'No policy adapter configuration: {path}')
        path = path / 'adapter_model.safetensors'
    elif path.is_dir():
        path = path / 'policy.pt'
    return file_identity(path)


def freeze_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, ensure_ascii=False, sort_keys=True)
    try:
        try:
            os.link(temporary, path)
        except FileExistsError:
            if read_json(path) != value:
                raise ValueError(f'Frozen evaluation identity changed: {path}')
    finally:
        temporary.unlink()


def sampler_protocol(selection_path, tasks):
    path = Path(selection_path).resolve()
    selection = read_json(path)
    if selection.get('accepted') is not True or selection.get('status') != 'accepted':
        raise ValueError('Complete controller evaluation waits for an accepted stochastic sampler')
    steps, eta, cfg = (selection.get(key) for key in ('steps', 'eta', 'cfg_scale'))
    if (type(steps) is not int or steps <= 0
            or any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in (eta, cfg))):
        raise ValueError('Sampler steps, eta and CFG must be finite positive numeric values')
    qualification = selection.get('qualified_on', {})
    expanded = selection.get('families') == 128
    original_path = Path(qualification['original_registration']).resolve() if expanded else path.parent / 'registration.json'
    registration = read_json(original_path)
    candidate = selection.get('candidate')
    decision = selection.get('decisions', {}).get(candidate)
    if decision is None:
        decision = selection.get('decision')
    if (not decision or decision.get('accepted') is not True
            or registration.get('samplers', {}).get(candidate) != [steps, eta]
            or selection.get('families') not in (64, 128)
            or selection.get('eval_seeds') != registration.get('eval_seeds')
            or selection.get('paired_records', selection.get('base_pair_records')) != selection['families'] * 4
            or registration.get('cfg_scale') != cfg
            or registration.get('tasks') != [task.model_dump(mode='json') for task in tasks]):
        raise ValueError('The accepted sampler lacks its exact registered comparison evidence')
    confirmation = None
    if expanded:
        confirmation_path = Path(qualification['confirmation_registration']).resolve()
        confirmation = read_json(confirmation_path)
        old_families = {task['family_id'] for task in registration['tasks']}
        new_families = {task['family_id'] for task in confirmation.get('tasks', [])}
        if (qualification.get('families') != 128 or qualification.get('original_families') != 64
                or qualification.get('confirmation_families') != 64
                or len(registration['tasks']) != 64 or len(confirmation.get('tasks', [])) != 64
                or len(old_families) != 64 or len(new_families) != 64 or old_families & new_families
                or confirmation.get('samplers', {}).get(candidate) != [steps, eta]
                or confirmation.get('cfg_scale') != cfg or confirmation.get('eval_seeds') != selection['eval_seeds']
                or confirmation.get('models') != registration.get('models')):
            raise ValueError('Expanded qualification requires 64 genuinely independent families under the same frozen sampler and policies')
        for comparison in (decision, selection.get('original_decision'), selection.get('independent_confirmation_decision')):
            if not comparison or any(comparison.get(key) != .05 for key in ('success_margin', 'defect_margin', 'protection_margin')):
                raise ValueError('Expanded qualification must retain all original and independent results with unchanged 5pp gates')
    policies = {role: read_json(original_path.parent / f'{role}_policy_identity.json') for role in ('actor', 'renderer')}
    if selection.get('renderer_policy_identity') != policies['renderer']:
        raise ValueError('Sampler qualification renderer identity differs from its actual comparison')
    return {'selection_path': str(path), 'selection': selection,
        'parameters': {'steps': steps, 'eta': eta, 'cfg_scale': cfg},
        'qualified_on': {'scope': 'frozen_sft_sampler_comparison', 'policies': policies,
            'models': registration['models'], 'comparison_families': selection['families'],
            'comparison_seeds': selection['eval_seeds'], 'original_registration': str(original_path),
            'independent_confirmation': qualification if expanded else None,
            'independent_confirmation_tasks': confirmation['tasks'] if confirmation else None},
        'qualification_is_not_a_claim_about_evaluated_policy': True}


def final_sft_proof(completion_path, actor_identity):
    completion = read_json(completion_path)
    if (completion.get('status') != 'completed' or completion.get('scope') != 'full_four_phase_actor_sft_v1'
            or type(completion.get('updates')) is not int or completion['updates'] < 200
            or policy_identity(completion.get('policy'), 'actor') != actor_identity):
        raise ValueError('Full SFT evaluation requires the actual completed four-phase actor export')
    metrics = [json.loads(line) for line in Path(completion['metrics']).read_text().splitlines() if line.strip()]
    if [row['step'] for row in metrics] != list(range(1, completion['updates'] + 1)):
        raise ValueError('Full SFT completion is not backed by its complete update metrics')
    marker = Path(metrics[-1]['checkpoint']) / 'COMMITTED.json'
    committed = read_json(marker)
    if (committed.get('status') != 'COMMITTED' or committed.get('new_version') != f"actor:{completion['updates']}"
            or committed.get('optimizer_steps') != ['actor'] or set(committed.get('components', {})) != {'actor'}):
        raise ValueError('The SFT export lacks its final actual actor commit')
    return {'completion_path': str(Path(completion_path).resolve()), 'completion': completion,
        'commit_path': str(marker.resolve()), 'commit': committed,
        'actor_state': file_identity(marker.parent / committed['components']['actor'])}


def committed_rl_policy(config, method, checkpoint=None):
    mode = {'agent_rl': 'agent', 'joint_rl': 'joint'}[method]
    folder = Path(config['run_dir']) / f'rl_{mode}'
    completion = read_json(folder / 'training_completed.json')
    if (completion.get('status') != 'completed' or completion.get('mode') != mode
            or completion.get('effective_target_met') is not True
            or type(completion.get('required_effective_updates')) is not int
            or completion['required_effective_updates'] <= 0
            or completion.get('effective_update_counts', {}).get('required_components', 0) < completion['required_effective_updates']):
        raise ValueError('RL evaluation requires completed effective updates, not an SFT initialization or interrupted state')
    metrics = [json.loads(line) for line in (folder / 'metrics.jsonl').read_text().splitlines() if line.strip()]
    if [row['round'] for row in metrics] != list(range(1, completion['rounds'] + 1)):
        raise ValueError('RL completion is not backed by its complete round metrics')
    marker = (Path(metrics[-1]['checkpoint']) / 'COMMITTED.json').resolve()
    if checkpoint is not None:
        explicit = Path(checkpoint)
        explicit = explicit / 'COMMITTED.json' if explicit.is_dir() else explicit
        if explicit.resolve() != marker:
            raise ValueError('Evaluation must use the final commit bound to completed RL metrics')
    committed = read_json(marker)
    if (committed.get('status') != 'COMMITTED' or committed.get('metadata', {}).get('mode') != mode
            or set(committed.get('components', {})) != {'actor', 'renderer'}
            or committed.get('new_version') != f"{mode}:{completion['rounds']}"
            or committed.get('transaction_id') != marker.parent.name
            or marker.parent.parent.resolve() != (folder / 'checkpoints').resolve()):
        raise ValueError('RL checkpoint is not a complete committed actor-renderer pair for this stage')
    recorded = committed['metadata']['config']
    for key in ('actor', 'actor_revision', 'renderer', 'renderer_revision', 'renderer_profile'):
        if recorded['models'].get(key) != config['models'].get(key):
            raise ValueError(f'RL checkpoint architecture differs from the evaluated model: {key}')
    if any(recorded['renderer'].get(key) != config['renderer'].get(key) for key in ('steps', 'eta', 'cfg_scale')):
        raise ValueError('RL checkpoint training did not use the locked common sampler')
    for role in ('actor', 'renderer'):
        if policy_identity(recorded['models'].get(role + '_policy'), role) != policy_identity(config['models'].get(role + '_policy'), role):
            raise ValueError('RL checkpoint used a different SFT initialization')
    if metrics[-1].get('effective_update_counts') != completion['effective_update_counts']:
        raise ValueError('RL completion and final checkpoint metrics disagree on effective updates')
    components = {}
    for name, filename in committed['components'].items():
        state_path = (marker.parent / filename).resolve()
        if state_path.parent != marker.parent:
            raise ValueError('Checkpoint component escaped its committed transaction')
        components[name] = file_identity(state_path)
    return {'source': 'committed_rl_pair', 'method': method, 'commit_path': str(marker),
        'commit': committed, 'completion': completion, 'components': components,
        'training_state': file_identity(marker.parent / 'training_state.pt')}


def evaluated_policy(config, method, sampler, sft_completion, checkpoint=None):
    identities = {role: policy_identity(config['models'].get(role + '_policy'), role) for role in ('actor', 'renderer')}
    if method == 'base':
        if checkpoint or any(identities.values()):
            raise ValueError('Base evaluation requires both untrained policies and no checkpoint override')
        return {'source': 'untrained_base', 'method': method, 'policies': identities}
    if not all(identities.values()):
        raise ValueError('Trained evaluation requires explicit actor and renderer SFT initialization files')
    sft = final_sft_proof(sft_completion, identities['actor'])
    if method == 'sft':
        if checkpoint:
            raise ValueError('SFT evaluation cannot silently substitute an RL checkpoint')
        if identities['renderer'] != sampler['qualified_on']['policies']['renderer']:
            raise ValueError('The SFT renderer export differs from the qualified SFT renderer')
        return {'source': 'completed_sft_exports', 'method': method, 'policies': identities, 'sft_proof': sft}
    result = committed_rl_policy(config, method, checkpoint)
    result.update(initialization_policies=identities, sft_proof=sft)
    return result


def verify_state(expected, actual, role):
    import torch
    if not expected or expected.keys() != actual.keys():
        raise ValueError(f'{role} loaded parameter keys differ from the selected policy')
    for key, value in expected.items():
        other = actual[key]
        if (not isinstance(value, torch.Tensor) or value.shape != other.shape
                or not torch.isfinite(value).all() or not torch.equal(value.cpu(), other.cpu())):
            raise ValueError(f'{role} did not load the selected policy tensor exactly: {key}')
    return {'exact_tensor_equality': True, 'tensors': len(expected), 'elements': sum(value.numel() for value in expected.values())}


def load_evaluated_policy(components, policy):
    import torch
    proofs = list(policy.get('components', {}).values()) + list(policy.get('policies', {}).values())
    if policy.get('training_state'):
        proofs.append(policy['training_state'])
    if policy.get('sft_proof'):
        proofs.append(policy['sft_proof']['actor_state'])
    if any(proof is not None and file_identity(proof['path']) != proof for proof in proofs):
        raise ValueError('An evaluated weight file changed after registration')
    if policy['source'] == 'untrained_base':
        actor = components.actor.adapter_state()['policy']
        renderer = components.renderer.policy_state()
        checks = {'actor': {k: v for k, v in actor.items() if 'lora_B' in k},
            'renderer': {k: v for k, v in renderer.items() if 'lora_B' in k or (k.startswith('control_adapter.') and '.2.' in k)}}
        if any(not state or any(torch.count_nonzero(value).item() for value in state.values()) for state in checks.values()):
            raise ValueError('Untrained adapters have nonzero output projections')
        return {'source': 'untrained_base', 'zero_output_projections_verified': True}
    if policy['source'] == 'committed_rl_pair':
        states = {role: torch.load(proof['path'], map_location='cpu', weights_only=True)
            for role, proof in policy['components'].items()}
        training = torch.load(policy['training_state']['path'], map_location='cpu', weights_only=True)
        if (training['behavior_version'] != policy['commit']['new_version']
                or training['data_cursor']['round'] != policy['completion']['rounds']):
            raise ValueError('RL tensor state does not match its final committed version and round')
        actor = states['actor']['model']['policy']
        renderer = states['renderer']['model']
        components.actor.restore_adapter_state({'policy': actor})
        components.renderer.load_policy_state(renderer)
    else:
        actor = torch.load(policy['sft_proof']['actor_state']['path'], map_location='cpu', weights_only=True)['model']['policy']
        renderer = torch.load(policy['policies']['renderer']['path'], map_location='cpu', weights_only=True)
    return {'source': policy['source'], 'actor': verify_state(actor, components.actor.adapter_state()['policy'], 'actor'),
        'renderer': verify_state(renderer, components.renderer.policy_state(), 'renderer'),
        'training_reference_loaded_for_evaluation': False, 'optimizer_or_rng_state_restored': False}
