import argparse
import json
import os
from pathlib import Path
import shutil
import time

import torch

from agentic_image.config import load_config
from agentic_image.core.types import PlanSpec, PublicTask
from agentic_image.learning.transaction import atomic_json
from agentic_image.render.types import ConditionBundle
from agentic_image.runtime.components import Components


parser = argparse.ArgumentParser()
parser.add_argument('--config', required=True)
parser.add_argument('--tasks', required=True)
parser.add_argument('--output', required=True)
parser.add_argument('--device', default='cuda:1')
parser.add_argument('--limit', type=int, default=16)
parser.add_argument('--policy')
parser.add_argument('--baseline')
parser.add_argument('--require-exit')
parser.add_argument('--training-metrics')
parser.add_argument('--expected-updates', type=int, default=20)
parser.add_argument('--seed', type=int)
parser.add_argument('--cohort-reservation')
args = parser.parse_args()
config = load_config(args.config)
if args.seed is not None:
    config['seed'] = args.seed
output = Path(args.output)
tasks = [PublicTask.model_validate(json.loads(row)) for row in Path(args.tasks).read_text().splitlines()][:args.limit]
cohort = None
if args.cohort_reservation:
    reserved = [json.loads(row) for row in Path(args.cohort_reservation).read_text().splitlines()]
    if len(reserved) != 64 or len({row['family_id'] for row in reserved}) != 64 or any(row['split_id'] != 'development' for row in reserved):
        raise ValueError('The registered development cohort must contain its original 64 distinct families')
    registered = {'development64:' + row['sample_id']: row for row in reserved}
    if len({task.family_id for task in tasks}) != len(tasks) or not tasks:
        raise ValueError('Each evaluation chunk must contain distinct registered development families')
    for task in tasks:
        row = registered.get(task.task_id)
        if row is None or task.split_id != 'development' or task.family_id != row['family_id'] or task.user_request.text != row['instruction']:
            raise ValueError('The public task is outside its fixed development registration')
    cohort = {'path': str(Path(args.cohort_reservation).resolve()), 'split': 'development', 'fixed_families': 64, 'chunk_families': len(tasks), 'indices': [registered[task.task_id]['evaluation_index'] for task in tasks], 'registered_seeds': [11, 23, 47, 73, 101]}
    if config['seed'] not in cohort['registered_seeds']:
        raise ValueError('The development evaluation seed is outside the fixed five-seed protocol')
policy = None
before = []
training_updates = 0
if bool(args.policy) != bool(args.baseline):
    raise ValueError('A trained policy comparison requires its completed baseline')
if args.policy:
    if not args.require_exit or not args.training_metrics:
        raise ValueError('The comparison requires successful training exit and committed metrics')
    outcome = json.loads(Path(args.require_exit).read_text())
    code = outcome.get('returncode') if isinstance(outcome, dict) else outcome
    if type(code) is not int or code != 0:
        raise RuntimeError('Renderer training has not exited successfully')
    metrics = [json.loads(line) for line in Path(args.training_metrics).read_text().splitlines()]
    training_updates = metrics[-1]['step'] if metrics else 0
    if training_updates != args.expected_updates:
        raise RuntimeError('Renderer training did not commit the expected number of updates')
    policy = Path(args.policy).resolve()
    if policy.is_dir():
        policy = policy / 'policy.pt'
    if not policy.is_file():
        raise FileNotFoundError(policy)
    baseline = Path(args.baseline).resolve()
    if baseline == output.resolve():
        raise ValueError('The trained comparison must not overwrite the baseline')
    baseline_summary = json.loads((baseline / 'summary.json').read_text())
    baseline_metadata = json.loads((baseline / 'metadata.json').read_text())
    before = json.loads((baseline / 'results.json').read_text())
    expected_tasks = len(tasks) if cohort else 16
    if baseline_summary.get('status') != 'completed' or len(before) != expected_tasks or len(tasks) != expected_tasks:
        raise ValueError('The comparison requires all registered baseline tasks')
    if cohort:
        if baseline_metadata.get('development_cohort') != cohort:
            raise ValueError('The baseline development registration differs from the trained comparison')
    elif any(task.split_id != 'official-dev' for task in tasks):
        raise ValueError('Only the registered official-dev tasks may enter this comparison')
    expected = {'steps': 50, 'eta': 0.3, 'cfg_scale': 4.0, 'seed': args.seed if args.seed is not None else 11, 'optimizer_updates': 0}
    if any(baseline_metadata.get(key) != value for key, value in expected.items()):
        raise ValueError('The baseline sampling protocol differs from the registered comparison')
    if baseline_metadata['renderer_revision'] != config['models']['renderer_revision']:
        raise ValueError('The baseline uses a different renderer revision')
    config['seed'] = baseline_metadata['seed']
    for key in ('steps', 'eta', 'cfg_scale'):
        config['renderer'][key] = baseline_metadata[key]
    for index, (task, row) in enumerate(zip(tasks, before)):
        plan = PlanSpec(global_text=task.user_request.text).model_dump(mode='json')
        if row['result']['metadata']['seed'] != expected['seed']:
            raise ValueError(f'Baseline result seed differs at index {index}')
        if row['index'] != index or row['task_id'] != task.task_id or row['family_id'] != task.family_id or row['request'] != task.user_request.text or row['source_image_id'] != task.user_request.source_image_id or row['plan'] != plan:
            raise ValueError(f'Baseline source or original-request PLAN differs at index {index}')
        stem = row['result']['bundle_id'].replace(':', '_')
        for suffix in ('.pt', '.json'):
            if not (baseline / 'bundles' / (stem + suffix)).is_file():
                raise FileNotFoundError(baseline / 'bundles' / (stem + suffix))
        recipe = json.loads((baseline / 'bundles' / (stem + '.json')).read_text())
        if recipe['plan'] != plan or recipe['source_image_id'] != task.user_request.source_image_id:
            raise ValueError(f'Saved source recipe differs at index {index}')
config['run_dir'] = str(output)
config['devices']['renderer'] = args.device
config['devices']['encoders'] = args.device
config['models']['renderer_policy'] = str(policy) if policy else None
config['models']['renderer_reference'] = None
output.mkdir(parents=True, exist_ok=True)
(output / 'previews').mkdir(exist_ok=True)
metadata = {'scope': 'trained_renderer_original_request_comparison' if policy else 'untrained_renderer_original_request_baseline', 'tasks': [task.task_id for task in tasks], 'steps': config['renderer']['steps'], 'eta': config['renderer']['eta'], 'cfg_scale': config['renderer']['cfg_scale'], 'seed': config['seed'], 'renderer_revision': config['models']['renderer_revision'], 'optimizer_updates': training_updates}
if cohort:
    metadata['development_cohort'] = cohort
if policy:
    metadata.update(policy_path=str(policy), policy_bytes=policy.stat().st_size, policy_mtime_ns=policy.stat().st_mtime_ns, baseline=str(baseline), fixed_condition_bundles=True, evaluation_optimizer_updates=0, calibrated_evaluation=False, human_reviewed=False)
if (output / 'metadata.json').exists() and json.loads((output / 'metadata.json').read_text()) != metadata:
    raise ValueError('Existing render registration differs')
atomic_json(output / 'metadata.json', metadata)
records_path = output / 'results.json'
records = json.loads(records_path.read_text()) if records_path.exists() else []
if any(row['index'] != index or row['task_id'] != tasks[index].task_id for index, row in enumerate(records)):
    raise ValueError('Existing render results do not match the registered order')
components = Components(config)
components.load_renderer()
behavior = f'renderer-global-ai:step{training_updates}' if policy else 'initial'
components.render_bridge.behavior_version = behavior
if policy:
    supplied = torch.load(policy, map_location='cpu', weights_only=True)
    loaded = components.renderer.policy_state()
    if supplied.keys() != loaded.keys() or any(not torch.equal(supplied[key], loaded[key]) for key in supplied):
        raise RuntimeError('Renderer policy weights were not loaded exactly')
    changed_b = sum(int(torch.count_nonzero(value)) for key, value in supplied.items() if 'lora_B' in key)
    if not changed_b:
        raise RuntimeError('The trained renderer has no nonzero LoRA B parameters')
    atomic_json(output / 'policy_load.json', {'path': str(policy), 'exact_tensor_equality': True, 'tensor_count': len(supplied), 'nonzero_lora_b_elements': changed_b, 'training_updates': training_updates})
    del supplied, loaded
components.record_memory('comparison_loaded' if policy else 'baseline_loaded')
for index, task in enumerate(tasks):
    if index < len(records):
        if not (output / f'{index:03d}.png').is_file():
            raise RuntimeError('A completed render is missing its original PNG')
        continue
    started = time.monotonic()
    plan = PlanSpec(global_text=task.user_request.text)
    if policy:
        prior = before[index]
        stem = prior['result']['bundle_id'].replace(':', '_')
        payload = output / 'bundles' / (stem + '.pt')
        if not payload.exists():
            os.link(baseline / 'bundles' / (stem + '.pt'), payload)
            shutil.copy2(baseline / 'bundles' / (stem + '.json'), payload.with_suffix('.json'))
        bundle = ConditionBundle.load(payload)
        components.compiler.validate_bundle(bundle)
        bindings = bundle.artifact_bindings
        slots = bindings.get('reference_slots', [])
        if bundle.bundle_id != prior['result']['bundle_id'] or bundle.region_count != 0 or bundle.canvas_hw != tuple(config['canvas']) or bundle.cfg_scale != config['renderer']['cfg_scale'] or bindings['plan'] != plan.model_dump(mode='json') or bindings['source_image_id'] != task.user_request.source_image_id or len(slots) != 1 or slots[0]['role'] != 'source' or slots[0]['image_id'] != task.user_request.source_image_id:
            raise ValueError('The saved baseline condition does not contain exactly the fixed public source and original request')
    else:
        bundle = components.compiler.compile(plan, task.user_request.source_image_id)
    result = components.render_bridge.render(bundle, config['seed'], f'{"after" if policy else "baseline"}:{index:03d}')
    image_path = output / f'{index:03d}.png'
    image = components.artifacts.get_image(result.image_id)
    image.save(image_path)
    preview = output / 'previews' / f'{index:03d}.jpg'
    image.save(preview, quality=80, optimize=True)
    record = {'index': index, 'task_id': task.task_id, 'family_id': task.family_id, 'request': task.user_request.text, 'source_image_id': task.user_request.source_image_id, 'plan': plan.model_dump(mode='json'), 'result': result.model_dump(mode='json'), 'image': str(image_path.resolve()), 'preview': str(preview.resolve()), 'seconds': time.monotonic() - started}
    if policy:
        record.update(before_image=before[index]['image'], before_bundle_id=before[index]['result']['bundle_id'], source_condition_reused_exactly=True)
    records.append(record)
    atomic_json(records_path, records)
    atomic_json(output / 'progress.json', {'completed': len(records), 'total': len(tasks), 'last_seconds': record['seconds'], 'updated_at': time.time()})
    components.record_memory(f'{"comparison" if policy else "baseline"}:{index + 1}')
    print(json.dumps(record), flush=True)
summary = {**metadata, 'completed': len(records), 'render_seconds': sum(row['seconds'] for row in records), 'peak_reserved_gib': torch.cuda.max_memory_reserved(args.device) / 2**30, 'status': 'completed'}
atomic_json(output / 'summary.json', summary)
atomic_json(output / 'completed.json', {'status': 'completed', 'summary': str((output / 'summary.json').resolve())})
print(json.dumps(summary), flush=True)
