import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from uuid import uuid4


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def read_json(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{uuid4().hex}')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def file_identity(path):
    path = Path(path).resolve()
    stat = path.stat()
    return {'path': str(path), 'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns}


def inspect_source(source):
    source = Path(source).resolve()
    registration = read_json(source / 'registration.json')
    tasks = rows(source / 'tasks.jsonl')
    if [task['task_id'] for task in tasks] != registration['selected_tasks']:
        raise ValueError('The shard task order differs from its original registration')
    if len({task['family_id'] for task in tasks}) != len(tasks):
        raise ValueError('The shard contains repeated training families')
    complete, interrupted, pending, record_ids = [], [], [], set()
    for index, task in enumerate(tasks):
        folder = source / 'tasks' / f'{index:04d}'
        completed = folder / 'completed.json'
        if completed.exists():
            saved = read_json(completed)
            if saved['task_id'] != task['task_id'] or saved['family_id'] != task['family_id']:
                raise ValueError('A completed task has a different identity')
            if saved['counts']['failed_tasks']:
                raise ValueError('A completed failed task needs a separate failure investigation')
            bundle = read_json(folder / 'state_bundle.json')
            records = rows(folder / 'actor_records.jsonl')
            if bundle['task_id'] != task['task_id'] or bundle['family_id'] != task['family_id']:
                raise ValueError('A completed state bundle has a different identity')
            if bundle['verified_record_ids'] != [record['record_id'] for record in records]:
                raise ValueError('A completed task has inconsistent admitted records')
            for record in records:
                if record['family_id'] != task['family_id'] or record['record_id'] in record_ids:
                    raise ValueError('Completed records contain a different or repeated family action')
                if not Path(record['snapshot_path']).is_file():
                    raise FileNotFoundError(record['snapshot_path'])
                record_ids.add(record['record_id'])
            complete.append(index)
        elif folder.exists() and any(folder.iterdir()):
            interrupted.append(index)
        else:
            pending.append(index)
    if len(interrupted) != 1 or complete != list(range(interrupted[0])):
        raise ValueError('Recovery requires one interrupted task after a contiguous completed prefix')
    index = interrupted[0]
    if pending != list(range(index + 1, len(tasks))):
        raise ValueError('Later tasks must be unstarted, not silently discarded')
    events_path = source / 'tasks' / f'{index:04d}' / 'events.jsonl'
    if not events_path.is_file():
        raise ValueError('The interrupted task has no saved execution events')
    lines = events_path.read_text().splitlines()
    events, torn_tail = [], False
    for number, line in enumerate(lines):
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            if number != len(lines) - 1:
                raise ValueError('Only an incomplete final event line is recoverable')
            torn_tail = True
            break
        for name in ('public_before_state', 'public_after_state'):
            if event[name]['task'] != tasks[index]:
                raise ValueError('An interrupted event belongs to a different public task')
        events.append(event)
    if not events:
        raise ValueError('No complete original event establishes the interrupted task identity')
    paths = [source / name for name in ('registration.json', 'tasks.jsonl', 'collection.yaml', 'sampler_selection.json')]
    paths.append(events_path)
    for item in complete:
        paths.extend(source / 'tasks' / f'{item:04d}' / name
                     for name in ('completed.json', 'state_bundle.json', 'actor_records.jsonl'))
    return {'source': str(source), 'interrupted_index': index, 'interrupted_task_id': tasks[index]['task_id'],
            'completed_indices': complete, 'unstarted_indices': pending,
            'completed_records_retained': len(record_ids), 'old_partial_records_imported': 0,
            'complete_old_events': len(events), 'torn_final_event_preserved': torn_tail,
            'source_guards': [file_identity(path) for path in paths]}


def check_guards(guards):
    for expected in guards:
        if file_identity(expected['path']) != expected:
            raise RuntimeError('The source attempt changed; its collector must be stopped before recovery')


def validate_inputs(source, original_tasks, selection, private_ready, sampler_selection):
    import yaml

    source = Path(source)
    config = yaml.safe_load((source / 'collection.yaml').read_text())
    registration = read_json(source / 'registration.json')
    inputs = {'task_manifest_sha256': original_tasks, 'training_selection_sha256': selection,
              'private_manifest_sha256': config['data']['private_tasks'], 'private_ready_sha256': private_ready,
              'calibration_sha256': config['judge']['calibration'], 'sampler_selection_sha256': sampler_selection}
    for key, path in inputs.items():
        if digest(path) != registration[key]:
            raise ValueError(f'Original frozen input changed: {key}')
    sampler = read_json(sampler_selection)
    if sampler != read_json(source / 'sampler_selection.json'):
        raise ValueError('The original accepted sampler copy differs')
    if sampler.get('status') != 'accepted' and sampler.get('accepted') is not True:
        raise ValueError('The original sampler was not accepted')
    for key in ('steps', 'eta', 'cfg_scale'):
        if config['renderer'][key] != sampler[key]:
            raise ValueError(f'The original sampler numerical setting changed: {key}')
    for role in ('actor', 'renderer'):
        weight = Path(config['models'][role + '_policy'])
        if weight.is_dir():
            weight /= 'adapter_model.safetensors'
        if file_identity(weight) != registration[role + '_weight_identity']:
            raise ValueError(f'The original {role} policy identity changed')
        if config['models'][role + '_policy'] != registration[role + '_policy']:
            raise ValueError(f'The registered {role} policy path changed')
    return config


def prepare(source, output, inputs, source_stopped, render_reuse='exact'):
    if not source_stopped:
        raise ValueError('Explicit --source-stopped acknowledgement is required; this tool never stops processes')
    if render_reuse not in {'exact', 'none'}:
        raise ValueError('Unknown render reuse policy')
    source, output = Path(source).resolve(), Path(output).resolve()
    if output == source or source in output.parents or output in source.parents:
        raise ValueError('Recovery must use a separate sibling attempt, never overwrite the source')
    plan = inspect_source(source)
    config = validate_inputs(source, **inputs)
    output.mkdir(parents=True, exist_ok=False)
    for name in ('registration.json', 'tasks.jsonl', 'sampler_selection.json'):
        shutil.copyfile(source / name, output / name)
    (output / 'tasks').mkdir()
    for index in plan['completed_indices']:
        (output / 'tasks' / f'{index:04d}').symlink_to(source / 'tasks' / f'{index:04d}', target_is_directory=True)
    import yaml

    config['run_dir'] = str(output)
    (output / 'collection.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    check_guards(plan['source_guards'])
    plan.update(schema='full_actor_recovery_attempt_v1', output=str(output), source_stopped_acknowledged=True,
                frozen_input_paths={key: str(Path(value).resolve()) for key, value in inputs.items()},
                recovery_mode='reexecute_only_interrupted_family_in_new_attempt',
                render_reuse=render_reuse,
                private_scores_reused=False, old_partial_actions_counted=False,
                completed_family_execution_allowed=False, unstarted_families_continue_normally=True)
    atomic_json(output / 'recovery_plan.json', plan)
    return plan


def validate_trace(trace, bundle, seed, job_id, behavior, config, stream_seed):
    expected = {'render_id': job_id, 'bundle_id': bundle.bundle_id, 'behavior_version': behavior,
                'steps': config['renderer']['steps'], 'eta': config['renderer']['eta'],
                'cfg_scale': config['renderer']['cfg_scale'], 'seed_init': stream_seed(seed, 'init'),
                'seed_steps': tuple(stream_seed(seed, index, 'sde') for index in range(config['renderer']['steps']))}
    for key, value in expected.items():
        if getattr(trace, key) != value:
            raise ValueError(f'Saved render trace identity mismatch: {key}')
    for key, value in {'attention_backend': config['renderer']['attention_backend'],
                       'model_revision': config['models']['renderer_revision'],
                       'compiler_version': bundle.compiler_version}.items():
        if trace.metadata.get(key) != value:
            raise ValueError(f'Saved render trace metadata mismatch: {key}')


class RecoveryRenderBridge:
    def __init__(self, fresh, source, config, audit_path, trace_loader, stream_seed, result_type):
        self.fresh = fresh
        self.source = Path(source)
        self.config = config
        self.audit_path = Path(audit_path)
        self.trace_loader = trace_loader
        self.stream_seed = stream_seed
        self.result_type = result_type

    @property
    def behavior_version(self):
        return self.fresh.behavior_version

    @behavior_version.setter
    def behavior_version(self, value):
        self.fresh.behavior_version = value

    def render(self, bundle, seed, job_id):
        safe = job_id.replace(':', '_').replace('/', '_')
        original = self.source / 'renders' / safe
        trace_path = original / 'trace.pt'
        result_path = original / 'result.json'
        mode = 'new_render'
        if result_path.exists() and not trace_path.is_file():
            raise ValueError('Saved render result is missing its original trace')
        if trace_path.is_file():
            guard = file_identity(trace_path)
            trace = self.trace_loader(trace_path)
            validate_trace(trace, bundle, seed, job_id, self.behavior_version, self.config, self.stream_seed)
            del trace
            result = None
            guards = [guard]
            if result_path.is_file():
                guards.append(file_identity(result_path))
                result = read_json(result_path)
                if (result['bundle_id'] != bundle.bundle_id or result['metadata']['seed'] != seed
                        or result['metadata']['behavior_version'] != self.behavior_version
                        or Path(result['trajectory_id']).resolve() != trace_path.resolve()):
                    raise ValueError('Saved render result differs from the exact registered render')
                self.result_type.model_validate(result)
                self.fresh.artifacts.get_ref(result['image_id'])
            check_guards(guards)
            destination = self.fresh.directory / safe
            destination.mkdir(parents=True, exist_ok=True)
            if not (destination / 'trace.pt').exists():
                try:
                    os.link(trace_path, destination / 'trace.pt')
                except OSError:
                    shutil.copyfile(trace_path, destination / 'trace.pt')
            if result is not None:
                result['trajectory_id'] = str((destination / 'trace.pt').resolve())
                atomic_json(destination / 'result.json', result)
                mode = 'exact_saved_result'
            else:
                mode = 'exact_saved_trace_decode'
        elif (self.fresh.directory / safe / 'result.json').is_file():
            mode = 'current_attempt_saved_result'
        elif (self.fresh.directory / safe / 'trace.pt').is_file():
            mode = 'current_attempt_saved_trace_decode'
        result = self.fresh.render(bundle, seed, job_id)
        with self.audit_path.open('a') as stream:
            stream.write(json.dumps({'job_id': job_id, 'bundle_id': bundle.bundle_id, 'seed': seed,
                                    'behavior_version': self.behavior_version, 'mode': mode,
                                    'original_trace': str(trace_path) if mode.startswith('exact_saved_') else None,
                                    'result': result.model_dump(mode='json'),
                                    'fresh_diffusion_executed': mode == 'new_render'}) + '\n')
        return result


def run(output, source_stopped):
    if not source_stopped:
        raise ValueError('Explicit --source-stopped acknowledgement is required')
    output = Path(output).resolve()
    plan = read_json(output / 'recovery_plan.json')
    if plan['output'] != str(output):
        raise ValueError('The recovery attempt moved after preparation')
    check_guards(plan['source_guards'])
    config = validate_inputs(plan['source'], **plan['frozen_input_paths'])
    for name in ('registration.json', 'tasks.jsonl', 'sampler_selection.json'):
        if digest(output / name) != digest(Path(plan['source']) / name):
            raise ValueError('The prepared recovery input changed')
    for index in plan['completed_indices']:
        path = output / 'tasks' / f'{index:04d}'
        expected = Path(plan['source']) / 'tasks' / f'{index:04d}'
        if not path.is_symlink() or path.resolve() != expected.resolve():
            raise ValueError('A completed family reference changed after preparation')
    for index in [plan['interrupted_index'], *plan['unstarted_indices']]:
        path = output / 'tasks' / f'{index:04d}'
        if path.exists() and any(path.iterdir()):
            raise ValueError('This attempt already executed; preserve it and prepare a new recovery attempt')
    from agentic_image.core.types import RenderResult
    from agentic_image.data.audit import require_pair_audit
    from agentic_image.data.interventions import collect_public_rule_interventions
    from agentic_image.learning.sampling import stream_seed
    from agentic_image.render.types import RenderTrace
    from agentic_image.runtime.components import Components

    config['run_dir'] = str(output)
    require_pair_audit(config['data']['paired_audit'], config.get('review_mode', 'human'))
    components = Components(config)
    components.load_all(private=True)
    if plan['render_reuse'] == 'exact':
        components.render_bridge = RecoveryRenderBridge(components.render_bridge, plan['source'], config,
            output / 'render_reuse.jsonl', RenderTrace.load, stream_seed, RenderResult)
    registration = read_json(output / 'registration.json')
    sampler = read_json(output / 'sampler_selection.json')
    report = collect_public_rule_interventions(components, output / 'tasks.jsonl', output,
        registration['renderer_version'], sampler.get('sampler_id') or sampler.get('candidate'),
        max_tasks=len(registration['selected_tasks']), seed=config['seed'])
    records = rows(output / 'actor_records.jsonl')
    if len({record['record_id'] for record in records}) != len(records):
        raise ValueError('Recovery introduced duplicate admitted action records')
    check_guards(plan['source_guards'])
    atomic_json(output / 'completed.json', {'status': 'completed', 'report': report,
        'training_admission_pending': True, 'recovery_source': plan['source'],
        'old_partial_actions_counted': False, 'unique_record_ids': len(records)})
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('inspect', 'prepare', 'run'))
    parser.add_argument('--source')
    parser.add_argument('--output')
    parser.add_argument('--source-stopped', action='store_true')
    parser.add_argument('--render-reuse', choices=('exact', 'none'), default='exact')
    parser.add_argument('--tasks', default='data/public/expanded_training128.tasks.jsonl')
    parser.add_argument('--selection', default='data/training_expansion/training_selection_v1.frozen.jsonl')
    parser.add_argument('--private-ready', default='deployment/training128-private-eval.ready.json')
    parser.add_argument('--sampler-selection', default='runs/sampler-fallback-200/sampler_selection.sde50_eta015.json')
    args = parser.parse_args()
    if args.mode == 'run':
        if not args.output:
            parser.error('--output is required')
        result = run(args.output, args.source_stopped)
    else:
        if not args.source:
            parser.error('--source is required')
        if args.mode == 'inspect':
            result = inspect_source(args.source)
        else:
            if not args.output:
                parser.error('--output is required')
            result = prepare(args.source, args.output, {'original_tasks': args.tasks,
                'selection': args.selection, 'private_ready': args.private_ready,
                'sampler_selection': args.sampler_selection}, args.source_stopped, args.render_reuse)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
