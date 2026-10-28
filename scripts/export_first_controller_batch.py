import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import tarfile
import time

from analyze_controller_evaluation import METHODS, COMPARISONS, SEEDS, comparison, registered_tasks, validate_episode
from sample_delivery_compression import DigestSink, low_priority


EVALUATION = Path('runs/development64-controller-episodes')
TARGETS = Path('delivery/first-batch-targets')
INDICES = tuple(range(8))
SEED = 11


def regular(root, path):
    path = Path(path)
    path = path if path.is_absolute() else root / path
    if '..' in path.parts or not path.is_relative_to(root):
        raise ValueError('A dependency escaped its project: ' + str(path))
    for parent in (path, *path.parents):
        if parent == root.parent:
            break
        if parent.is_symlink():
            raise ValueError('Dependency symlinks are not supported: ' + str(path))
    info = path.stat()
    if not stat.S_ISREG(info.st_mode) or info.st_size == 0:
        raise ValueError('A nonempty regular dependency is required: ' + str(path))
    return path


def identity(path):
    value = path.stat()
    return {'bytes': value.st_size, 'mtime_ns': value.st_mtime_ns, 'inode': value.st_ino, 'device': value.st_dev}


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def tensor_metadata(path):
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import torch
    torch.set_num_threads(1)
    return torch.load(path, map_location='cpu', weights_only=True, mmap=True)


def artifact_ids(value):
    if isinstance(value, str):
        return {value} if re.fullmatch(r'artifact:[a-f0-9]{64}', value) else set()
    if isinstance(value, dict):
        return set().union(*(artifact_ids(item) for pair in value.items() for item in pair))
    if isinstance(value, (list, tuple)):
        return set().union(*(artifact_ids(item) for item in value))
    return set()


class Sources:
    def __init__(self, root):
        self.root = root
        self.files = {}
        self.missing = []

    def add(self, path, category):
        path = Path(path)
        path = path if path.is_absolute() else self.root / path
        try:
            path = regular(self.root, path)
        except FileNotFoundError:
            self.missing.append({'path': str(path), 'category': category})
            return None
        self.files[str(path)] = {'archive_path': 'project/' + str(path.relative_to(self.root)),
            'category': category, **identity(path)}
        return path

    def read(self, path, category):
        found = self.add(path, category)
        return json.loads(found.read_text()) if found else None

    def asset(self, artifact_id, role):
        if not re.fullmatch(r'artifact:[a-f0-9]{64}', artifact_id):
            raise ValueError('Invalid artifact identity')
        relative = Path('data/artifacts/refs') / (artifact_id.split(':')[1] + '.json')
        ref = self.read(relative, 'artifact_reference')
        if ref is None:
            return {'artifact_id': artifact_id, 'role': role, 'status': 'missing_reference'}
        payload = Path(ref['relative_path'])
        if ref.get('artifact_id') != artifact_id or payload.is_absolute() or '..' in payload.parts or payload.parts[0] != 'values':
            raise ValueError('Artifact reference identity or payload path differs')
        path = self.add(Path('data/artifacts') / payload, 'artifact_payload')
        return {'artifact_id': artifact_id, 'value_id': ref.get('value_id'), 'role': role, 'kind': ref.get('kind'),
            'reference': 'project/' + str(relative), 'payload': 'project/data/artifacts/' + str(payload),
            'status': 'available' if path else 'missing_payload'}


def prepare_targets(root):
    root = Path(root).resolve()
    reserved = lines(regular(root, 'data/evaluation/development64/reserved.frozen.jsonl'))
    prepared = lines(regular(root, 'data/evaluation/development64/prepared.jsonl'))
    by_index = {}
    for row in prepared:
        index = row['evaluation_index']
        if index in by_index:
            raise ValueError('Duplicate prepared evaluation index')
        by_index[index] = row
    files, selected = {}, []
    for index in INDICES:
        row, original = by_index[index], reserved[index]
        if (row['family_id'] != original['family_id'] or row['sample_id'] != original['sample_id']
                or row['evaluation_index'] != index or row['split_id'] != 'development'):
            raise ValueError('Prepared target identity differs from the fixed reservation')
        targets = [item for item in row['images'] if item['candidate_role'] == 'target']
        if len(targets) != 1:
            raise ValueError('Exactly one prepared target is required per family')
        target = targets[0]
        ref_path = regular(root, Path('data/artifacts/refs') / (target['artifact_id'].split(':')[1] + '.json'))
        ref = json.loads(ref_path.read_text())
        metadata = ref['metadata']
        if (target['original_artifact_id'] != original['target_image_id'] or ref['artifact_id'] != target['artifact_id']
                or ref['value_id'] != target['value_id'] or ref['relative_path'] != target['relative_path']
                or metadata.get('original_artifact_id') != original['target_image_id']
                or metadata.get('sample_id') != row['sample_id'] or metadata.get('role') != 'private_target'
                or metadata.get('dataset_profile') != 'official-viewer-reencoded-v1'
                or metadata.get('viewer_revision') != target['viewer_revision']):
            raise ValueError('Target provenance differs from its canonical reference')
        payload = Path(ref['relative_path'])
        if payload.is_absolute() or '..' in payload.parts or payload.parts[0] != 'values':
            raise ValueError('Target payload must be a relative artifact value')
        value_path = regular(root, Path('data/artifacts') / payload)
        ref_name = Path('refs') / ref_path.name
        for source, destination in [(ref_path, ref_name), (value_path, payload)]:
            files[str(source)] = {'archive_path': str(TARGETS / destination), 'category': 'private_target', **identity(source)}
        selected.append({'index': index, 'sample_id': row['sample_id'], 'family_id': row['family_id'],
            'original_target_artifact_id': target['original_artifact_id'], 'original_target_value_id': target['original_value_id'],
            'artifact_id': target['artifact_id'], 'value_id': target['value_id'], 'reference_path': str(ref_name),
            'payload_path': str(payload), 'payload_bytes': value_path.stat().st_size, 'viewer_revision': target['viewer_revision'],
            'original_revision': metadata['original_revision'], 'reencoded': True})
    manifest = {'schema': 'first_batch_private_targets_v1', 'indices': list(INDICES), 'seed': SEED,
        'source_prepared_manifest': 'data/evaluation/development64/prepared.jsonl', 'targets': selected,
        'scope': 'Existing official-viewer reencoded targets for delivery only; not original-byte equivalence, training inputs or new evaluation.'}
    return {'status': 'ready', 'source_bytes': sum(row['bytes'] for row in files.values()), 'files': files,
        'generated': {str(TARGETS / 'targets.json'): manifest}, 'manifest': manifest}


def collect_plan(root, target_root=TARGETS, loader=tensor_metadata):
    root = Path(root).resolve()
    target_root = Path(target_root)
    if target_root.is_absolute() or '..' in target_root.parts:
        raise ValueError('Target staging must be a relative project directory')
    sources = Sources(root)
    tasks_path = sources.add('data/evaluation/development64/public_tasks.jsonl', 'frozen_registration')
    reservation_path = sources.add('data/evaluation/development64/reserved.frozen.jsonl', 'frozen_registration')
    training_path = sources.add('data/public/expanded_training128.tasks.jsonl', 'training_isolation_manifest')
    private_path = sources.add('data/private_eval_v1/development64/private_tasks.jsonl', 'private_evaluation')
    if not all([tasks_path, reservation_path, training_path, private_path]):
        raise ValueError('Frozen task, reservation, training and private manifests are required')
    tasks, reserved, private = lines(tasks_path), lines(reservation_path), lines(private_path)
    isolation = registered_tasks(tasks, reserved, [lines(training_path)])
    private_by_id = {row['task_id']: row for row in private}
    if len(private_by_id) != len(private):
        raise ValueError('Duplicate private task identity')
    common = sources.read(EVALUATION / 'common_protocol.json', 'common_protocol')
    if (not common or common.get('tasks') != tasks or common.get('image_seeds') != list(SEEDS)
            or common.get('private_target_used_for_actor_actions') is not False
            or type(common.get('training_updates_during_evaluation')) is not int or common['training_updates_during_evaluation'] != 0):
        raise ValueError('Frozen isolated evaluation protocol is required')
    targets = sources.read(target_root / 'targets.json', 'target_provenance')
    target_rows = {row['index']: row for row in targets['targets']} if targets else {}
    if targets and (targets.get('schema') != 'first_batch_private_targets_v1' or targets.get('indices') != list(INDICES)
            or targets.get('seed') != SEED or len(target_rows) != 8):
        raise ValueError('Private target input scope differs')
    assets, closure, observed, policies = [], [], {}, {}
    for index in INDICES:
        task, original = tasks[index], reserved[index]
        evaluation = private_by_id.get(task['task_id'])
        if not evaluation or evaluation['family_id'] != task['family_id']:
            raise ValueError('Private task identity differs')
        public_source = task['user_request']['source_image_id']
        if evaluation.get('source_image_id') != public_source:
            raise ValueError('Public and private evaluation source identities differ')
        roles = {'source': public_source, 'protected_mask': evaluation['evaluation'].get('original_protected_mask_id'),
            'original_edit_mask': original.get('edit_mask_id')}
        record = {'index': index, 'task_id': task['task_id'], 'family_id': task['family_id'], 'assets': {}}
        for role, artifact in roles.items():
            record['assets'][role] = sources.asset(artifact, role) if artifact else {'status': 'not_provided', 'role': role}
        target = target_rows.get(index)
        if target:
            if (target['family_id'] != task['family_id'] or target['sample_id'] != original['sample_id']
                    or target['original_target_artifact_id'] != original['target_image_id'] or target.get('reencoded') is not True):
                raise ValueError('Private target does not match the fixed family')
            ref = sources.read(target_root / target['reference_path'], 'target_reference')
            payload = sources.add(target_root / target['payload_path'], 'target_payload')
            if ref and (ref.get('artifact_id') != target['artifact_id'] or ref.get('value_id') != target['value_id']
                    or ref.get('relative_path') != target['payload_path'] or ref.get('metadata', {}).get('role') != 'private_target'):
                raise ValueError('Private target reference differs from its delivery manifest')
            if payload and payload.stat().st_size != target['payload_bytes']:
                raise ValueError('Private target size differs from its delivery manifest')
            record['assets']['target'] = {**target, 'payload': 'project/' + str(target_root / target['payload_path']),
                'status': 'available' if ref and payload else 'missing_payload'}
        else:
            sources.missing.append({'path': str(target_root / 'targets.json'), 'index': index, 'category': 'private_target'})
            record['assets']['target'] = {'status': 'missing_target'}
        assets.append(record)
    for method in METHODS:
        rows = {}
        registration = sources.read(EVALUATION / method / 'registration.json', 'policy_registration')
        policy_source = {'base': 'untrained_base', 'sft': 'completed_sft_exports'}.get(method, 'committed_rl_pair')
        if registration:
            policy = registration.get('evaluated_policy', {})
            if (registration.get('method') != method or registration.get('common_protocol') != common
                    or policy.get('method') != method or policy.get('source') != policy_source):
                raise ValueError('Policy registration differs: ' + method)
            policies[method] = policy
        worker = root / EVALUATION / method / 'workers/000_seed11'
        for index in INDICES:
            folder = root / EVALUATION / method / 'episodes' / f'{index:03d}' / 'seed11'
            episode = sources.read(folder / 'episode.json', 'episode')
            if episode is None:
                continue
            if not registration:
                raise ValueError('Saved episode has no matching registration')
            key = validate_episode(episode, method, Path(f'{index:03d}/seed11/episode.json'), tasks, common['sampler_protocol']['parameters'])
            state = sources.read(folder / 'state.json', 'episode_state')
            terminal = sources.read(folder / 'terminal_private.json', 'private_score')
            if state and (state.get('done') is not True or state['task']['task_id'] != episode['task_id']
                    or state['task']['family_id'] != episode['family_id']
                    or state.get('delivered_image_id') != episode['delivered_image_id']):
                raise ValueError('Episode state is unfinished or differs from the saved result')
            if terminal and terminal != episode['terminal']:
                raise ValueError('Saved private terminal evidence differs')
            record = episode['episode_record']
            if Path(record['environment_path']) != folder / 'state.json' or len(record['action_paths']) != episode['budget']['actions']:
                raise ValueError('Episode action or state closure differs')
            entry = {'method': method, 'index': index, 'seed': SEED, 'action_files': [], 'actions': [], 'visual_files': [], 'renders': []}
            needed = artifact_ids(tasks[index]['user_request']) | ({episode['delivered_image_id']} - {None})
            if state:
                needed |= artifact_ids(list(state.get('image_nodes', {})))
            for number, name in enumerate(record['action_paths']):
                path = Path(name)
                if path != folder / f'action_{number:02d}.pt':
                    raise ValueError('Action dependency is outside its selected episode')
                found = sources.add(path, 'action_replay')
                sources.add(folder / f'result_{number:02d}.json', 'action_result')
                sources.add(folder / f'cost_{number:02d}.json', 'action_cost')
                if not found:
                    continue
                action = loader(found)
                entry['action_files'].append(str(path.relative_to(root)))
                entry['actions'].append({key: action.get(key) for key in ('text', 'finish_reason', 'seed', 'behavior_version')})
                snapshot = action.get('snapshot', {})
                needed |= artifact_ids(snapshot.get('image_artifact_ids', []))
                for visual_name in snapshot.get('visual_files', []):
                    visual = Path(visual_name)
                    if not visual.is_relative_to(worker / 'actor_images'):
                        raise ValueError('Action visual dependency is outside its selected worker')
                    sources.add(visual, 'actor_visual_replay')
                    entry['visual_files'].append(str(visual.relative_to(root)))
            if len(record['render_records']) != episode['budget']['renders'] or len(record['render_records']) > 3:
                raise ValueError('Render dependency count differs from its recorded budget')
            for render in record['render_records']:
                trace = Path(render['trajectory_id'])
                if trace.name != 'trace.pt' or trace.parent.parent != worker / 'renders':
                    raise ValueError('Render trace is outside its selected worker')
                bundle_id = render['bundle_id']
                if not re.fullmatch(r'bundle:[a-f0-9]{64}', bundle_id):
                    raise ValueError('Invalid frozen condition identity')
                bundle = worker / 'bundles' / ('bundle_' + bundle_id.split(':')[1] + '.pt')
                found_trace, found_bundle = sources.add(trace, 'render_trace'), sources.add(bundle, 'condition_tensor')
                recipe = sources.read(bundle.with_suffix('.json'), 'condition_recipe')
                result = sources.read(trace.with_name('result.json'), 'render_result')
                if result and result != render:
                    raise ValueError('Render result differs from the selected episode')
                if recipe and recipe.get('bundle_id') != bundle_id:
                    raise ValueError('Frozen condition recipe identity differs')
                if found_trace:
                    trace_data = loader(found_trace)
                    if trace_data.get('bundle_id') != bundle_id or trace_data.get('behavior_version') != render['metadata']['behavior_version']:
                        raise ValueError('Trace identity or evaluated policy differs')
                if found_bundle:
                    bundle_data = loader(found_bundle)
                    if bundle_data.get('bundle_id') != bundle_id:
                        raise ValueError('Frozen condition tensor identity differs')
                    needed |= artifact_ids(bundle_data.get('artifact_bindings', {}))
                needed.add(render['image_id'])
                entry['renders'].append({'trace': str(trace.relative_to(root)), 'bundle': str(bundle.relative_to(root)),
                    'recipe': str(bundle.with_suffix('.json').relative_to(root)), 'bundle_id': bundle_id})
            entry['assets'] = [sources.asset(value, 'episode_dependency') for value in sorted(needed)]
            closure.append(entry)
            rows[key] = episode
        if rows:
            sources.add(worker / 'policy_load.json', 'policy_load_proof')
        observed[method] = rows
    gates, proofs = {}, {}
    requirements = [('deployment/development64-controller-joint_rl-000-seed11.exit.json', {'returncode': 0}),
        ('deployment/env-restored-gpu-check.exit.json', {'returncode': 0}),
        ('deployment/env-restored-gpu-check.json', {'status': 'verified', 'gpu_verified': True, 'checkpoint_recovery_verified': True}),
        ('deployment/full-joint-rl32-export.json', {'status': 'completed', 'stage': 'joint', 'effective_target_met': True})]
    for name, expected in requirements:
        value = sources.read(name, 'completion_proof')
        proofs[name] = value
        gates[name] = bool(value and all(type(value.get(k)) is type(v) and value[k] == v for k, v in expected.items()))
    exported = proofs['deployment/full-joint-rl32-export.json']
    recovered = proofs['deployment/env-restored-gpu-check.json']
    final_policy = policies.get('joint_rl')
    if exported and final_policy:
        if (str(Path(final_policy['commit_path']).parent) != exported.get('checkpoint')
                or exported.get('required_effective_updates') != 32
                or final_policy.get('commit', {}).get('status') != 'COMMITTED'
                or final_policy.get('completion', {}).get('effective_target_met') is not True):
            raise ValueError('Joint evaluation and final exported checkpoint identities differ')
    if recovered and exported and recovered.get('recovery_archive') != exported.get('source'):
        raise ValueError('Restored GPU check used a different recovery archive')
    configurations = {}
    for method, rows in observed.items():
        values = list(rows.values())
        configurations[method] = {'episodes': len(values), 'required_first_batch_episodes': 8,
            'automatic_successes': sum(row['terminal']['success'] for row in values),
            'automatic_success_rate': sum(row['terminal']['success'] for row in values) / 8 if len(values) == 8 else None,
            'protocol_failures': sum(row['protocol_failure'] for row in values),
            'missing_delivered_images': sum(row['delivered_image_id'] is None for row in values),
            'episodes_with_unknown_atoms': sum(any(not atom['known'] for atom in row['terminal']['atoms']) for row in values)}
    source_bytes = sum(row['bytes'] for row in sources.files.values())
    ready = not sources.missing and all(gates.values()) and all(len(rows) == 8 for rows in observed.values())
    summary = {'schema': 'first_controller_batch_summary_v1', 'scope': 'Fixed indices000-007, seed11, four policies; 8x4 evaluation episodes, not the full1280 benchmark or new training.',
        'families': [task['family_id'] for task in tasks[:8]], 'seed': SEED, 'configurations': configurations,
        'comparisons': {after + '_minus_' + before: comparison(observed[before], observed[after], 20000, 20261028)
            for before, after in COMPARISONS}, 'human_reviewed': False, 'visual_review_performed_by_exporter': False,
        'limitations': ['Automatic judgments retain unknowns and failures; no rescoring is performed.',
            'Intervals resample only these eight selected families and do not estimate the full registered benchmark.',
            'Protected masks are native arrays for preserved regions, not target-edit masks.',
            'Private targets are existing official-viewer reencodings; original-byte equivalence is not claimed.',
            'Serialized replay paths retain the original project prefix and require that layout on restored Linux.']}
    return {'schema': 'first_controller_batch_export_v1', 'status': 'ready' if ready else 'not_ready',
        'observed_at': datetime.now(timezone.utc).isoformat(), 'project': str(root), 'indices': list(INDICES), 'seed': SEED,
        'expected_episodes': 32, 'observed_episodes': sum(len(rows) for rows in observed.values()),
        'full1280_complete': False, 'gpu_work_started': False, 'source_bytes': source_bytes,
        'bytes_by_category': dict(Counter({category: sum(item['bytes'] for item in sources.files.values() if item['category'] == category)
            for category in {item['category'] for item in sources.files.values()}})),
        'files': sources.files, 'unresolved_dependencies': sources.missing, 'required_gates': gates,
        'training_manifest_isolation': isolation, 'assets': assets, 'selected_replay_dependency_closure': closure,
        'evaluated_policies': policies, 'summary': summary,
        'external_dependencies': ['Previously delivered exact base models, Linux environments and current project code.',
            'SFT policy packages and Agent/Joint final recoverable weight packages; their exact identities are recorded in evaluated_policies.',
            'Compatible Linux host NVIDIA driver and original project path; no new GPU replay is claimed by this export.']}


def write_archive(plan, output, manifest, rate_mib=8, maximum_bytes=2 * 2**30, lower_priority=True):
    if plan['status'] != 'ready':
        raise RuntimeError('First batch is not ready; inspect unresolved dependencies and required gates')
    if not math.isfinite(rate_mib) or not 0 < rate_mib <= 32 or plan['source_bytes'] > maximum_bytes:
        raise ValueError('Archive exceeds its bounded size or read-rate limit')
    output, manifest = Path(output), Path(manifest)
    partial = output.with_name(output.name + '.part')
    if any(path.exists() for path in (output, partial, manifest)):
        raise FileExistsError('Archive, partial output or manifest already exists')
    files = plan['files']
    if any(identity(Path(path)) != {key: record[key] for key in ('bytes', 'mtime_ns', 'inode', 'device')} for path, record in files.items()):
        raise RuntimeError('A selected source changed after planning')
    generated = plan.get('generated') or {'FIRST_BATCH/manifest.json': plan, 'FIRST_BATCH/summary.json': plan['summary']}
    output.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output.parent).free < plan['source_bytes'] + 16 * 2**20:
        raise RuntimeError('Insufficient disk space for the bounded first-batch archive')
    if lower_priority:
        low_priority()
    sink = DigestSink(partial)
    started, consumed = time.monotonic(), [0]
    class Reader:
        def __init__(self, stream):
            self.stream = stream
        def read(self, size):
            data = self.stream.read(size)
            consumed[0] += len(data)
            delay = consumed[0] / (rate_mib * 2**20) - (time.monotonic() - started)
            if delay > 0:
                time.sleep(delay)
            return data
    try:
        with gzip.GzipFile(filename='', mode='wb', fileobj=sink, compresslevel=1, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode='w|', copybufsize=2**20) as archive:
                for path, record in sorted(files.items()):
                    info = archive.gettarinfo(path, arcname=record['archive_path'])
                    with Path(path).open('rb') as stream:
                        archive.addfile(info, Reader(stream))
                for name, value in generated.items():
                    payload = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode()
                    info = tarfile.TarInfo(name)
                    info.size = len(payload)
                    archive.addfile(info, io.BytesIO(payload))
        sink.finish()
        if any(identity(Path(path)) != {key: record[key] for key in ('bytes', 'mtime_ns', 'inode', 'device')} for path, record in files.items()):
            raise RuntimeError('A selected source changed while packaging')
        partial.replace(output)
        result = {'schema': 'first_batch_archive_receipt_v1', 'status': 'completed', 'archive': str(output.resolve()),
            'source': str(output.resolve()), 'mtime': int(output.stat().st_mtime),
            'bytes': sink.size, 'sha256': sink.digest.hexdigest(), 'source_bytes': consumed[0], 'files': len(files),
            'elapsed_seconds': time.monotonic() - started, 'read_mib_per_second_limit': rate_mib,
            'source_files_unchanged': True, 'full1280_complete': False, 'gpu_work_started': False,
            'scope': plan.get('summary', {}).get('scope', plan.get('manifest', {}).get('scope'))}
        manifest.parent.mkdir(parents=True, exist_ok=True)
        with manifest.open('x') as stream:
            json.dump(result, stream, indent=2)
            stream.write('\n')
        return result
    except BaseException:
        sink.stream.close()
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('prepare-targets', 'first-batch'))
    parser.add_argument('--project', type=Path, default=Path('/workspace/agentic-image-pilot-small'))
    parser.add_argument('--target-root', type=Path, default=TARGETS)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--read-mib-per-second', type=float, default=8)
    args = parser.parse_args()
    plan = prepare_targets(args.project) if args.stage == 'prepare-targets' else collect_plan(args.project, args.target_root)
    if args.execute:
        if not args.output or not args.manifest:
            parser.error('Execution requires new output and manifest paths')
        print(json.dumps(write_archive(plan, args.output, args.manifest, args.read_mib_per_second)))
    else:
        print(json.dumps(plan, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
