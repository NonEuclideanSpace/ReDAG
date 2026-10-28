import argparse
import copy
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import time


def recovery_plan(project, archive, manifest):
    project, archive, manifest = Path(project).resolve(), Path(archive), Path(manifest)
    for path in (archive, manifest):
        if not path.is_file() or path.is_symlink():
            raise ValueError('Recovery inputs must be existing regular files')
    record = json.loads(manifest.read_text())
    if (record.get('schema') != 'rl_recovery_export_v1' or record.get('status') != 'completed'
            or record.get('stage') not in ('agent', 'joint') or record.get('effective_target_met') is not True
            or record.get('source') != str(archive) or archive.stat().st_size != record.get('bytes')):
        raise ValueError('A completed matching RL recovery export is required')
    expected = {}
    for name, info in record['files'].items():
        path = Path(name)
        if not path.is_absolute() or '..' in path.parts or not path.is_relative_to(project):
            raise ValueError('Recovery member is outside its recorded project')
        expected[str(path.relative_to(project))] = info['bytes']
    checkpoint = Path(record['checkpoint'])
    if not checkpoint.is_relative_to(project) or '..' in checkpoint.parts:
        raise ValueError('Recovery checkpoint is outside its recorded project')
    stage = checkpoint.parent.parent
    if stage.name != 'rl_' + record['stage']:
        raise ValueError('Recovery stage and checkpoint differ')
    for name in ('COMMITTED.json', 'actor.pt', 'renderer.pt', 'training_state.pt'):
        if str((checkpoint / name).relative_to(project)) not in expected:
            raise ValueError('Recovery export is missing a checkpoint member')
    config = str((stage / 'committed_restore_config.json').relative_to(project))
    expected[config] = None
    return {'project': str(project), 'archive': str(archive), 'manifest': record,
        'expected_members': expected, 'checkpoint_relative': str(checkpoint.relative_to(project)),
        'stage_relative': str(stage.relative_to(project)), 'config_relative': config}


def extract_recovery(plan, destination, rate=16 * 2**20):
    destination = Path(destination)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError('Recovery extraction must use a new directory')
    if not 0 < rate <= 32 * 2**20:
        raise ValueError('Recovery extraction rate must be positive and at most32 MiB/s')
    expected = plan['expected_members']
    required = sum(size or 0 for size in expected.values()) + 2**20
    if shutil.disk_usage(destination.parent).free < required:
        raise RuntimeError('Insufficient space for an independent checkpoint extraction')
    destination.mkdir()
    digest, restored, total, started = hashlib.sha256(), set(), 0, time.monotonic()
    class Reader:
        def __init__(self, stream):
            self.stream = stream
        def read(self, size):
            data = self.stream.read(size)
            digest.update(data)
            return data
    with Path(plan['archive']).open('rb') as source:
        reader = Reader(source)
        with tarfile.open(fileobj=reader, mode='r|gz') as archive:
            for member in archive:
                name = PurePosixPath(member.name)
                if name.is_absolute() or '..' in name.parts or str(name) not in expected or str(name) in restored:
                    raise ValueError('Recovery archive has an unexpected or duplicate member')
                size = expected[str(name)]
                if not member.isfile() or (size is not None and size != member.size) or (size is None and member.size > 2**20):
                    raise ValueError('Recovery member type or byte count differs')
                target = destination / str(name)
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as incoming, target.open('xb') as outgoing:
                    while data := incoming.read(2**20):
                        outgoing.write(data)
                        total += len(data)
                        delay = total / rate - (time.monotonic() - started)
                        if delay > 0:
                            time.sleep(delay)
                restored.add(str(name))
        while reader.read(2**20):
            pass
    if restored != set(expected) or digest.hexdigest() != plan['manifest']['sha256']:
        raise ValueError('Extracted recovery archive is incomplete or its stream checksum differs')
    config = json.loads((destination / plan['config_relative']).read_text())
    if config != plan['manifest']['committed_config']:
        raise ValueError('Extracted committed configuration differs from its export')
    return {'status': 'extracted', 'archive': plan['archive'], 'sha256': digest.hexdigest(),
        'archive_bytes': plan['manifest']['bytes'], 'extracted_bytes': total, 'files': len(restored),
        'destination': str(destination), 'checkpoint': str(destination / plan['checkpoint_relative']),
        'stage': str(destination / plan['stage_relative']), 'config': config}


def validate_recovery_devices(config, visible_device_count):
    devices = config['devices']
    if visible_device_count != 2 or any(devices.get(key) not in ('cuda:0', 'cuda:1') for key in ('actor', 'renderer', 'encoders', 'judge')):
        raise ValueError('Recovery requires both original CUDA devices and their recorded assignments')
    if devices['actor'] == devices['renderer']:
        raise ValueError('Expected the recorded controller and renderer to use different GPUs')
    return dict(devices)


def equal_state(actual, expected, label):
    import torch
    if isinstance(expected, torch.Tensor):
        if not isinstance(actual, torch.Tensor) or actual.dtype != expected.dtype or not torch.equal(actual.cpu(), expected.cpu()):
            raise ValueError('Restored tensor differs: ' + label)
    elif isinstance(expected, dict):
        if not isinstance(actual, dict) or actual.keys() != expected.keys():
            raise ValueError('Restored state keys differ: ' + label)
        for key in expected:
            equal_state(actual[key], expected[key], label + '.' + str(key))
    elif isinstance(expected, (list, tuple)):
        if type(actual) is not type(expected) or len(actual) != len(expected):
            raise ValueError('Restored sequence differs: ' + label)
        for index, value in enumerate(expected):
            equal_state(actual[index], value, label + '.' + str(index))
    elif actual != expected:
        raise ValueError('Restored value differs: ' + label)


def restored_config(request):
    config = copy.deepcopy(request['config'])
    recovered = Path(request['recovered_root'])
    original = Path(request['original_project'])
    for key in ('actor_policy', 'actor_reference', 'renderer_policy', 'renderer_reference'):
        path = Path(config['models'][key])
        if '..' in path.parts:
            raise ValueError('Recovered adapter paths must stay inside the extracted project')
        relative = path.relative_to(original) if path.is_absolute() else path
        config['models'][key] = str(recovered / relative)
    config['models']['actor_nf4'] = request['actor_nf4']
    config['models']['renderer'] = request['renderer_base']
    config['run_dir'] = str(Path(request['output']) / 'recovered-preflight')
    config['data']['artifacts'] = str(Path(request['output']) / 'inputs/artifacts')
    config['resources']['max_reserved_gib'] = 42
    return config


def run_recovery(request):
    import random
    import numpy as np
    import torch
    from agentic_image.learning.optim import make_optimizer, make_rl_scheduler
    from agentic_image.learning.transaction import TrainingComponent, JointTransaction, restore_rng, rng_state, atomic_json
    from agentic_image.runtime.components import Components
    from agentic_image.runtime.preflight import run_preflight

    config = restored_config(request)
    devices = validate_recovery_devices(config, torch.cuda.device_count())
    components = Components(config)
    checkpoint = Path(request['checkpoint'])
    committed = json.loads((checkpoint / 'COMMITTED.json').read_text())
    expected_state = torch.load(checkpoint / 'training_state.pt', map_location='cpu', weights_only=True)
    if len(expected_state['rng']['cuda']) != 2:
        raise ValueError('The archived CUDA RNG must describe the original two GPUs')
    expected, training = {}, []
    for name in ('actor', 'renderer'):
        expected[name] = torch.load(checkpoint / (name + '.pt'), map_location='cpu', weights_only=True)
        model = components.load_actor() if name == 'actor' else components.load_renderer()
        parameters = model.policy_parameters()
        optimizer = make_optimizer(parameters, config['rl'][name + '_learning_rate'], epsilon=config['rl']['epsilon'])
        scheduler = make_rl_scheduler(optimizer, config['rl']['warmup_rounds'])
        capture, restore = ((model.adapter_state, model.restore_adapter_state) if name == 'actor'
            else (model.policy_state, model.load_policy_state))
        training.append(TrainingComponent(name, parameters, optimizer, capture, restore, scheduler))
    transaction = JointTransaction(checkpoint.parent, training)
    state = transaction.load_committed(checkpoint.name)
    equal_state(state, expected_state, 'training_state')
    if state['behavior_version'] != committed['new_version'] or state['data_cursor'] != committed['next_data_cursor']:
        raise ValueError('Recovered training cursor differs from COMMITTED')
    for entry in training:
        equal_state(entry.capture(), expected[entry.name], entry.name)
    equal_state(rng_state(), expected_state['rng'], 'rng')
    def draw():
        return {'python': random.random(), 'numpy': float(np.random.random()), 'torch': torch.rand(4),
            'cuda': [torch.rand(4, device='cuda:' + str(index)).cpu() for index in range(2)]}
    first = draw()
    restore_rng(state['rng'])
    equal_state(draw(), first, 'next_random_draw')
    restore_rng(state['rng'])
    output = Path(config['run_dir'])
    report = {'status': 'checkpoint_restored', 'checkpoint': str(checkpoint),
        'behavior_version': state['behavior_version'], 'data_cursor': state['data_cursor'],
        'restore_interface': 'JointTransaction.load_committed', 'devices': devices,
        'model_optimizer_scheduler_active_equal': True, 'rng_equal': True, 'next_random_draw_equal': True,
        'cuda_rng_device_count': 2, 'device_remapping_used': False,
        'optimizer_state_entries': {name: len(value['optimizer']['state']) for name, value in expected.items()},
        'preflight': {}}
    atomic_json(output / 'checkpoint_restore.json', report)
    for name in ('actor', 'renderer'):
        report['preflight'][name] = run_preflight(components, request['tasks'][name], name, cycles=1)
        atomic_json(output / 'checkpoint_restore.json', report)
    restore_rng(state['rng'])
    for entry in training:
        equal_state(entry.capture(), expected[entry.name], entry.name + '_after_preflight')
    equal_state(rng_state(), state['rng'], 'rng_after_preflight')
    report.update(status='verified', restored_checkpoint_unchanged_after_preflight=True)
    atomic_json(output / 'checkpoint_restore.json', report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', type=Path, required=True)
    args = parser.parse_args()
    run_recovery(json.loads(args.request.read_text()))


if __name__ == '__main__':
    main()
