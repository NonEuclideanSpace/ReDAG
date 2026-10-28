import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import signal
import shutil
import stat
import subprocess
import time

HELPER_SPEC = importlib.util.spec_from_file_location('isolated_restore_helper', Path(__file__).with_name('check_isolated_rental_restore.py'))
HELPER = importlib.util.module_from_spec(HELPER_SPEC)
HELPER_SPEC.loader.exec_module(HELPER)
check_file, system_bindings, write = HELPER.check_file, HELPER.system_bindings, HELPER.write
RECOVERY_SPEC = importlib.util.spec_from_file_location('checkpoint_recovery_helper', Path(__file__).with_name('restored_checkpoint_recovery.py'))
RECOVERY = importlib.util.module_from_spec(RECOVERY_SPEC)
RECOVERY_SPEC.loader.exec_module(RECOVERY)


ACTOR_REVISION = '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'
RENDERER_REVISION = 'a3b4f4849157f664bdbc776fd7453c2783562f4d'
PYTHON = '/usr/local/miniconda3/envs/py312/bin/python'
SAMPLES = {
    'actor': ('viewer_actor_task.jsonl', 'viewer-probe:magicbrush:140513:1'),
    'renderer': ('viewer_renderer_task.jsonl', 'deployment-probe:zebra-source-reference'),
}
IMAGE_IDS = {
    'artifact:b0ec9f0c44b45540ebfb0091d9e3afdd431a9503a7127f9261750afd5a12c756',
    'artifact:0eb6f3cec35c58cbac46523656e52d85415a44cdadf57fd6d1541ce42277c9c2',
}


def gpu_device_nodes(indices):
    listing = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid',
        '--format=csv,noheader'], text=True, timeout=10)
    visible = {}
    for line in listing.splitlines():
        index, uuid = (part.strip() for part in line.split(','))
        if int(index) in visible:
            raise ValueError('Duplicate visible GPU index')
        visible[int(index)] = uuid
    devices = {}
    for path in Path('/proc/driver/nvidia/gpus').glob('*/information'):
        fields = dict((key.strip(), value.strip()) for key, value in
            (line.split(':', 1) for line in path.read_text().splitlines() if ':' in line))
        uuid = fields.get('GPU UUID')
        if uuid in devices:
            raise ValueError('Duplicate driver GPU identity')
        devices[uuid] = int(fields['Device Minor'])
    result = []
    for index in indices:
        uuid = visible[index]
        minor = devices[uuid]
        node = Path('/dev') / ('nvidia' + str(minor))
        info = node.stat()
        if minor < 0 or not stat.S_ISCHR(info.st_mode) or os.minor(info.st_rdev) != minor:
            raise ValueError('GPU UUID does not resolve to its actual character device')
        result.append({'nvidia_index': index, 'uuid': uuid, 'device_minor': minor,
            'device_node': str(node)})
    return result


def parser():
    result = argparse.ArgumentParser()
    result.add_argument('--project', type=Path, default=Path.cwd())
    result.add_argument('--root', type=Path, default=Path('/workspace/env-restore-check/root'))
    result.add_argument('--proot', type=Path, default=Path('/workspace/env-restore-check/proot-runtime/usr/bin/proot'))
    result.add_argument('--execution', choices=('native', 'proot'))
    result.add_argument('--gpu', type=int, choices=(0, 1), default=1)
    result.add_argument('--components', nargs='+', choices=('actor', 'renderer'), default=['actor', 'renderer'])
    result.add_argument('--output', type=Path, default=Path('runs/env-restored-gpu-check'))
    result.add_argument('--report', type=Path, default=Path('deployment/env-restored-gpu-check.json'))
    result.add_argument('--timeout-seconds', type=int, default=1800)
    result.add_argument('--recovery-archive', type=Path)
    result.add_argument('--recovery-manifest', type=Path)
    result.add_argument('--execute', action='store_true')
    return result


def validate_output(project, output):
    project, output = project.resolve(), output.absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError('Verification output must be new: ' + str(output))
    if output.parent.resolve() != project / 'runs' or not output.name.startswith('env-restored-gpu-check'):
        raise ValueError('Verification output must be a dedicated runs/env-restored-gpu-check directory')
    return output


def model_inputs(project):
    actor = project / 'models/actor-4b-nf4'
    renderer = project / 'models/renderer' / RENDERER_REVISION
    manifest = json.loads((actor / 'base_manifest.json').read_text())
    if manifest.get('model_id') != 'Qwen/Qwen3.5-4B' or manifest.get('revision') != ACTOR_REVISION:
        raise ValueError('Existing actor NF4 identity differs from the pinned configuration')
    for directory in (actor, renderer, *(renderer / name for name in ('tokenizer', 'text_encoder', 'vae', 'transformer'))):
        if not directory.is_dir() or directory.is_symlink():
            raise ValueError('A real existing model directory is required: ' + str(directory))
    for directory in (actor, renderer):
        for path in directory.rglob('*'):
            if path.is_symlink() and not path.resolve().is_relative_to(directory):
                raise ValueError('Model symlink escapes its declared bind: ' + str(path))
    return [actor, renderer]


def source_identity(project):
    files = []
    digest = hashlib.sha256()
    for path in sorted((project / 'src').rglob('*.py')):
        if path.is_symlink():
            raise ValueError('Current source binding must not contain symlinks')
        payload = path.read_bytes()
        name = str(path.relative_to(project))
        value = hashlib.sha256(payload).hexdigest()
        digest.update((name + '\0' + value + '\n').encode())
        files.append({'path': name, 'bytes': len(payload), 'sha256': value})
    if not files:
        raise ValueError('Current explicitly versioned project source is absent')
    return {'sha256': digest.hexdigest(), 'files': files}


def sample_inputs(project, components):
    result = [(project / 'configs/pilot-small.yaml', Path('config.yaml'))]
    ids = set()
    for component in components:
        filename, task_id = SAMPLES[component]
        path = project / 'data/preflight' / filename
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if len(rows) != 1 or rows[0]['task_id'] != task_id:
            raise ValueError('Preflight must use the registered real public sample')
        request = rows[0]['user_request']
        ids.add(request['source_image_id'])
        ids.update(request.get('appearance_image_ids', []))
        result.append((path, Path(filename)))
    if not ids or not ids <= IMAGE_IDS:
        raise ValueError('Unexpected preflight artifact identity')
    for identity in sorted(ids):
        relative = Path('refs') / (identity.split(':', 1)[1] + '.json')
        ref = project / 'data/artifacts' / relative
        record = json.loads(ref.read_text())
        payload = Path(record['relative_path'])
        if record['artifact_id'] != identity or record['kind'] != 'rgb' or payload.is_absolute() or '..' in payload.parts or payload.parts[0] != 'values':
            raise ValueError('Preflight artifact reference is not a registered RGB value')
        result.extend([(ref, Path('artifacts') / relative), (project / 'data/artifacts' / payload, Path('artifacts') / payload)])
    for source, _ in result:
        if not source.is_file() or source.is_symlink():
            raise ValueError('A regular existing preflight input is required: ' + str(source))
    return result


def cli_arguments(component, virtual_output):
    result = ['-m', 'agentic_image.cli', 'preflight', '--config', str(virtual_output / 'inputs/config.yaml'),
        '--tasks', str(virtual_output / 'inputs' / SAMPLES[component][0]), '--component', component, '--cycles', '1']
    overrides = {'run_dir': str(virtual_output / component), 'data.artifacts': str(virtual_output / 'inputs/artifacts'),
        'devices.actor': 'cuda:0', 'devices.renderer': 'cuda:0', 'devices.encoders': 'cuda:0', 'devices.judge': 'cuda:0',
        'models.actor_policy': 'null', 'models.actor_reference': 'null',
        'models.renderer_policy': 'null', 'models.renderer_reference': 'null', 'resources.max_reserved_gib': '42'}
    for key, value in overrides.items():
        result.extend(['--set', key + '=' + value])
    return result


PROBE = '''import importlib,json,os,runpy,sys
from pathlib import Path
def record(path):
    path=Path(path).resolve(); value=path.stat()
    return {'path':str(path),'device':value.st_dev,'inode':value.st_ino,'bytes':value.st_size}
modules={name:record(importlib.import_module(name).__file__) for name in ['torch','transformers','diffusers','peft','bitsandbytes','causal_conv1d','fla','triton','agentic_image']}
import torch
evidence={'executable':record(sys.executable),'prefix':sys.prefix,'modules':modules,'cuda_visible_devices':os.environ['CUDA_VISIBLE_DEVICES'],'cuda_device_count':torch.cuda.device_count(),'cuda_available':torch.cuda.is_available()}
origins=Path(sys.argv[1])
def native_inventory():
    if 'RESTORED_NATIVE_PREFIX' not in os.environ:
        return
    prefix=Path(os.environ['RESTORED_NATIVE_PREFIX']).resolve()
    source=Path(os.environ['PYTHONPATH']).resolve()
    output=origins.parent.resolve()
    restored,generated={},{}
    for name,module in list(sys.modules.items()):
        filename=getattr(module,'__file__',None)
        if not isinstance(filename,str) or filename.startswith('<'):
            continue
        path=Path(filename).resolve()
        if not path.is_file():
            continue
        if path.is_relative_to(prefix):
            restored[str(path)]=record(path)
        elif path.is_relative_to(source):
            continue
        elif path in (output/'probe.py',output/'recovery-probe.py'):
            continue
        elif path.is_relative_to(output/'cache') and path.suffix=='.so':
            generated[str(path)]=record(path)
        else:
            raise RuntimeError('Python module escaped the independent restored environment: '+name+' '+str(path))
    evidence['native_loaded_files']=list(restored.values())
    evidence['generated_runtime_extensions']=list(generated.values())
native_inventory()
origins.write_text(json.dumps(evidence,indent=2)+'\\n')
expected=2 if sys.argv[2]=='--restore-checkpoint' else 1
if not evidence['cuda_available'] or evidence['cuda_device_count']!=expected:
    raise RuntimeError('The required selected CUDA device count is unavailable')
if expected==2:
    sys.argv=sys.argv[3:]
    runpy.run_path(sys.argv[0],run_name='__main__')
else:
    sys.argv=['agentic_image.cli',*sys.argv[4:]]
    runpy.run_module('agentic_image.cli',run_name='__main__')
native_inventory()
origins.write_text(json.dumps(evidence,indent=2)+'\\n')
'''


def check_runtime_file(record, root, roots, execution):
    root = Path(root).resolve()
    if execution == 'native':
        physical = Path(record['path']).resolve()
        if not physical.is_relative_to(root):
            raise ValueError('Native Python dependency escaped the restored root')
        record = {**record, 'path': '/' + str(physical.relative_to(root))}
    return check_file(record, root, roots)


def native_environment(root, environment):
    prefix = root / Path(PYTHON).parent.parent.relative_to('/')
    package_libraries = sorted(prefix.glob('lib/python*/site-packages/nvidia/*/lib'))
    package_libraries += sorted(prefix.glob('lib/python*/site-packages/torch/lib'))
    if any(not path.resolve().is_relative_to(prefix.resolve()) for path in package_libraries):
        raise ValueError('Restored GPU library directory escaped its Python prefix')
    return {**environment, 'PYTHONHOME': str(prefix), 'RESTORED_NATIVE_PREFIX': str(prefix),
        'PATH': str(prefix / 'bin') + ':/usr/bin:/bin',
        'LD_LIBRARY_PATH': ':'.join(map(str, [prefix / 'lib', *package_libraries])) + ':' + environment['LD_LIBRARY_PATH']}


def metadata(paths):
    return {str(path): (path.stat().st_dev, path.stat().st_ino, path.stat().st_size, path.stat().st_mtime_ns) for path in paths}


def run_bounded(command, environment, log, timeout):
    with log.open('x') as stream:
        process = subprocess.Popen(command, env=environment, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait(timeout=timeout)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
        if code:
            raise RuntimeError('Restored GPU preflight failed; see ' + str(log))


def interrupted(signum, frame):
    raise InterruptedError('Verification interrupted by signal ' + str(signum))


def main():
    args = parser().parse_args()
    project, root = args.project.resolve(), args.root.resolve()
    if not 1 <= args.timeout_seconds <= 3600 or len(set(args.components)) != len(args.components):
        raise ValueError('Use unique sequential components and a timeout of 1..3600 seconds')
    if bool(args.recovery_archive) != bool(args.recovery_manifest):
        raise ValueError('Checkpoint recovery requires both its archive and export manifest')
    recovery = None
    if args.recovery_archive:
        if set(args.components) != {'actor', 'renderer'}:
            raise ValueError('Checkpoint recovery verifies both training components together')
        archive = args.recovery_archive if args.recovery_archive.is_absolute() else project / args.recovery_archive
        manifest_path = args.recovery_manifest if args.recovery_manifest.is_absolute() else project / args.recovery_manifest
        recovery = RECOVERY.recovery_plan(project, archive, manifest_path)
    execution = args.execution or ('native' if recovery else 'proot')
    output = validate_output(project, args.output if args.output.is_absolute() else project / args.output)
    report = args.report if args.report.is_absolute() else project / args.report
    manifest = json.loads((project / 'delivery/linux-environments/manifest.json').read_text())
    extraction = json.loads((root.parent / 'extraction_completed.json').read_text())
    if manifest['status'] != 'complete' or extraction['status'] != 'extracted' or extraction['root'] != str(root):
        raise ValueError('Existing independent environment extraction is required')
    if (root / '.restore-check-marker').read_text().strip() != extraction['marker']:
        raise ValueError('Restored root marker differs from extraction evidence')
    if {x['archive']: x['sha256'] for x in extraction['archives']} != {k: v['sha256'] for k, v in manifest['archives'].items()}:
        raise ValueError('Restored environment identity differs from its archive manifest')
    if manifest['project'] != str(project) or (execution == 'proot' and not args.proot.is_file()) or not (root / PYTHON.lstrip('/')).is_file():
        raise ValueError('Original project prefix, restored Python and existing proot must be available')
    models, samples, identity = model_inputs(project), sample_inputs(project, args.components), source_identity(project)
    observed = [source for source, _ in samples] + [models[0] / 'base_manifest.json']
    if recovery:
        observed.extend([archive, manifest_path])
    before = metadata(observed)
    result = {'status': 'prepared', 'gpu_verified': False, 'scope': 'Restored environment GPU execution with explicitly bound current code and existing models; not complete offline model restoration',
        'root': str(root), 'physical_gpu': args.gpu, 'logical_gpu': 0, 'components': args.components, 'source_identity': identity,
        'models': [str(p) for p in models], 'inputs': [str(p) for p, _ in samples], 'output': str(output),
        'source_bind_access': 'Ordinary proot binds; audited call paths only read source inputs. Not a read-only filesystem sandbox.',
        'cycles_per_component': 1, 'maximum_reserved_gib': 42, 'checkpoint_or_training_mutations': False, 'results': {}, 'execution': execution}
    if recovery:
        result.update(scope='Restored archived environment and independently extracted trained checkpoint with current code and existing base-model bindings; host kernel, GPU driver and system ELF libraries remain dependencies',
            physical_gpu=None, logical_gpu=None, physical_gpus=[0, 1], checkpoint_recovery_verified=False, recovery_archive=recovery['archive'],
            recovered_training_sources_mutated=False, exact_original_cuda_device_mapping=True,
            system_dependencies=['Linux x86_64 kernel', 'host NVIDIA driver and CUDA device nodes', 'proot and libtalloc', 'host glibc and system ELF libraries', 'host C compiler, binutils, headers and ldconfig for GPU JIT compilation'],
            base_models_independently_restored=False, project_code_independently_restored=False)
    if execution == 'native':
        result['source_bind_access'] = 'Native independent extracted Python; copied small inputs, current source and existing base models remain explicit host dependencies.'
        result['scope'] = 'Native independently restored Python environment and extracted checkpoint; host source, base models, GPU driver and compilation toolchain remain explicit dependencies.'
        result['system_dependencies'] = ['Linux x86_64 kernel', 'host NVIDIA driver and CUDA device nodes',
            'host glibc and system ELF libraries', 'host C compiler, binutils, headers and ldconfig for GPU JIT compilation']
    write(report, result)
    if not args.execute:
        print(json.dumps({'status': 'prepared', 'report': str(report), 'gpu_started': False}))
        return
    signal.signal(signal.SIGTERM, interrupted)
    output.mkdir(parents=True)
    (output / 'inputs/artifacts/refs').mkdir(parents=True)
    (output / 'inputs/artifacts/values').mkdir(parents=True)
    probe = output / 'probe.py'
    probe.write_text(PROBE)
    if recovery:
        extracted = RECOVERY.extract_recovery(recovery, output / 'recovered-checkpoint')
        request = {**extracted, 'recovered_root': extracted['destination'], 'original_project': str(project),
            'output': str(output), 'actor_nf4': str(models[0]), 'renderer_base': str(models[1]),
            'tasks': {name: str(output / 'inputs' / SAMPLES[name][0]) for name in ('actor', 'renderer')}}
        write(output / 'recovery-request.json', request)
        (output / 'recovery-probe.py').write_bytes(Path(__file__).with_name('restored_checkpoint_recovery.py').read_bytes())
        result['recovery_extraction'] = {key: value for key, value in extracted.items() if key != 'config'}
        write(report, result)
    libraries = args.proot.parent.parent / 'lib/x86_64-linux-gnu'
    launcher = (['/lib64/ld-linux-x86-64.so.2', '--library-path', str(libraries), str(args.proot)]
        if list(libraries.glob('libtalloc.so*')) else [str(args.proot)])
    bindings = [(Path(p), Path(p)) for p in system_bindings()]
    for path in [Path('/sys'), *Path('/dev').glob('nvidia*'), Path('/usr/local/nvidia/lib'), Path('/usr/local/nvidia/lib64')]:
        if path.exists():
            bindings.append((path, path))
    for path in map(Path, ['/usr/bin', '/bin', '/usr/sbin', '/sbin', '/usr/include',
            '/usr/lib/gcc', '/etc/ld.so.cache']):
        if path.exists():
            bindings.append((path, path))
    result['gpu_device_nodes'] = gpu_device_nodes([0, 1] if recovery else [args.gpu])
    bindings += [(project / 'src', project / 'src'), (output, output), *((p, p) for p in models)]
    for source, relative in samples:
        target = output / 'inputs' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if execution == 'native':
            shutil.copyfile(source, target)
        else:
            target.touch(exist_ok=False)
            bindings.append((source, target))
    base = launcher + ['-r', str(root), '-w', str(project)]
    for source, target in bindings:
        base.extend(['-b', str(source) + ':' + str(target)])
    environment = {'PATH': str(Path(PYTHON).parent) + ':/usr/bin:/bin', 'PYTHONPATH': str(project / 'src'),
        'CUDA_VISIBLE_DEVICES': '0,1' if recovery else str(args.gpu), 'CUDA_DEVICE_ORDER': 'PCI_BUS_ID', 'NVIDIA_VISIBLE_DEVICES': '0,1' if recovery else str(args.gpu),
        'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'PYTHONNOUSERSITE': '1', 'PYTHONDONTWRITEBYTECODE': '1',
        'PYTHONFAULTHANDLER': '1', 'PYTHONUNBUFFERED': '1',
        'TOKENIZERS_PARALLELISM': 'false', 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
        'LANG': 'C.UTF-8', 'LC_ALL': 'C.UTF-8', 'LD_LIBRARY_PATH': '/usr/local/nvidia/lib64:/usr/local/nvidia/lib:/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu'}
    for key in ('HOME', 'TMPDIR', 'HF_HOME', 'XDG_CACHE_HOME', 'TRITON_CACHE_DIR', 'TORCH_EXTENSIONS_DIR', 'TORCHINDUCTOR_CACHE_DIR', 'CUDA_CACHE_PATH'):
        path = output / 'cache' / key.lower()
        path.mkdir(parents=True)
        environment[key] = str(path)
    executable = PYTHON
    if execution == 'native':
        environment = native_environment(root, environment)
        base, bindings = [], []
        executable = str(root / PYTHON.lstrip('/'))
    result.update(status='running', started_at=time.time(), bindings=[{'source': str(a), 'target': str(b)} for a, b in bindings])
    write(report, result)
    try:
        for component in (['checkpoint_recovery'] if recovery else args.components):
            origins = output / (component + '-origins.json')
            arguments = (['--restore-checkpoint', str(output / 'recovery-probe.py'), '--request', str(output / 'recovery-request.json')]
                if recovery else cli_arguments(component, output))
            command = base + [executable, str(probe), str(origins), *arguments]
            run_bounded(command, environment, output / (component + '.log'), args.timeout_seconds)
            provenance = json.loads(origins.read_text())
            roots = [PurePosixPath(x) for row in manifest['archives'].values() for x in row['roots']]
            check_runtime_file(provenance['executable'], root, roots, execution)
            for name, record in provenance['modules'].items():
                if name == 'agentic_image':
                    path = Path(record['path'])
                    if not path.is_relative_to(project / 'src') or metadata([path])[str(path)][:3] != (record['device'], record['inode'], record['bytes']):
                        raise ValueError('Bound current project code has unexpected provenance')
                else:
                    check_runtime_file(record, root, roots, execution)
            if execution == 'native':
                if Path(provenance['prefix']).resolve() != Path(environment['PYTHONHOME']).resolve() or not provenance.get('native_loaded_files'):
                    raise ValueError('Native Python prefix or import inventory differs')
                for record in provenance['native_loaded_files']:
                    check_runtime_file(record, root, roots, execution)
            values_path = (output / 'recovered-preflight/checkpoint_restore.json' if recovery
                else output / component / ('preflight_' + component + '.json'))
            values = json.loads(values_path.read_text())
            if recovery and (values.get('status') != 'verified' or values.get('model_optimizer_scheduler_active_equal') is not True
                    or values.get('rng_equal') is not True or values.get('next_random_draw_equal') is not True
                    or values.get('device_remapping_used') is not False or values.get('restored_checkpoint_unchanged_after_preflight') is not True):
                raise ValueError('Actual trained checkpoint recovery did not pass')
            if component == 'actor' and values['checks']['actor'].get('passed') is not True:
                raise ValueError('Actual restored actor diagnose did not pass')
            if component == 'renderer' and not (output / component / 'preflight_render.png').is_file():
                raise ValueError('Actual restored renderer produced no image')
            result['results'][component] = {'provenance': provenance, 'preflight': values}
            write(report, result)
        if metadata(observed) != before or source_identity(project) != identity:
            raise RuntimeError('Bound small inputs or current source changed during verification')
        result.update(status='verified', gpu_verified=True, source_inputs_unchanged=True, finished_at=time.time())
        if recovery:
            result['checkpoint_recovery_verified'] = True
    except BaseException as error:
        result.update(status='failed', error=str(error), finished_at=time.time())
        raise
    finally:
        write(report, result)
    print(json.dumps({'status': result['status'], 'report': str(report), 'gpu_verified': result['gpu_verified']}))


if __name__ == '__main__':
    main()
