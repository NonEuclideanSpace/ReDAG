import argparse
import json
import os
from pathlib import Path
import subprocess
import time


parser = argparse.ArgumentParser()
parser.add_argument('--only', choices=('all', 'results', 'environments'), default='all')
args = parser.parse_args()
local = Path(__file__).resolve().parents[1]
root = Path('/Volumes/TOSHIBA EXT/dag-pilot-rental-backup-20261027T130933Z-9769b4')
volume = root.parent
if not os.path.ismount(volume):
    raise RuntimeError('The external volume is not mounted')
device = volume.stat().st_dev
remote = 'root@117.50.181.105:'
project = '/workspace/agentic-image-pilot-small/'
transport = 'ssh -S /tmp/dag-pilot-rental-backup-ssh -o ProxyCommand=false -o BatchMode=yes -p 29410'
state_path = root / 'manifests/delivery-sync.json'
(root / 'manifests').mkdir(parents=True, exist_ok=True)
(root / 'logs').mkdir(parents=True, exist_ok=True)
state = {'scope': args.only, 'started_at': time.time(), 'status': 'running', 'completed': [], 'active': None}


def save():
    state['updated_at'] = time.time()
    temporary = state_path.with_suffix('.tmp')
    temporary.write_text(json.dumps(state, indent=2))
    temporary.replace(state_path)
    (local / 'deployment/external-backup-status.json').write_text(json.dumps(state, indent=2))


def transfer(name, source, destination, options=()):
    if not os.path.ismount(volume) or volume.stat().st_dev != device:
        raise RuntimeError('The external volume identity changed')
    destination.mkdir(parents=True, exist_ok=True)
    command = ['nice', '-n', '15', 'rsync', '-rtpl', '--partial', '--partial-dir=.rsync-partial', '--stats', '--progress', '--bwlimit=8192', '--timeout=90', '--exclude=*.tmp', '--exclude=*.part', '--exclude=*.interrupted-*', '--exclude=._*', '--exclude=__pycache__', '--exclude=.pytest_cache', *options]
    if source.startswith(remote):
        command += ['--rsync-path=nice -n 15 ionice -c 3 rsync', '-e', transport]
    command += [source, str(destination) + '/']
    log_path = root / 'logs' / (name + '.log')
    state['active'] = {'name': name, 'source': source, 'destination': str(destination), 'log': str(log_path), 'started_at': time.time()}
    save()
    for attempt in range(1, 7):
        with log_path.open('ab', buffering=0) as stream:
            child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, env={**os.environ, 'COPYFILE_DISABLE': '1'})
            state['active'].update(pid=child.pid, attempt=attempt)
            save()
            code = child.wait()
        if code in (0, 24):
            state['completed'].append({**state['active'], 'finished_at': time.time(), 'returncode': code, 'live_source_files_vanished': code == 24})
            state['active'] = None
            save()
            return
        state['active'].update(returncode=code, last_failure_at=time.time())
        save()
        time.sleep(30)
    raise RuntimeError(f'Transfer failed after bounded retries: {name}')


def transfer_environment_archives():
    destination = root / 'remote/environments/linux-environments'
    transfer('environment-archives', remote + project + 'delivery/linux-environments/', destination, ('--exclude=.pack.lock',))
    manifest_path = destination / 'manifest.json'
    if not manifest_path.exists():
        state['environment_delivery'] = {'status': 'pending_remote_pack_completion', 'destination': str(destination)}
        save()
        return
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('status') != 'complete':
        raise ValueError('Remote environment archive manifest is incomplete')
    verifier = local / 'scripts/restore_rental_environments.py'
    subprocess.run(['python3', str(verifier), str(destination)], check=True)
    state['environment_delivery'] = {'status': 'archives_transferred_and_hash_verified', 'destination': str(destination), 'runtime_restoration_verified': False, 'unresolved_non_system_symlinks': manifest.get('unresolved_non_system_symlinks', [])}
    save()


try:
    if args.only != 'environments':
        for run, stage in [('public-initialization', 'sft_protocol'), ('public-stage-transitions', 'sft_protocol'), ('protocol-warmup', 'sft_protocol'), ('plan-global-ai', 'sft_protocol')]:
            folder = f'runs/{run}/{stage}/policy/'
            transfer('policy-' + run, remote + project + folder, root / 'remote/agentic-image-pilot-small' / folder)
        for run in ('renderer-expanded-200', 'paired-global-ai'):
            folder = f'runs/{run}/sft_renderer/adapters/'
            transfer('policy-' + run, remote + project + folder, root / 'remote/agentic-image-pilot-small' / folder)
        exclusions = ('--exclude=.venv*', '--exclude=.runtime', '--exclude=delivery', '--exclude=models', '--exclude=runs', '--exclude=data', '--exclude=.git')
        transfer('local-code', str(local) + '/', root / 'local/pilot-small', exclusions)
        transfer('remote-code-and-results', remote + project, root / 'remote/agentic-image-pilot-small', ('--exclude=models', '--exclude=.venv-tools', '--exclude=.runtime', '--exclude=delivery', '--exclude=data/artifacts', '--exclude=runs/*/sft_*/checkpoints'))
        transfer('unique-model-weights', remote + project + 'models/', root / 'remote/agentic-image-pilot-small/models', ('--copy-links', '--exclude=/actor/', '--exclude=/renderer/', '--exclude=/judge/'))
        transfer('artifacts', remote + project + 'data/artifacts/', root / 'remote/agentic-image-pilot-small/data/artifacts')
        transfer('final-project-delta', remote + project, root / 'remote/agentic-image-pilot-small', ('--exclude=models', '--exclude=.venv-tools', '--exclude=.runtime', '--exclude=delivery'))
    if args.only in ('all', 'environments'):
        transfer_environment_archives()
    state.update(status='requested_pass_completed', finished_at=time.time(), active=None)
    save()
except BaseException as error:
    state.update(status='failed', error=f'{type(error).__name__}: {error}', failed_at=time.time())
    save()
    raise
