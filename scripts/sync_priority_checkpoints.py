import json
import os
from pathlib import Path
import subprocess
import time


ROOT = Path('/Volumes/TOSHIBA EXT/dag-pilot-rental-backup-20261027T130933Z-9769b4')
assert os.path.ismount(ROOT.parent)
DEVICE = ROOT.parent.stat().st_dev
STATUS = Path('pilot-small/deployment/priority-checkpoint-backup-status.json')
REMOTE = '/workspace/agentic-image-pilot-small'
SSH = ['ssh', '-S', '/tmp/dag-pilot-rental-control-ssh', '-o', 'ProxyCommand=false', '-o', 'BatchMode=yes', '-p', '29410', 'root@117.50.181.105']
state = {'started_at': time.time(), 'status': 'running', 'completed': [], 'active': None}


def save():
    state['updated_at'] = time.time()
    tmp = STATUS.with_suffix('.tmp')
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(STATUS)


try:
    for name, gate, folder in [
        ('initialization80', 'deployment/public-initialization-sft80-resume4t.exit.json', 'runs/public-initialization/sft_protocol/policy'),
        ('renderer128families200', 'deployment/renderer-expanded-200-resume4t.exit.json', 'runs/renderer-expanded-200/sft_renderer/adapters'),
    ]:
        state['active'] = {'name': name, 'phase': 'waiting_for_completed_training'}
        save()
        deadline = time.monotonic() + 4 * 3600
        while time.monotonic() < deadline:
            try:
                probe = subprocess.run(SSH + [f'cat {REMOTE}/{gate}'], capture_output=True, text=True, timeout=20)
            except subprocess.TimeoutExpired:
                state['active']['last_probe_timeout_at'] = time.time()
                save()
                time.sleep(30)
                continue
            if probe.returncode == 0:
                result = json.loads(probe.stdout)
                if result['returncode'] != 0:
                    raise RuntimeError(f'{name} training did not complete successfully: {result}')
                break
            time.sleep(30)
        else:
            raise TimeoutError(f'{name} checkpoint was not ready')
        if not os.path.ismount(ROOT.parent) or ROOT.parent.stat().st_dev != DEVICE:
            raise RuntimeError('The requested external disk identity changed')
        destination = ROOT / 'remote/agentic-image-pilot-small' / folder
        destination.mkdir(parents=True, exist_ok=True)
        command = ['nice', '-n', '15', 'rsync', '-rtL', '--partial', '--partial-dir=.rsync-partial', '--stats', '--progress', '--timeout=90', '--bwlimit=4096', '--rsync-path=nice -n 15 ionice -c 3 rsync', '-e', 'ssh -S /tmp/dag-pilot-rental-backup-ssh -o ProxyCommand=false -o BatchMode=yes -p 29410', f'root@117.50.181.105:{REMOTE}/{folder}/', str(destination) + '/']
        log = ROOT / 'logs' / f'priority-{name}.log'
        for attempt in range(1, 7):
            with log.open('ab', buffering=0) as output:
                process = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT)
                state['active'].update(phase='copying', pid=process.pid, attempt=attempt, destination=str(destination), log=str(log))
                save()
                code = process.wait()
            if code == 0:
                state['completed'].append({**state['active'], 'finished_at': time.time(), 'returncode': code})
                save()
                break
            time.sleep(30)
        else:
            raise RuntimeError(f'{name} checkpoint copy failed after six attempts')
    state.update(status='completed', active=None, finished_at=time.time())
    save()
except BaseException as error:
    state.update(status='failed', error=f'{type(error).__name__}: {error}')
    save()
    raise
