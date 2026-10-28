import argparse
import json
import os
from pathlib import Path
import subprocess
import time


parser = argparse.ArgumentParser()
parser.add_argument('--old', required=True)
parser.add_argument('--new', required=True)
parser.add_argument('--not-before', type=float, required=True)
parser.add_argument('--status', required=True)
parser.add_argument('--hours', type=float, default=1)
args = parser.parse_args()
old, new = Path(args.old), Path(args.new)
if old == new or any(p.parent != Path('/tmp') or not p.name.startswith('dag-pilot-rental-') for p in (old, new)):
    raise ValueError('Expected distinct rental control sockets under /tmp')
status = Path(args.status)
deadline = time.monotonic() + args.hours * 3600
state = {'status': 'waiting_for_old_socket_removal', 'pid': os.getpid(),
         'old': str(old), 'new': str(new), 'started_at': time.time()}
while time.monotonic() < deadline:
    if old.is_symlink() and new.is_socket() and old.resolve() == new.resolve():
        state.update(status='completed', finished_at=time.time())
        break
    if time.time() >= args.not_before and not old.exists() and not old.is_symlink() and new.is_socket():
        check = subprocess.run(['ssh', '-S', str(new), '-O', 'check', '-p', '29410',
                                'root@117.50.181.105'], capture_output=True, timeout=5)
        if check.returncode == 0:
            old.symlink_to(new)
            state.update(status='completed', finished_at=time.time(),
                         existing_authenticated_master_verified=True)
            break
    state['updated_at'] = time.time()
    status.write_text(json.dumps(state, indent=2))
    time.sleep(2)
else:
    state.update(status='bounded_wait_expired', finished_at=time.time())
status.write_text(json.dumps(state, indent=2))
print(json.dumps(state), flush=True)
