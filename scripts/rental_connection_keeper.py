import getpass
import fcntl
import argparse
import os
from pathlib import Path
import pty
import select
import signal
import shlex
import subprocess
import sys
import termios
import time


parser = argparse.ArgumentParser()
parser.add_argument('--control', default='/tmp/dag-pilot-rental-control-ssh')
parser.add_argument('--hours', type=float, default=6)
parser.add_argument('--interface')
args = parser.parse_args()
control = args.control
password = getpass.getpass('Rental SSH password (memory only): ')
deadline = time.monotonic() + args.hours * 3600
attempt = 0
child = None


def stop(signum, frame):
    if child and child.poll() is None:
        child.terminate()
    raise SystemExit(0)


signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGINT, stop)


def attach_terminal():
    os.setsid()
    fcntl.ioctl(0, termios.TIOCSCTTY, 0)


while time.monotonic() < deadline:
    if Path(control).exists():
        check = subprocess.run(['ssh', '-S', control, '-O', 'check', '-p', '29410', 'root@117.50.181.105'], capture_output=True)
        if check.returncode == 0:
            print('Existing control connection is active; not creating another.', flush=True)
            break
    attempt += 1
    master, slave = pty.openpty()
    command = ['ssh', '-v', '-M', '-S', control, '-o', 'ControlPersist=no', '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=3', '-o', 'ConnectTimeout=12', '-o', 'PreferredAuthentications=password', '-o', 'PubkeyAuthentication=no', '-o', 'NumberOfPasswordPrompts=1', '-o', 'StrictHostKeyChecking=yes', '-p', '29410', 'root@117.50.181.105', '-N']
    if args.interface:
        transport = shlex.join([sys.executable, str(Path(__file__).with_name('rental_bound_transport.py').resolve()), '--interface', args.interface])
        command[1:1] = ['-o', 'ProxyCommand=' + transport]
    child = subprocess.Popen(command, stdin=slave, stdout=slave, stderr=slave, preexec_fn=attach_terminal)
    os.close(slave)
    buffer = b''
    sent = False
    authenticated = False
    started = time.monotonic()
    print(f'Attempt {attempt}: connecting through one SSH process.', flush=True)
    while child.poll() is None and time.monotonic() < deadline:
        readable, _, _ = select.select([master], [], [], 1)
        if readable:
            try:
                chunk = os.read(master, 8192)
            except OSError:
                break
            if not chunk:
                break
            buffer = (buffer + chunk)[-16384:]
            if b'password:' in buffer and not sent:
                os.write(master, password.encode() + b'\n')
                sent = True
                print(f'Attempt {attempt}: responding immediately to the verified-host password prompt.', flush=True)
            if b'Authenticated to ' in buffer and not authenticated:
                authenticated = True
                print(f'Attempt {attempt}: AUTHENTICATED; control connection active.', flush=True)
        if not authenticated and time.monotonic() - started > 90:
            child.terminate()
            break
    os.close(master)
    if child.poll() is None:
        child.terminate()
    child.wait(timeout=10)
    text = buffer.decode(errors='replace')
    errors = [line.strip() for line in text.splitlines() if any(term in line.lower() for term in ['exceeded maxstartups', 'connection closed', 'permission denied', 'timed out', 'no route', 'connection reset'])]
    print(f'Attempt {attempt}: exit {child.returncode}; ' + ' | '.join(errors[-3:]), flush=True)
    if not authenticated and attempt >= 12:
        print('Bounded recovery attempts exhausted; credential forgotten on exit.', flush=True)
        break
    time.sleep(30)
password = ''
