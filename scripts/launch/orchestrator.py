"""GPU-aware job scheduler (standard library only).  scripts/launch/pipeline.py defines the ReDAG jobs.

Jobs form a DAG and need 0 (CPU), 1 or more GPUs.  Whenever GPUs are free, the scheduler starts the ready job with the
longest remaining critical path, pinned to its GPUs through CUDA_VISIBLE_DEVICES, so no GPU waits for a phase
boundary.  A ready job that needs more GPUs than are free gets a reservation (EASY backfilling): smaller jobs may use
the waiting GPUs only if they are expected to finish before it can start.

All state is on disk, so the scheduler can be stopped, killed or restarted at any time:
  runs/logs/<job>.done              finished; holds the job's stamp (same marker as env.sh run_job)
  runs/logs/<job>.log               job output, appended per attempt
  runs/logs/timings.jsonl           one row per finished attempt
  runs/state/jobs/<job>.lock        flock held while the job runs; a restarted scheduler adopts running jobs
  runs/state/jobs/<job>.json        the attempt being run (command, GPUs, start time, process group)
  runs/state/jobs/<job>.exit        exit status of the last attempt
  runs/state/jobs/<job>.failed      attempts exhausted (retried automatically after a code change, or with --retry-failed)
  runs/state/jobs/<job>.progress    progress written by the job itself (redag/progress.py)
  runs/state/status.json            snapshot for `status` and external monitors
"""
import fcntl
import json
import math
import os
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

HOUR = 3600.0


@dataclass
class Job:
    name: str
    cmd: object                     # argv list, or a string run with bash -c
    gpus: int = 0
    deps: tuple = ()
    hours: float = 0.1              # estimated wall hours (A800); refined from finished siblings and progress
    group: str = ''                 # phase label for display
    family: str = ''                # shards of one job share a family: finished ones calibrate the others
    ready: object = None            # callable -> bool: extra condition, e.g. a checkpoint written by a running job
    waiting_for: str = ''           # shown while `ready` is false
    soft: tuple = ()                # ((job, fraction), ...): planning only, ready after that share of the job
    attempts: int = 2
    stamp: str = ''                 # a .done marker holding another stamp does not count
    env: dict = field(default_factory=dict)


def atomic_write(path, text):
    path = Path(path)
    tmp = path.with_name(f'{path.name}.{os.getpid()}.tmp')
    tmp.write_text(text, encoding='utf-8')
    tmp.replace(path)


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return default


def lock_held(path):
    """True if another process holds an flock on path (checked without blocking)."""
    try:
        fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    except OSError:
        return False
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return True
    finally:
        os.close(fd)
    return False


def query_gpus():
    """[{index, name, used, total}] from nvidia-smi (MiB), or [] without GPUs."""
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.used,memory.total', '--format=csv,noheader,nounits'],
                             capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return []
    if out.returncode:
        return []
    rows = []
    for line in out.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(',')]
        if len(parts) == 4 and parts[0].isdigit():
            rows.append(dict(index=int(parts[0]), name=parts[1], used=float(parts[2]), total=float(parts[3])))
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    if visible is not None and visible.strip() and all(x.strip().isdigit() for x in visible.split(',')):
        keep = {int(x) for x in visible.split(',')}
        rows = [r for r in rows if r['index'] in keep]
    return rows


def validate(jobs):
    names = {j.name for j in jobs}
    if len(names) != len(jobs):
        raise ValueError('duplicate job names')
    for job in jobs:
        missing = [d for d in job.deps if d not in names] + [s for s, _ in job.soft if s not in names]
        if missing:
            raise ValueError(f'{job.name}: unknown dependencies {missing}')
    order, state = [], {}

    def visit(name, stack):
        if state.get(name) == 2:
            return
        if state.get(name) == 1:
            raise ValueError('dependency cycle: ' + ' -> '.join(stack + [name]))
        state[name] = 1
        for dep in by_name[name].deps:
            visit(dep, stack + [name])
        state[name] = 2
        order.append(name)
    by_name = {j.name: j for j in jobs}
    for job in jobs:
        visit(job.name, [])
    return order  # dependencies first


def bottom_levels(jobs, hours):
    """Longest estimated path (hours) from the start of each job to the end of the pipeline."""
    children = {j.name: [] for j in jobs}
    for job in jobs:
        for dep in job.deps:
            children[dep].append((job.name, 1.0))
        for dep, fraction in job.soft:
            children[dep].append((job.name, fraction))
    level = {}
    for name in reversed(validate(jobs)):
        tail = max((level[c] + (fraction - 1.0) * hours[name] for c, fraction in children[name]), default=0.0)
        level[name] = hours[name] + max(tail, 0.0)
    return level


def reservation(need, free, running, start=0.0):
    """Earliest time >= start at which `need` GPUs are free, and the spare GPUs at that time.

    running: [(expected_end, gpus)] of jobs holding GPUs now; free: GPUs free now."""
    points = sorted({start, *[end for end, _ in running if end > start]})
    for t in points:
        available = free + sum(g for end, g in running if end <= t)
        if available >= need:
            return t, available - need
    return math.inf, 0


def choose(ready, free, running, now, cpu_free, upcoming=None):
    """Pick the jobs to start now.

    ready: [(job, est_seconds, priority)] sorted by priority, highest first; free: free GPU count; running: [(end, gpus)]
    of jobs holding GPUs; upcoming: optional (job, ready_at, priority) of a multi-GPU job that becomes ready soon,
    passed only when the critical path binds.  A job that cannot start gets a reservation: lower-priority jobs then
    start only if they end before it can start, or if enough GPUs remain for it."""
    starts, held, running = [], None, list(running)
    if upcoming is not None:
        job, ready_at, priority = upcoming
        t, spare = reservation(job.gpus, free, running, max(now, ready_at))
        held = {'time': t, 'spare': spare, 'priority': priority}
    for job, est, priority in ready:
        if job.gpus == 0:
            if cpu_free > 0:
                starts.append(job)
                cpu_free -= 1
            continue
        if job.gpus > free:
            if held is None or priority > held['priority']:
                t, spare = reservation(job.gpus, free, running, now)
                held = {'time': t, 'spare': spare, 'priority': priority}
            continue
        end = now + est
        if held is not None and priority <= held['priority']:
            if end <= held['time']:
                pass  # finishes before the reserved job can start
            elif held['spare'] >= job.gpus:
                held['spare'] -= job.gpus  # enough GPUs stay free for the reserved job
            else:
                continue
        starts.append(job)
        free -= job.gpus
        running.append((end, job.gpus))
    return starts


def simulate(jobs, est, gpus, done=(), running=(), now=0.0, cpu_slots=2, progress=None):
    """List-schedule the remaining jobs with `choose`; returns {job: (start, end)} in seconds from now.

    running: [(name, remaining_seconds, gpus)].  Readiness conditions are modelled by `soft` fractions."""
    by_name = {j.name: j for j in jobs}
    done = set(done)
    level = bottom_levels(jobs, {n: est[n] / HOUR for n in by_name})
    timeline, active, t = {}, {}, now
    for name, remaining, n in running:
        active[name] = (t + remaining, n)
        timeline[name] = (t, t + remaining)
    pending = [j for j in jobs if j.name not in done and j.name not in active]
    finished = set(done)
    guard = 0
    while pending or active:
        guard += 1
        if guard > 100000:
            raise RuntimeError('simulation did not converge')

        def soft_ok(job):
            for dep, fraction in job.soft:
                if dep in finished:
                    continue
                if dep not in timeline:
                    return False
                start, end = timeline[dep]
                if t < start + fraction * (end - start) - 1e-6:
                    return False
            return True
        ready = [j for j in pending if all(d in finished for d in j.deps) and soft_ok(j) and j.gpus <= gpus]
        ready.sort(key=lambda j: (-level[j.name], j.name))
        free = gpus - sum(n for _, n in active.values())
        cpu_free = cpu_slots - sum(1 for name in active if by_name[name].gpus == 0)
        starts = choose([(j, est[j.name], level[j.name]) for j in ready], free,
                        [v for k, v in active.items() if by_name[k].gpus], t, cpu_free)
        for job in starts:
            active[job.name] = (t + est[job.name], job.gpus)
            timeline[job.name] = (t, t + est[job.name])
            pending.remove(job)
        if not active:
            break  # nothing can run (needs more GPUs than available, or an impossible dependency)
        # advance to the next event: a job ends, or a soft dependency's threshold passes
        events = [end for end, _ in active.values()]
        for job in pending:
            for dep, fraction in job.soft:
                if dep in timeline and dep not in finished:
                    start, end = timeline[dep]
                    point = start + fraction * (end - start)
                    if point > t + 1e-6:
                        events.append(point)
        t = min(events)
        for name in [k for k, (end, _) in active.items() if end <= t + 1e-6]:
            del active[name]
            finished.add(name)
    return timeline


class Notifier:
    """Webhook messages (NOTIFY_URL): ServerChan, PushPlus, Feishu, DingTalk, WeCom, Bark, ntfy, or generic JSON."""

    def __init__(self, url=None, prefix='ReDAG', log=print):
        self.url, self.prefix, self.log = (url or '').strip(), prefix, log

    def payload(self, title, text):
        from urllib.parse import parse_qs, urlparse
        url, host = self.url, urlparse(self.url).netloc
        title = f'[{self.prefix}] {title}'
        if 'ftqq.com' in host:
            return url, 'form', {'title': title[:100], 'desp': text}
        if 'pushplus' in host:
            token = parse_qs(urlparse(url).query).get('token', [''])[0]
            return url.split('?')[0], 'json', {'token': token, 'title': title, 'content': text.replace('\n', '<br>')}
        if 'feishu' in host or 'larksuite' in host:
            return url, 'json', {'msg_type': 'text', 'content': {'text': f'{title}\n{text}'}}
        if 'dingtalk' in host or 'qyapi.weixin' in host:
            return url, 'json', {'msgtype': 'text', 'text': {'content': f'{title}\n{text}'}}
        if 'day.app' in host:
            return url, 'json', {'title': title, 'body': text}
        if 'ntfy' in host:
            return url, 'text', text
        return url, 'json', {'title': title, 'text': text, 'content': text}

    def send(self, title, text=''):
        if not self.url:
            return False
        import urllib.parse
        import urllib.request
        try:
            url, kind, body = self.payload(title, text)
            headers = {'User-Agent': 'redag-pipeline'}
            if kind == 'json':
                data, headers['Content-Type'] = json.dumps(body, ensure_ascii=False).encode(), 'application/json'
            elif kind == 'form':
                data, headers['Content-Type'] = urllib.parse.urlencode(body).encode(), 'application/x-www-form-urlencoded'
            else:
                data, headers['Title'] = body.encode(), urllib.parse.quote(f'[{self.prefix}] {title}')
            with urllib.request.urlopen(urllib.request.Request(url, data=data, headers=headers), timeout=15) as response:
                response.read()
            return True
        except Exception as error:  # notifications must never stop the pipeline
            self.log(f'notification failed: {error}')
            return False


def execute(spec_path):
    """Run one job attempt (called as `orchestrator.py _exec <spec>` in its own session).

    Holds the job lock for the whole attempt (the child inherits the locked descriptor, so the lock stays held even if
    this wrapper dies), appends to the job log, then records the exit status, the timing row and the .done marker."""
    spec = json.loads(Path(spec_path).read_text(encoding='utf-8'))
    lock = os.open(spec['lock'], os.O_RDWR | os.O_CREAT, 0o644)
    for _ in range(100):
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except BlockingIOError:
            time.sleep(0.1)
    else:
        return 75  # another process runs this job
    stop = []
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, lambda signum, frame: stop.append(signum))  # the child gets the same signal (process group)
    env = dict(os.environ, **spec['env'])
    cmd = spec['cmd'] if isinstance(spec['cmd'], list) else ['bash', '-c', spec['cmd']]
    started = time.time()
    with open(spec['log'], 'a', encoding='utf-8') as log:
        log.write(f"\n===== {spec['name']} attempt {spec['attempt']} on {socket.gethostname()} GPUs [{env.get('CUDA_VISIBLE_DEVICES', '')}] "
                  f"{time.strftime('%F %T')} =====\n")
        log.flush()
        try:
            child = subprocess.Popen(cmd, cwd=spec['cwd'], env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                     pass_fds=(lock,))
            status = child.wait()
        except OSError as error:
            log.write(f'cannot start: {error}\n')
            status = 127
        seconds = time.time() - started
        log.write(f"===== exit {status} after {seconds / 60:.1f} min =====\n")
    ended = time.strftime('%F %T')
    atomic_write(spec['exit'], json.dumps(dict(status=status, seconds=seconds, ended=ended, attempt=spec['attempt'],
                                               signals=stop, started=started)))
    with open(spec['timings'], 'a', encoding='utf-8') as stream:
        stream.write(json.dumps(dict(job=spec['name'], seconds=round(seconds), status=status, ended=ended, attempt=spec['attempt'],
                                     gpus=spec['gpus'])) + '\n')
    if status == 0:
        atomic_write(spec['done'], spec['stamp'] + '\n')
    return 0 if status == 0 else 1


class Scheduler:
    def __init__(self, jobs, runs, cwd, gpus=None, cpu_slots=None, poll=5.0, notifier=None, gpu_free_mib=2000,
                 code_version='', retry_failed=False, log=None, max_hours=None, tick=None, tick_every=600.0, claim=None):
        self.order = validate(jobs)
        self.jobs = {j.name: j for j in jobs}
        self.runs, self.cwd = Path(runs), str(cwd)
        self.logs, self.state = self.runs / 'logs', self.runs / 'state'
        self.jobdir = self.state / 'jobs'
        for folder in (self.logs, self.jobdir):
            folder.mkdir(parents=True, exist_ok=True)
        self.fixed_gpus = gpus  # None: detect with nvidia-smi
        self.cpu_slots, self.poll, self.gpu_free_mib = cpu_slots, poll, gpu_free_mib
        self.notifier = notifier or Notifier()
        self.code_version, self.retry_failed, self.max_hours = code_version, retry_failed, max_hours
        self.running = {}   # name -> {proc, gpus, started, attempt, adopted}
        self.attempts = {}  # name -> attempts used by this scheduler
        self.retry_at = {}  # name -> earliest time of the next attempt
        self.events = []
        self.stopping = False
        self.retry_delay = 60.0
        self.tick, self.tick_every, self.last_tick = tick, tick_every, 0.0  # e.g. the health report
        self.claim = claim  # name -> bool, asked right before a start (jobs shared with other machines; None: always run)
        self.log = log or (lambda text: print(time.strftime('[%F %T] ') + text, flush=True))
        self.gpu_info, self.foreign = [], set()

    # ---------------------------------------------------------------- paths and on-disk state
    def path(self, name, suffix):
        return self.jobdir / f'{name}.{suffix}'

    def done_marker(self, name):
        return self.logs / f'{name}.done'

    def is_done(self, name):
        marker = self.done_marker(name)
        if not marker.exists():
            return False
        stamp = self.jobs[name].stamp
        return not stamp or marker.read_text(encoding='utf-8').strip() == stamp

    def failed_info(self, name):
        return read_json(self.path(name, 'failed'))

    def is_failed(self, name):
        info = self.failed_info(name)
        if not info:
            return False
        if self.retry_failed or info.get('code') != self.code_version:
            return False  # code changed since (or explicit retry): try again
        return True

    def gpu_list(self):
        if self.fixed_gpus is not None:
            return list(self.fixed_gpus)
        return [g['index'] for g in self.gpu_info]

    def refresh_gpus(self):
        if self.fixed_gpus is None:
            self.gpu_info = query_gpus()
        held = {g for r in self.running.values() for g in r['gpus']}
        self.foreign = {g['index'] for g in self.gpu_info
                        if self.gpu_free_mib is not None and g['index'] not in held and g['used'] > self.gpu_free_mib}

    def free_gpus(self):
        held = {g for r in self.running.values() for g in r['gpus']}
        busy = {g['index'] for g in self.gpu_info if self.gpu_free_mib is not None and g['used'] > self.gpu_free_mib}
        return [g for g in self.gpu_list() if g not in held and g not in busy]

    # ---------------------------------------------------------------- estimates
    def estimates(self, now=None):
        """Seconds each job is expected to take in total (running jobs: their expected total)."""
        now = now or time.time()
        measured = {}
        for row in read_jsonl(self.logs / 'timings.jsonl'):
            if row.get('status') == 0 and row.get('job') in self.jobs:
                measured[row['job']] = row['seconds']
        family_runs = {}
        for name, seconds in measured.items():
            fam = self.jobs[name].family
            if fam:
                family_runs.setdefault(fam, []).append((seconds, self.jobs[name].hours))
        est = {}
        for name, job in self.jobs.items():
            if name in measured:
                est[name] = max(measured[name], 1.0)
            elif job.family in family_runs:  # scale by finished siblings: actual / estimated
                ratios = sorted(s / max(h * HOUR, 1.0) for s, h in family_runs[job.family])
                est[name] = job.hours * HOUR * ratios[len(ratios) // 2]
            else:
                est[name] = job.hours * HOUR
        for name, info in self.running.items():
            est[name] = (now - info['started']) + self.remaining(name, now, est[name])
        return est

    def progress(self, name):
        return read_json(self.path(name, 'progress'))

    def remaining(self, name, now, total_est):
        """Expected seconds left for a running job: from its progress rate when available, else from the estimate."""
        info = self.running[name]
        elapsed = now - info['started']
        prog = self.progress(name)
        if prog and prog.get('total') and prog.get('time', 0) >= info['started']:
            first = info.setdefault('first_progress', (prog['done'], prog['time']))
            done_now, total = prog['done'], prog['total']
            if done_now >= total:
                return 60.0
            if done_now > first[0] and prog['time'] > first[1]:
                rate = (done_now - first[0]) / (prog['time'] - first[1])
                return max((total - done_now) / rate - (now - prog['time']), 30.0)
            if done_now > 0:
                return max(elapsed * (total - done_now) / done_now, 30.0)
        return max(total_est - elapsed, 60.0)

    # ---------------------------------------------------------------- job lifecycle
    def adopt(self):
        """Jobs still running from an earlier scheduler (their lock is held) are tracked, not restarted."""
        for name in self.order:
            if name in self.running or self.is_done(name):
                continue
            if not lock_held(self.path(name, 'lock')):
                self.path(name, 'restart').unlink(missing_ok=True)  # stale request for a job that is not running
                continue
            spec = read_json(self.path(name, 'json'), {})
            self.running[name] = dict(proc=None, gpus=spec.get('gpu_ids', []), started=spec.get('started', time.time()),
                                      attempt=spec.get('attempt', 1), adopted=True, pgid=spec.get('pgid'))
            self.attempts[name] = spec.get('attempt', 1)
            self.log(f'adopted running job {name} (GPUs {spec.get("gpu_ids", [])})')

    def launch(self, name, gpus):
        job = self.jobs[name]
        attempt = self.attempts.get(name, 0) + 1
        self.attempts[name] = attempt
        for suffix in ('exit', 'progress'):
            self.path(name, suffix).unlink(missing_ok=True)
        env = {k: str(v) for k, v in job.env.items()}
        env.update(REDAG_JOB=name, REDAG_PROGRESS_FILE=str(self.path(name, 'progress')))
        if job.gpus:
            env['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpus))
        elif self.gpu_list():
            env['CUDA_VISIBLE_DEVICES'] = ''  # CPU jobs must not grab a GPU
        spec = dict(name=name, cmd=job.cmd, env=env, cwd=self.cwd, attempt=attempt, gpu_ids=list(gpus), gpus=','.join(map(str, gpus)),
                    stamp=job.stamp, lock=str(self.path(name, 'lock')), exit=str(self.path(name, 'exit')),
                    done=str(self.done_marker(name)), log=str(self.logs / f'{name}.log'),
                    timings=str(self.logs / 'timings.jsonl'), started=time.time())
        spec_path = self.path(name, 'json')
        atomic_write(spec_path, json.dumps(spec, indent=1))
        proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '_exec', str(spec_path)], cwd=self.cwd,
                                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        spec['pgid'] = proc.pid
        atomic_write(spec_path, json.dumps(spec, indent=1))
        self.running[name] = dict(proc=proc, gpus=list(gpus), started=spec['started'], attempt=attempt, adopted=False, pgid=proc.pid)
        self.log(f'start {name} (attempt {attempt}' + (f', GPU {spec["gpus"]})' if gpus else ', CPU)'))

    def check_running(self):
        for name in list(self.running):
            info = self.running[name]
            if info['proc'] is not None:
                if info['proc'].poll() is None:
                    continue
            elif lock_held(self.path(name, 'lock')):
                continue
            del self.running[name]
            exit_info = read_json(self.path(name, 'exit'), {})
            status = exit_info.get('status') if exit_info.get('attempt') == info['attempt'] else None
            minutes = (time.time() - info['started']) / 60
            if status == 0 and self.is_done(name):
                self.log(f'done  {name} in {minutes:.1f} min')
                self.path(name, 'failed').unlink(missing_ok=True)
                self.events.append(('done', name))
                continue
            if self.stopping:
                self.log(f'stopped {name}; it resumes from its last checkpoint on the next run')
                continue
            if self.path(name, 'restart').exists():  # `pipeline.py restart`: new code for a running job, not a failure
                self.path(name, 'restart').unlink(missing_ok=True)
                self.attempts[name] = max(0, self.attempts.get(name, 1) - 1)
                self.retry_at.pop(name, None)
                self.log(f'restart {name} (requested; resumes from its last checkpoint, not counted as an attempt)')
                continue
            reason = f'exit status {status}' if status is not None else 'interrupted (no exit status)'
            tail = tail_lines(self.logs / f'{name}.log', 12)
            if self.attempts.get(name, 1) < self.jobs[name].attempts:
                self.log(f'FAIL  {name} ({reason}) after {minutes:.1f} min; retrying in {self.retry_delay:.0f} s\n{tail}')
                self.retry_at[name] = time.time() + self.retry_delay
            else:
                self.log(f'FAIL  {name} ({reason}); attempts exhausted\n{tail}')
                atomic_write(self.path(name, 'failed'), json.dumps(dict(reason=reason, code=self.code_version, time=time.time(),
                                                                         attempts=self.attempts.get(name, 1), tail=tail)))
                self.events.append(('failed', name))
                self.notifier.send(f'job failed: {name}', f'{reason} after {minutes:.0f} min\n{tail[-1500:]}')

    # ---------------------------------------------------------------- scheduling
    def classify(self, now=None):
        """{name: state} with state in done/running/failed/blocked/waiting/ready/held."""
        now = now or time.time()
        states = {}
        for name in self.order:
            job = self.jobs[name]
            if self.is_done(name):
                states[name] = 'done'
            elif name in self.running:
                states[name] = 'running'
            elif self.is_failed(name):
                states[name] = 'failed'
            elif any(states[d] in ('failed', 'blocked') for d in job.deps):
                states[name] = 'blocked'
            elif any(states[d] != 'done' for d in job.deps):
                states[name] = 'waiting'
            elif job.ready is not None and not safe_call(job.ready):
                states[name] = 'held'
            elif job.gpus > len(self.gpu_list()):
                states[name] = 'held'  # needs more GPUs than this machine has
            elif now < self.retry_at.get(name, 0):
                states[name] = 'held'
            else:
                states[name] = 'ready'
        return states

    def held_by_condition(self, name):
        """Held by a condition that can change from outside (another machine's job, a claim, a decision gate, an RL
        checkpoint): keep waiting for it instead of ending as stalled.  Not on a machine that could never run the job."""
        job = self.jobs[name]
        return job.ready is not None and job.gpus <= len(self.gpu_list()) and not safe_call(job.ready)

    def priorities(self, est):
        return bottom_levels(list(self.jobs.values()), {n: est[n] / HOUR for n in self.jobs})

    def schedule(self, states, est, level, now, draining=False):
        if draining:
            return
        ready = sorted((n for n, s in states.items() if s == 'ready'), key=lambda n: (-level[n], n))
        free = self.free_gpus()
        cpu_slots = self.cpu_slots or (2 if self.gpu_list() else 1)
        cpu_free = cpu_slots - sum(1 for n in self.running if self.jobs[n].gpus == 0)
        running = [(self.running[n]['started'] + est[n], len(self.running[n]['gpus'])) for n in self.running if self.running[n]['gpus']]
        upcoming = self.upcoming(states, est, level, now)
        starts = choose([(self.jobs[n], est[n], level[n]) for n in ready], len(free), running, now, cpu_free, upcoming)
        for job in starts:
            if self.claim is not None and not self.claim(job.name):
                self.log(f'{job.name} was claimed by another machine')
                continue
            take = free[:job.gpus]
            free = free[job.gpus:]
            self.launch(job.name, take)

    def upcoming(self, states, est, level, now):
        """When the critical path binds (many GPUs), reserve GPUs for the next critical multi-GPU job before it is ready."""
        gpus = len(self.gpu_list())
        if gpus < 2:
            return None
        open_jobs = [n for n, s in states.items() if s not in ('done', 'failed', 'blocked')]
        work = sum(est[n] * max(self.jobs[n].gpus, 0) for n in open_jobs) / gpus
        critical = max((level[n] * HOUR for n in open_jobs), default=0)
        if critical < 0.8 * work:
            return None
        best = None
        for name in open_jobs:
            job = self.jobs[name]
            if states[name] != 'waiting' or job.gpus < 2:
                continue
            pending = [d for d in job.deps if states[d] != 'done']
            if not all(states[d] == 'running' for d in pending):
                continue
            ready_at = max(self.running[d]['started'] + est[d] for d in pending)
            if best is None or level[name] > best[2]:
                best = (job, ready_at, level[name])
        return best

    # ---------------------------------------------------------------- status
    def status(self, states=None, est=None, now=None):
        now = now or time.time()
        states = states or self.classify(now)
        est = est or self.estimates(now)
        jobs = {}
        for name in self.order:
            job = self.jobs[name]
            row = {'state': states[name], 'group': job.group, 'gpus': job.gpus, 'est_hours': round(est[name] / HOUR, 2)}
            if name in self.running:
                info = self.running[name]
                left = self.remaining(name, now, est[name])
                row.update(gpu_ids=info['gpus'], attempt=info['attempt'], elapsed_hours=round((now - info['started']) / HOUR, 2),
                           eta=round(now + left), left_hours=round(left / HOUR, 2), progress=self.progress(name))
            if states[name] == 'failed':
                row['failed'] = self.failed_info(name)
            if states[name] == 'held' and job.ready is not None and not safe_call(job.ready):
                row['waiting_for'] = job.waiting_for
            if states[name] == 'held' and job.gpus > len(self.gpu_list()):
                row['waiting_for'] = f'needs {job.gpus} GPUs, this machine has {len(self.gpu_list())}'
            jobs[name] = row
        projected = self.project(states, est, now)
        counts = {}
        for s in states.values():
            counts[s] = counts.get(s, 0) + 1
        return dict(updated=now, host=socket.gethostname(), code=self.code_version, gpus=self.gpu_info or [{'index': g} for g in self.gpu_list()],
                    foreign_busy_gpus=sorted(self.foreign), counts=counts, jobs=jobs, projected_finish=projected,
                    gpu_hours_left=round(sum(est[n] * self.jobs[n].gpus for n, s in states.items() if s not in ('done', 'failed', 'blocked')) / HOUR, 1))

    def project(self, states, est, now):
        gpus = len(self.gpu_list())
        if not gpus:
            return None
        done = [n for n, s in states.items() if s == 'done']
        running = [(n, max(self.remaining(n, now, est[n]), 1.0), self.jobs[n].gpus) for n in self.running]
        open_jobs = [j for j in self.jobs.values() if states[j.name] not in ('failed', 'blocked')]
        keep = {j.name for j in open_jobs}
        trimmed = [Job(**{**j.__dict__, 'deps': tuple(d for d in j.deps if d in keep),
                          'soft': () if states[j.name] in ('ready', 'running') or (j.ready is not None and safe_call(j.ready))
                          else tuple((d, f) for d, f in j.soft if d in keep)}) for j in open_jobs]
        try:
            timeline = simulate(trimmed, est, gpus, done=[n for n in done if n in keep],
                                running=[r for r in running if r[0] in keep], now=0.0)
        except (RuntimeError, ValueError):
            return None
        if not timeline:
            return now
        return round(now + max(end for _, end in timeline.values()))

    def write_status(self, states, est, now):
        try:
            atomic_write(self.state / 'status.json', json.dumps(self.status(states, est, now), indent=1, ensure_ascii=False))
        except OSError as error:
            self.log(f'cannot write status: {error}')

    # ---------------------------------------------------------------- main loop
    def control(self):
        value = (self.state / 'control').read_text().strip() if (self.state / 'control').exists() else ''
        return value

    def stop_running(self, wait=90):
        self.stopping = True
        for name, info in self.running.items():
            pgid = info.get('pgid')
            if pgid:
                try:
                    os.killpg(pgid, signal.SIGTERM)
                    self.log(f'sent SIGTERM to {name}')
                except ProcessLookupError:
                    pass
        deadline = time.time() + wait
        while self.running and time.time() < deadline:
            time.sleep(2)
            self.check_running()
        for name, info in list(self.running.items()):
            if info.get('pgid'):
                try:
                    os.killpg(info['pgid'], signal.SIGKILL)
                except ProcessLookupError:
                    pass

    def run(self, on_finish=None):
        lock_fd = os.open(self.state / 'scheduler.lock', os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.log('another scheduler is already running (runs/state/scheduler.lock); nothing to do')
            return 'already-running'
        atomic_write(self.state / 'scheduler.pid', f'{os.getpid()}\n')
        (self.state / 'control').unlink(missing_ok=True)
        self.refresh_gpus()
        self.adopt()
        for name in list(self.running):  # jobs started before sharing was configured: record that they run here
            if self.claim is not None and not self.claim(name):
                self.log(f'WARNING: {name} runs here but another machine claimed it')
        started = time.time()
        self.log(f'scheduler started: {len(self.gpu_list())} GPU(s) {self.gpu_list()}, {len(self.jobs)} jobs, code {self.code_version}')
        first = True
        outcome = 'stalled'
        while True:
            now = time.time()
            self.refresh_gpus()
            self.check_running()
            control = self.control()
            if control == 'stop':
                self.log('stop requested: terminating running jobs (they resume from their last checkpoint)')
                self.stop_running()
                outcome = 'stopped'
                break
            states = self.classify(now)
            est = self.estimates(now)
            level = self.priorities(est)
            if self.max_hours and now - started > self.max_hours * HOUR and control != 'drain':
                atomic_write(self.state / 'control', 'drain\n')
                self.log(f'time budget of {self.max_hours} h reached: draining')
                control = 'drain'
            self.schedule(states, est, level, now, draining=control == 'drain')
            states = self.classify(now)
            self.write_status(states, est, now)
            if first:
                first = False
                status = read_json(self.state / 'status.json', {})
                finish = status.get('projected_finish')
                self.notifier.send('scheduler started', f'{len(self.gpu_list())} GPU(s); {status.get("counts")}; projected finish '
                                   + (time.strftime('%m-%d %H:%M', time.localtime(finish)) if finish else 'n/a'))
            for kind, name in self.events:
                if kind == 'done' and name in MILESTONES:
                    self.notifier.send(f'milestone: {name}', MILESTONES[name])
            self.events.clear()
            if self.tick is not None and now - self.last_tick >= self.tick_every:
                self.last_tick = now
                try:
                    self.tick()
                except Exception as error:  # a broken report must never stop the pipeline
                    self.log(f'periodic hook failed: {error!r}')
            if not self.running:
                if all(v == 'done' for v in states.values()):
                    outcome = 'complete'
                    break
                if control == 'drain':
                    outcome = 'drained'
                    break
                if not any(v == 'ready' or (v == 'held' and (self.retry_at.get(n, 0) > now or self.held_by_condition(n)))
                           for n, v in states.items()):
                    outcome = 'stalled'  # only failed / blocked jobs, or jobs this machine cannot run, are left
                    break
            time.sleep(self.poll)
        states = self.classify()
        self.write_status(states, self.estimates(), time.time())
        summary = ', '.join(f'{k} {v}' for k, v in sorted(read_json(self.state / 'status.json', {}).get('counts', {}).items()))
        self.log(f'scheduler finished: {outcome} ({summary})')
        if outcome in ('complete', 'stalled'):
            detail = summary
            if outcome == 'stalled':
                stuck = [n for n, s in states.items() if s in ('failed', 'held')]
                detail += '\nnot runnable: ' + ', '.join(stuck[:12])
            self.notifier.send('pipeline complete' if outcome == 'complete' else 'pipeline stalled', detail)
        (self.state / 'control').unlink(missing_ok=True)
        if on_finish:
            on_finish(outcome)
        os.close(lock_fd)
        return outcome


MILESTONES = {}  # job name -> message sent when it finishes (filled by pipeline.py)


def safe_call(fn):
    try:
        return bool(fn())
    except Exception:
        return False


def read_jsonl(path):
    rows = []
    try:
        with open(path, encoding='utf-8') as stream:
            for line in stream:
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    continue
    except OSError:
        pass
    return rows


def tail_lines(path, n):
    try:
        with open(path, 'rb') as stream:
            stream.seek(0, 2)
            size = stream.tell()
            stream.seek(max(size - 8192, 0))
            lines = stream.read().decode('utf-8', 'replace').splitlines()
        return '\n'.join(lines[-n:])
    except OSError:
        return ''


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '_exec':
        sys.exit(execute(sys.argv[2]))
    print(__doc__)
