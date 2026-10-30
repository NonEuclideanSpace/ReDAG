"""Scheduler tests with fake jobs (standard library only; no torch, no GPUs needed)."""
import json
import os
import signal
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts' / 'launch'))
from orchestrator import Job, Scheduler, choose, lock_held, reservation, simulate  # noqa: E402

PY = sys.executable


def py(code):
    return [PY, '-c', textwrap.dedent(code)]


def make(tmp_path, jobs, **kwargs):
    kwargs.setdefault('gpus', [0, 1])
    kwargs.setdefault('poll', 0.05)
    kwargs.setdefault('gpu_free_mib', None)
    return Scheduler(jobs, tmp_path / 'runs', tmp_path, log=lambda text: None, **kwargs)


def test_reservation_and_backfill():
    small, big = Job('small', 'x', gpus=1), Job('big', 'x', gpus=2)
    # one GPU free, the other busy for one more hour: the 2-GPU job waits, a 30-min job may use the free GPU, a 2-h job may not
    assert reservation(2, 1, [(3600, 1)], 0) == (3600, 0)
    assert choose([(big, 7200, 10), (small, 1800, 5)], 1, [(3600, 1)], 0, 2) == [small]
    assert choose([(big, 7200, 10), (small, 7200, 5)], 1, [(3600, 1)], 0, 2) == []
    # a more important small job is never held back by the reservation of a less important one
    assert choose([(small, 7200, 20), (big, 7200, 10)], 1, [(3600, 1)], 0, 2) == [small]
    cpu = Job('cpu', 'x')
    assert choose([(cpu, 10, 1), (Job('cpu2', 'x'), 10, 1)], 0, [], 0, 1) == [cpu]


def test_simulation_packs_gpus():
    jobs = [Job('a', 'x', gpus=1, hours=1), Job('b', 'x', gpus=1, hours=1), Job('c', 'x', gpus=2, hours=1, deps=('a', 'b')),
            Job('d', 'x', gpus=1, hours=0.5, soft=(('c', 0.5),))]
    est = {j.name: j.hours * 3600 for j in jobs}
    two = simulate(jobs, est, 2)
    assert max(end for _, end in two.values()) == pytest.approx(2.5 * 3600)  # a||b, c, then d (c holds both GPUs)
    four = simulate(jobs, est, 4)
    assert four['d'][0] == pytest.approx(1.5 * 3600)  # starts half way through c on a spare GPU
    one = simulate(jobs, est, 1)
    assert 'c' not in one  # needs two GPUs


def test_runs_dag_with_gpu_pinning_and_retry(tmp_path):
    out = tmp_path / 'out'
    out.mkdir()
    record = f"import os, time; open(r'{out}/' + os.environ['REDAG_JOB'], 'w').write(os.environ.get('CUDA_VISIBLE_DEVICES', '-')); time.sleep(0.3)"
    flaky = f"""
        import os, sys
        p = r'{out}/flaky.count'
        n = int(open(p).read()) if os.path.exists(p) else 0
        open(p, 'w').write(str(n + 1))
        sys.exit(1 if n == 0 else 0)"""
    jobs = [Job('a', py(record)), Job('b', py(record), gpus=1, deps=('a',)), Job('c', py(record), gpus=1, deps=('a',)),
            Job('d', py(record), gpus=2, deps=('b', 'c')), Job('e', py(flaky), gpus=1, deps=('a',), attempts=2)]
    scheduler = make(tmp_path, jobs)
    scheduler.retry_delay = 0
    outcome = scheduler.run()
    assert outcome == 'complete', (tmp_path / 'runs' / 'logs' / 'e.log').read_text()
    gpus = {name: (out / name).read_text() for name in 'abcd'}
    assert gpus['a'] == '' and {gpus['b'], gpus['c']} == {'0', '1'} and gpus['d'] in ('0,1', '1,0')
    assert (out / 'flaky.count').read_text() == '2'
    timings = [json.loads(line) for line in (tmp_path / 'runs' / 'logs' / 'timings.jsonl').read_text().splitlines()]
    assert sorted(r['job'] for r in timings) == ['a', 'b', 'c', 'd', 'e', 'e']
    status = json.loads((tmp_path / 'runs' / 'state' / 'status.json').read_text())
    assert status['counts'] == {'done': 5}


def test_failure_blocks_only_dependents_and_is_not_retried_without_changes(tmp_path):
    jobs = [Job('bad', py('import sys; sys.exit(3)'), gpus=1, attempts=1), Job('after', py('pass'), gpus=1, deps=('bad',)),
            Job('other', py('pass'), gpus=1)]
    assert make(tmp_path, jobs, code_version='v1').run() == 'stalled'
    runs = tmp_path / 'runs'
    assert (runs / 'logs' / 'other.done').exists() and not (runs / 'logs' / 'after.done').exists()
    assert json.loads((runs / 'state' / 'jobs' / 'bad.failed').read_text())['reason'] == 'exit status 3'
    again = make(tmp_path, jobs, code_version='v1')
    assert again.classify()['bad'] == 'failed' and again.classify()['after'] == 'blocked'
    assert make(tmp_path, jobs, code_version='v2').classify()['bad'] == 'ready'   # code changed: try again
    assert make(tmp_path, jobs, code_version='v1', retry_failed=True).classify()['bad'] == 'ready'


def test_restarted_scheduler_adopts_running_job(tmp_path):
    out = tmp_path / 'started.txt'
    job_code = f"import time; open(r'{out}', 'a').write('x'); time.sleep(2.5)"
    driver = tmp_path / 'driver.py'
    driver.write_text(textwrap.dedent(f"""
        import sys
        sys.path.insert(0, r'{Path(__file__).resolve().parents[1] / 'scripts' / 'launch'}')
        from orchestrator import Job, Scheduler
        jobs = [Job('long', {py(job_code)!r}, gpus=1), Job('next', {py('pass')!r}, gpus=1, deps=('long',))]
        Scheduler(jobs, r'{tmp_path / 'runs'}', r'{tmp_path}', gpus=[0], poll=0.05, gpu_free_mib=None, log=print).run()
    """))
    first = subprocess.Popen([PY, str(driver)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    lock = tmp_path / 'runs' / 'state' / 'jobs' / 'long.lock'
    deadline = time.time() + 20
    while not (lock.exists() and lock_held(lock)) and time.time() < deadline:
        time.sleep(0.05)
    first.send_signal(signal.SIGKILL)  # the scheduler dies; the job keeps running in its own session
    first.wait()
    assert lock_held(lock)
    jobs = [Job('long', py(job_code), gpus=1), Job('next', py('pass'), gpus=1, deps=('long',))]
    second = make(tmp_path, jobs, gpus=[0])
    assert second.run() == 'complete'
    assert out.read_text() == 'x'  # adopted, not started twice
    assert (tmp_path / 'runs' / 'logs' / 'long.log').read_text().count('attempt 1') == 1


def test_stamp_change_reruns_and_ready_condition_holds_job(tmp_path):
    flag = tmp_path / 'flag'
    jobs = [Job('pre', py('pass'), gpus=1, stamp='v1'),
            Job('maker', py(f"import time; time.sleep(0.4); open(r'{flag}', 'w').write('1')"), gpus=1),
            Job('needs_flag', py('pass'), gpus=1, ready=flag.exists, waiting_for='flag file')]
    scheduler = make(tmp_path, jobs)
    assert scheduler.classify()['needs_flag'] == 'held'
    assert scheduler.run() == 'complete'
    assert make(tmp_path, jobs).classify()['pre'] == 'done'
    jobs[0].stamp = 'v2'
    assert make(tmp_path, jobs).classify()['pre'] == 'ready'


def test_stop_request_terminates_without_marking_failure(tmp_path):
    jobs = [Job('slow', py('import time; time.sleep(30)'), gpus=1, attempts=1)]
    scheduler = make(tmp_path, jobs)
    result = {}
    thread = threading.Thread(target=lambda: result.update(outcome=scheduler.run()))
    thread.start()
    lock = tmp_path / 'runs' / 'state' / 'jobs' / 'slow.lock'
    deadline = time.time() + 20
    while not (lock.exists() and lock_held(lock)) and time.time() < deadline:
        time.sleep(0.05)
    (tmp_path / 'runs' / 'state' / 'control').write_text('stop\n')
    thread.join(60)
    assert result['outcome'] == 'stopped'
    assert not (tmp_path / 'runs' / 'state' / 'jobs' / 'slow.failed').exists()
    assert make(tmp_path, jobs).classify()['slow'] == 'ready'


def test_second_scheduler_refuses_to_run(tmp_path):
    jobs = [Job('slow', py('import time; time.sleep(1.5)'), gpus=1)]
    first = make(tmp_path, jobs)
    thread = threading.Thread(target=first.run)
    thread.start()
    time.sleep(0.5)
    assert make(tmp_path, jobs).run() == 'already-running'
    thread.join(30)


def test_cpu_only_machine_runs_cpu_jobs_then_stops(tmp_path):
    jobs = [Job('download', py('pass')), Job('train', py('pass'), gpus=1, deps=('download',))]
    scheduler = make(tmp_path, jobs, gpus=[])
    assert scheduler.run() == 'stalled'
    states = make(tmp_path, jobs, gpus=[]).classify()
    assert states == {'download': 'done', 'train': 'held'}


def test_requested_restart_relaunches_without_using_an_attempt(tmp_path):
    count = tmp_path / 'count.txt'
    # first run: wait to be killed; second run (after the restart): finish
    code = f"""
        import os, time
        path = r'{count}'
        n = int(open(path).read()) if os.path.exists(path) else 0
        open(path, 'w').write(str(n + 1))
        if n == 0:
            time.sleep(30)
    """
    jobs = [Job('job', py(code), gpus=1, attempts=1)]  # one attempt: a counted failure would mark it failed
    scheduler = make(tmp_path, jobs)
    result = {}
    thread = threading.Thread(target=lambda: result.update(outcome=scheduler.run()))
    thread.start()
    state = tmp_path / 'runs' / 'state' / 'jobs'
    deadline = time.time() + 20
    while not (count.exists() and lock_held(state / 'job.lock')) and time.time() < deadline:
        time.sleep(0.05)
    pgid = json.loads((state / 'job.json').read_text())['pgid']
    (state / 'job.restart').write_text('now\n')  # what `pipeline.py restart job` does
    os.killpg(pgid, signal.SIGTERM)
    thread.join(60)
    assert result['outcome'] == 'complete'
    assert count.read_text() == '2'
    assert not (state / 'job.failed').exists() and not (state / 'job.restart').exists()
