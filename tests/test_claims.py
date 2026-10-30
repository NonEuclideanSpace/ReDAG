"""Several machines, one pipeline: shared jobs are claimed atomically, NOT_BEFORE orders side jobs, and a scheduler
waiting for another machine keeps running instead of ending as stalled (standard library only)."""
import argparse
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts' / 'launch'))
import pipeline  # noqa: E402
from orchestrator import Job, Scheduler  # noqa: E402

PY = sys.executable


def test_local_claims_are_exclusive_and_releasable(tmp_path):
    a = pipeline.Claims(tmp_path / 'runs', me='machine-a')
    b = pipeline.Claims(tmp_path / 'runs', me='machine-b')
    assert a.claim('canvases_shard0') and a.claim('canvases_shard0')   # idempotent for the owner (restarts)
    assert not b.claim('canvases_shard0') and b.owner('canvases_shard0') == 'machine-a'
    assert b.claim('canvases_shard1') and a.owner('canvases_shard1') == 'machine-b'
    assert a.release('canvases_shard0') and b.claim('canvases_shard0')


def test_remote_claim_commands(tmp_path):
    """The shell snippets sent over ssh, run locally: atomic claim, listing, release."""
    folder = tmp_path / 'claims host'

    class Local(pipeline.Claims):
        def ssh(self, command):
            return subprocess.run(['bash', '-c', command], capture_output=True, text=True).stdout

    a = Local(tmp_path / 'runs', host='root@peer:2222', me='machine-a', ttl=0)
    b = Local(tmp_path / 'runs', host='root@peer:2222', me='machine-b', ttl=0)
    for c in (a, b):
        c.remote = pipeline.shlex.quote(str(folder))
    assert a.claim('rs_part1') and not b.claim('rs_part1')
    assert b.owners() == {'rs_part1': 'machine-a'} and b.owner('rs_part1') == 'machine-a'
    assert b.release('rs_part1') and b.claim('rs_part1') and a.owner('rs_part1') == 'machine-b'


def test_unreachable_claims_host_falls_back_after_a_while(tmp_path):
    class Down(pipeline.Claims):
        def ssh(self, command):
            if self.unreachable_since is None:
                self.unreachable_since = time.time()
            return None
    c = Down(tmp_path / 'runs', host='root@peer:2222', me='machine-a', fallback_minutes=0.0005)
    c.cache = {'rs_part0': 'machine-b'}
    assert not c.claim('rs_part1')            # first failure: do not start anything yet
    time.sleep(0.05)
    assert c.claim('rs_part1') and c.owner('rs_part0') is None   # still unreachable after the window: run anyway


def clean_env(monkeypatch):
    for name in ('CLAIMS_HOST', 'LOCAL_JOBS', 'EXTERNAL_JOBS', 'SHARED_JOBS', 'NOT_BEFORE', 'HOLD', 'SYNC_PEER'):
        monkeypatch.delenv(name, raising=False)  # a shell that sourced env.sh / local.env must not leak into the test


def test_shared_jobs_and_not_before_wiring(tmp_path, monkeypatch):
    clean_env(monkeypatch)
    monkeypatch.setenv('RUNS', str(tmp_path / 'runs'))
    monkeypatch.setenv('EXTERNAL_JOBS', '^(sft_editor|annotate_shard[0-7])$')
    monkeypatch.setenv('SHARED_JOBS', '^(annotate_shard[0-7]|canvases_shard[0-3]|rs_part[0-3])$')
    monkeypatch.setenv('NOT_BEFORE', '^(sft_planner|eval_planner_zero_shot)$>rs_merge')
    monkeypatch.setattr(pipeline, 'query_gpus', lambda: [dict(index=0, name='GPU', used=0, total=97887)])
    s = pipeline.settings()
    scheduler, jobs, _, _ = pipeline.make_scheduler(argparse.Namespace(), s, quiet=True)
    by_name = {j.name: j for j in jobs}
    claims_dir = Path(s['RUNS']) / 'state' / 'claims'
    assert by_name['annotate_shard3'].ready()                  # shared wins over EXTERNAL_JOBS
    assert not by_name['sft_editor'].ready()                   # still external
    claims_dir.mkdir(parents=True)
    (claims_dir / 'canvases_shard2').write_text('other-machine')
    assert not by_name['canvases_shard2'].ready() and 'claimed by other-machine' in by_name['canvases_shard2'].waiting_for
    assert scheduler.claim('canvases_shard1') and not scheduler.claim('canvases_shard2') and scheduler.claim('sft_planner')
    assert not by_name['sft_planner'].ready() and 'rs_merge (NOT_BEFORE)' in by_name['sft_planner'].waiting_for
    (Path(s['RUNS']) / 'logs' / 'rs_merge.done').write_text('')
    assert by_name['sft_planner'].ready() and by_name['eval_planner_sft'].ready is None


def test_scheduler_skips_jobs_claimed_elsewhere_and_waits_for_other_machines(tmp_path):
    flag = tmp_path / 'other_machine_done'
    started = tmp_path / 'started'
    jobs = [Job('mine', [PY, '-c', f"open(r'{started}', 'a').write('mine ')"], gpus=1),
            Job('theirs', [PY, '-c', f"open(r'{started}', 'a').write('theirs ')"], gpus=1),
            Job('after_peer', [PY, '-c', 'pass'], gpus=1, ready=flag.exists, waiting_for='peer')]
    scheduler = Scheduler(jobs, tmp_path / 'runs', tmp_path, gpus=[0], poll=0.05, gpu_free_mib=None, log=lambda text: None,
                          claim=lambda name: name != 'theirs')
    result = {}
    thread = threading.Thread(target=lambda: result.update(outcome=scheduler.run()))
    thread.start()
    time.sleep(1.0)
    assert thread.is_alive()                  # waits for the condition instead of ending as stalled
    (tmp_path / 'runs' / 'logs' / 'theirs.done').write_text('')   # what the sync brings from the other machine
    flag.write_text('1')
    thread.join(30)
    assert result['outcome'] == 'complete' and started.read_text() == 'mine '


def test_sync_compares_modification_times_to_the_nanosecond(monkeypatch):
    seen = []
    monkeypatch.setattr(pipeline.subprocess, 'run', lambda cmd, **kwargs: seen.append(cmd))
    pipeline.rsync('a/', 'b/', 'ssh', [])
    assert '--update' in seen[0] and '--modify-window=-1' in seen[0]


def test_sync_markers_follow_jobs_added_later(monkeypatch, tmp_path):
    for name in ('CANVAS_SHARDS', 'RS_SHARDS'):
        monkeypatch.delenv(name, raising=False)
    calls = []
    monkeypatch.setattr(pipeline, 'rsync', lambda src, dst, ssh, filters, dry=False: calls.append(filters) or type('R', (), {'returncode': 0})())
    pipeline.cmd_sync(type('A', (), {'peer': 'root@h:1', 'key': 'k', 'remote_root': '/r', 'loop': 0, 'dry_run': False})(), pipeline.settings())
    finals = calls[1]
    assert '--include=runs/logs/*.done' in finals and '--exclude=runs/logs/preflight.done' in finals
    assert '--exclude=runs/logs/download_models.done' in finals and finals.index('--exclude=runs/logs/preflight.done') < finals.index('--include=runs/logs/*.done')
