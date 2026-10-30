"""Several machines, one pipeline: shared jobs are claimed atomically, NOT_BEFORE orders side jobs, and a scheduler
waiting for another machine keeps running instead of ending as stalled (standard library only)."""
import argparse
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

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


def test_interrupted_transfers_never_leave_a_cut_off_file_under_the_real_name(monkeypatch):
    seen = []
    monkeypatch.setattr(pipeline.subprocess, 'run', lambda cmd, **kwargs: seen.append(cmd))
    pipeline.rsync('a/', 'b/', 'ssh', ['--exclude=*'])
    cmd = seen[0]
    assert '--partial' not in cmd and '--partial-dir=.rsync-partial' in cmd
    assert cmd.index('--exclude=.rsync-partial/') < cmd.index('--exclude=*')


def fake_rsync(calls, stdout=''):
    def run(src, dst, ssh, filters, dry=False):
        calls.append(filters)
        return type('R', (), {'returncode': 0, 'stdout': stdout, 'stderr': ''})()
    return run


def nothing_being_written(monkeypatch):
    monkeypatch.setattr(pipeline, 'fresh_remote', lambda host, ssh, root, seconds: [])
    monkeypatch.setattr(pipeline, 'fresh_local', lambda root, seconds: [])


def test_sync_markers_follow_jobs_added_later(monkeypatch, tmp_path):
    for name in ('CANVAS_SHARDS', 'RS_SHARDS'):
        monkeypatch.delenv(name, raising=False)
    calls = []
    monkeypatch.setattr(pipeline, 'rsync', fake_rsync(calls))
    nothing_being_written(monkeypatch)
    pipeline.cmd_sync(type('A', (), {'peer': 'root@h:1', 'key': 'k', 'remote_root': '/r', 'loop': 0, 'dry_run': False})(), pipeline.settings())
    finals = calls[0]
    assert '--out-format=%n' in finals
    assert '--include=runs/logs/*.done' in finals and '--exclude=runs/logs/preflight.done' in finals
    assert '--exclude=runs/logs/download_models.done' in finals and finals.index('--exclude=runs/logs/preflight.done') < finals.index('--include=runs/logs/*.done')
    assert finals.index('--exclude=runs/state/jobs/preflight.progress') < finals.index('--include=runs/state/jobs/*.progress')
    assert len(calls) == 4 and '--exclude=runs/logs/**' in calls[1]


def test_sync_sends_only_the_markers_listed_before_the_outputs(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(pipeline, 'rsync', fake_rsync(calls, 'runs/\nruns/logs/\nruns/logs/rs_part1.done\nruns/logs/rs_part1.log\n'))
    nothing_being_written(monkeypatch)
    listings = []
    real_unlink = pipeline.os.unlink

    def keep(path):
        if str(path).endswith('.list'):
            listings.append(Path(path).read_text())
        real_unlink(path)

    monkeypatch.setattr(pipeline.os, 'unlink', keep)
    pipeline.cmd_sync(type('A', (), {'peer': 'root@h:1', 'key': 'k', 'remote_root': '/r', 'loop': 0, 'dry_run': False})(), pipeline.settings())
    assert len(calls) == 6 and '--exclude=runs/logs/**' in calls[1]
    assert any(f.startswith('--files-from=') for f in calls[2]) and '--out-format=%n' not in calls[2]
    assert listings == ['runs/logs/rs_part1.done\nruns/logs/rs_part1.log\n'] * 2


def test_sync_holds_back_a_marker_written_after_the_outputs_were_copied(monkeypatch, tmp_path):
    if shutil.which('rsync') is None:
        pytest.skip('rsync not installed')
    remote, local = tmp_path / 'remote', tmp_path / 'local'
    for root in (remote, local):
        (root / 'runs' / 'logs').mkdir(parents=True)
        (root / 'data' / 'final').mkdir(parents=True)
    (remote / 'data' / 'final' / 'a.jsonl').write_text('a\n')
    (remote / 'runs' / 'logs' / 'job_a.done').write_text('')
    real, raced = pipeline.rsync, []

    def racing(src, dst, ssh, filters, dry=False):
        result = real(src.replace('root@h:', ''), dst.replace('root@h:', ''), ssh, filters, dry)
        if '--exclude=runs/logs/**' in filters and src.startswith('root@h:') and not raced:
            raced.append(1)
            (remote / 'data' / 'final' / 'b.jsonl').write_text('b\n')
            (remote / 'runs' / 'logs' / 'job_b.done').write_text('')
        return result

    monkeypatch.setattr(pipeline, 'rsync', racing)
    monkeypatch.setattr(pipeline, 'ROOT', local)
    nothing_being_written(monkeypatch)
    args = type('A', (), {'peer': 'root@h:1', 'key': 'k', 'remote_root': str(remote), 'loop': 0, 'dry_run': False})()
    pipeline.cmd_sync(args, pipeline.settings())
    assert (local / 'runs' / 'logs' / 'job_a.done').exists() and (local / 'data' / 'final' / 'a.jsonl').read_text() == 'a\n'
    assert not (local / 'runs' / 'logs' / 'job_b.done').exists()
    pipeline.cmd_sync(args, pipeline.settings())
    assert (local / 'runs' / 'logs' / 'job_b.done').exists() and (local / 'data' / 'final' / 'b.jsonl').read_text() == 'b\n'


def test_unstarted_rejection_sampling_parts_split_into_pieces(monkeypatch):
    clean_env(monkeypatch)
    monkeypatch.setenv('RS_SHARDS', '6')
    monkeypatch.setenv('RS_SPLIT_PARTS', '3 5')
    monkeypatch.setenv('RS_PIECES', '3')
    jobs, _ = pipeline.build_jobs(pipeline.settings())
    by_name = {j.name: j for j in jobs}
    assert 'rs_part2' in by_name and 'rs_part3' not in by_name and 'rs_part4' in by_name
    piece = by_name['rs_part3_1'].cmd
    assert piece[piece.index('--piece') + 1] == '1' and piece[piece.index('--pieces') + 1] == '3'
    assert piece[piece.index('--seed') + 1] == str(42 + 3 * 100000) and piece[piece.index('--data') + 1].endswith('controller_rs_states.part3.jsonl')
    merge = by_name['rs_merge']
    outputs = [Path(c).name for c in merge.cmd if 'controller_sft.part' in c]
    assert outputs == ['controller_sft.part0.jsonl', 'controller_sft.part1.jsonl', 'controller_sft.part2.jsonl', 'controller_sft.part3_0.jsonl',
                       'controller_sft.part3_1.jsonl', 'controller_sft.part3_2.jsonl', 'controller_sft.part4.jsonl', 'controller_sft.part5_0.jsonl',
                       'controller_sft.part5_1.jsonl', 'controller_sft.part5_2.jsonl']
    assert set(merge.deps) == {'rs_part0', 'rs_part1', 'rs_part2', 'rs_part3_0', 'rs_part3_1', 'rs_part3_2', 'rs_part4', 'rs_part5_0',
                               'rs_part5_1', 'rs_part5_2'}


def test_sync_leaves_files_still_being_written_for_the_next_round(monkeypatch, tmp_path):
    if shutil.which('rsync') is None:
        pytest.skip('rsync not installed')
    remote, local = tmp_path / 'remote', tmp_path / 'local'
    for root in (remote, local):
        (root / 'runs' / 'logs').mkdir(parents=True)
        (root / 'data' / 'final').mkdir(parents=True)
    old = time.time() - 120
    for name, text in (('data/final/done.jsonl', 'a\n'), ('runs/logs/job.done', ''), ('runs/logs/job.log', 'ok\n')):
        (remote / name).write_text(text)
        os.utime(remote / name, (old, old))
    (remote / 'data' / 'final' / 'writing.png').write_bytes(b'half')
    (remote / 'runs' / 'logs' / 'running.log').write_text('step 1\n')
    real = pipeline.rsync
    monkeypatch.setattr(pipeline, 'rsync', lambda src, dst, ssh, filters, dry=False: real(src.replace('root@h:', ''), dst.replace('root@h:', ''), ssh, filters, dry))
    monkeypatch.setattr(pipeline, 'fresh_remote', lambda host, ssh, root, seconds: pipeline.fresh_local(root, seconds))
    monkeypatch.setattr(pipeline, 'ROOT', local)
    args = type('A', (), {'peer': 'root@h:1', 'key': 'k', 'remote_root': str(remote), 'loop': 0, 'dry_run': False, 'fresh': 15})()
    pipeline.cmd_sync(args, pipeline.settings())
    assert (local / 'data/final/done.jsonl').read_text() == 'a\n' and (local / 'runs/logs/job.done').exists() and (local / 'runs/logs/job.log').exists()
    assert not (local / 'data/final/writing.png').exists() and not (local / 'runs/logs/running.log').exists()
    for name in ('data/final/writing.png', 'runs/logs/running.log'):
        os.utime(remote / name, (old, old))
    pipeline.cmd_sync(args, pipeline.settings())
    assert (local / 'data/final/writing.png').read_bytes() == b'half' and (local / 'runs/logs/running.log').read_text() == 'step 1\n'


def test_files_being_written_are_listed_relative_to_the_root(tmp_path):
    (tmp_path / 'data' / 'final').mkdir(parents=True)
    (tmp_path / 'runs' / 'logs').mkdir(parents=True)
    (tmp_path / 'data' / 'final' / 'new.png').write_bytes(b'x')
    (tmp_path / 'runs' / 'logs' / 'old.log').write_text('x')
    old = time.time() - 120
    os.utime(tmp_path / 'runs' / 'logs' / 'old.log', (old, old))
    assert pipeline.fresh_local(tmp_path, 15) == ['/data/final/new.png']
