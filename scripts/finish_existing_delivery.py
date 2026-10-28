import argparse
import fcntl
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import time

from archive_reclamation import manifest_digest, read_regular_json, validate_receipt


ROOT = Path(__file__).resolve().parents[1]
REMOTE = '/workspace/agentic-image-pilot-small'
DELIVERY = REMOTE + '/delivery/project-results'
BACKUP = Path('/Volumes/TOSHIBA EXT/dag-pilot-rental-backup-20261027T130933Z-9769b4')
SSH = ['ssh', '-S', '/tmp/dag-pilot-rental-direct1-ssh', '-o', 'ProxyCommand=false',
       '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8', '-p', '29410', 'root@117.50.181.105']
PHASES = [
    ('archive-full-rl-complete', 'runs/full-actor-rl', 'reclamation-full-rl.json'),
    ('archive-formal-agent_rl-complete', 'runs/development64-controller-episodes/agent_rl', 'reclamation-formal-agent_rl.json'),
    ('archive-formal-joint_rl-complete', 'runs/development64-controller-episodes/joint_rl', 'reclamation-formal-joint_rl.json'),
    ('archive-formal-base-complete', 'runs/development64-controller-episodes/base', 'reclamation-formal-base.json'),
]


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def remote(code, value=None):
    result = subprocess.run(SSH + ['/usr/local/miniconda3/envs/py312/bin/python -c ' + shlex.quote(code)],
        input=json.dumps(value).encode() if value is not None else None,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
    if result.returncode:
        raise RuntimeError(result.stderr.decode(errors='replace')[-1200:])
    return json.loads(result.stdout)


def observe_phase(job, group, gate):
    return remote('''import json,sys,pathlib,stat,time,hashlib
p=pathlib.Path('/workspace/agentic-image-pilot-small'); out=p/'delivery/project-results'
job,group,gate=json.load(sys.stdin)
g=p/'deployment'/gate
if g.exists():
 r=json.loads(g.read_text()); assert r['status']=='reclaimed'; print(json.dumps({'status':'gate_already_reclaimed','receipt':r}));raise SystemExit
s=p/'deployment'/(job+'.started.json');e=p/'deployment'/(job+'.exit.json')
if not e.exists():
 print(json.dumps({'status':'waiting_for_job_exit','started':s.exists()}));raise SystemExit
started=json.loads(s.read_text());ended=json.loads(e.read_text());assert ended['returncode']==0,ended
matches=[x for x in out.glob('*.pass.json') if started['started_at']-1 <= int(x.name.split('.')[0])/1e9 <= ended['finished_at']+1]
assert len(matches)==1, [x.name for x in matches]
path=matches[0];r=json.loads(path.read_text());generation=path.name.split('.')[0]
assert r['status']=='snapshot_catalog_caught_up' and r['only_groups']==[group]
assert not r['errors'] and not r['pending_changed_or_new_paths'] and r['active'] is None
assert not list(out.glob(generation+'-*.part'))
rows=[];member_count=0
for item in r['archives']:
 if not item['archive'].startswith(generation+'-'):continue
 name=item['archive'];proof=json.loads((out/(name[:-7]+'.manifest.json')).read_text())
 assert proof['group']==group and not proof['live_changes']
 assert all(m['stable'] and (m['path']==group or m['path'].startswith(group+'/')) for m in proof['members'])
 member_count+=len(proof['members'])
 digest=hashlib.sha256(json.dumps(proof,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode()).hexdigest()
 f=out/name
 if f.exists():
  st=f.lstat();assert stat.S_ISREG(st.st_mode) and st.st_size==proof['archive_bytes']
 else:
  reclaimed=json.loads((out/(name[:-7]+'.reclaimed.json')).read_text());assert reclaimed['state']=='remote_copy_reclaimed'
 rows.append({k:proof[k] for k in ['archive','archive_bytes','sha256']}|{'manifest_sha256':digest})
assert member_count==r['pending_members']
print(json.dumps({'status':'closed','observed_at':time.time(),'generation':generation,'job':job,'group':group,'exit':ended,'pass':str(path),'archive_count':len(rows),'member_count':member_count,'archives':rows,'source_bytes':r['source_bytes_read']}))
''', [job, group, gate])


def local_request(observation):
    assert os.path.ismount(BACKUP.parent)
    receipts, pending = [], []
    receipt_dir = BACKUP / 'manifests/transfer-receipts/project-results'
    receipt_dir.mkdir(parents=True, exist_ok=True)
    for item in observation['archives']:
        name = item['archive']
        report = ROOT / 'deployment/result-archive-transfers' / (name + '.json')
        if not report.exists():
            pending.append(name)
            continue
        transfer = read_regular_json(report)
        if transfer.get('status') != 'completed':
            pending.append(name)
            continue
        destination = BACKUP / 'remote/project-results-archives' / name
        proof = read_regular_json(destination.with_name(name[:-7] + '.manifest.json'))
        assert not destination.is_symlink() and destination.stat().st_size == item['archive_bytes']
        assert manifest_digest(proof) == item['manifest_sha256']
        assert transfer['source_checksum_verified'] is True and transfer['sha256'] == item['sha256']
        assert transfer['expected_bytes'] == transfer['assembled_bytes'] == transfer['received_bytes'] == item['archive_bytes']
        receipt = dict(item, source=transfer['source'], destination=transfer['destination'],
            external_verification={'status':'completed','method':'stream_sha256','bytes':transfer['expected_bytes'],
                'sha256':transfer['sha256'],'source':transfer['source'],'destination':transfer['destination'],
                'verified_at':transfer['finished_at']})
        validate_receipt(receipt, proof, DELIVERY)
        assert receipt['destination'] == str(destination)
        receipts.append(receipt)
        target = receipt_dir / report.name
        if not target.exists() or target.read_bytes() != report.read_bytes():
            shutil.copy2(report, target)
    return {'schema':'rental_archive_reclamation_requests_v1','delivery_directory':DELIVERY,'receipts':receipts}, pending


def reclaim_and_gate(request, gate):
    return remote('''import sys,json,pathlib,time,shutil,os
p=pathlib.Path('/workspace/agentic-image-pilot-small');sys.path.insert(0,str(p/'scripts'))
from archive_reclamation import reclaim
request,gate=json.load(sys.stdin);result=reclaim(request['delivery_directory'],request)
result.update(observed_at=time.time(),free_bytes_after=shutil.disk_usage(p).free)
if result['status']=='reclaimed':
 q=p/'deployment'/gate;t=q.with_suffix('.tmp');t.write_text(json.dumps(result,indent=2)+'\\n');os.replace(t,q)
print(json.dumps(result))
''', [request, gate])


def copy_pass(observation):
    target = BACKUP / 'manifests/project-results/passes' / Path(observation['pass']).name
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + '.tmp')
    with temporary.open('wb') as stream:
        subprocess.run(SSH + ['cat ' + shlex.quote(observation['pass'])], stdout=stream, check=True, timeout=60)
    data = json.loads(temporary.read_text())
    assert data['status'] == 'snapshot_catalog_caught_up' and not data['errors'] and not data['pending_changed_or_new_paths']
    temporary.replace(target)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hours', type=float, default=8)
    args = parser.parse_args()
    assert 0 < args.hours <= 8
    assert os.path.ismount(BACKUP.parent)
    with (ROOT/'deployment/finish-existing-delivery.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = {'pid':os.getpid(),'started_at':time.time(),'full_delivery_complete':False,'phases':{}}
        deadline = time.monotonic() + args.hours * 3600
        while time.monotonic() < deadline:
            try:
                all_done = True
                for job, group, gate in PHASES:
                    observed = observe_phase(job, group, gate)
                    state['current_job'] = job
                    if observed['status'] == 'gate_already_reclaimed':
                        state['phases'][job] = {'status':'reclaimed','receipt':gate}
                        continue
                    all_done = False
                    if observed['status'] != 'closed':
                        state['status'] = observed['status']
                        break
                    request, pending = local_request(observed)
                    state.update(status='waiting_for_verified_external_copies',generation=observed['generation'],
                        pending_archives=len(pending),verified_archives_in_phase=len(request['receipts']),
                        verified_bytes_in_phase=sum(r['archive_bytes'] for r in request['receipts']))
                    if pending:
                        break
                    copy_pass(observed)
                    request_path = ROOT/'deployment'/('reclamation-'+observed['generation']+'.json')
                    save(request_path, request)
                    result = reclaim_and_gate(request, gate)
                    state['phases'][job] = {'status':result['status'],'receipt':gate}
                    if result['status'] == 'reclaimed':
                        save(ROOT/'deployment'/gate, result)
                        save(BACKUP/'reports'/gate, result)
                    break
                if all_done:
                    state['status'] = 'archive_phase_gates_complete_final_static_seal_still_required'
                    state['finished_at'] = time.time()
                state['updated_at'] = time.time()
                state.pop('observation_error',None)
                save(ROOT/'deployment/finish-existing-delivery-status.json',state)
                if all_done:
                    return
            except Exception as error:
                state.update(status='observation_or_validation_error',updated_at=time.time(),
                    observation_error=type(error).__name__+': '+str(error))
                save(ROOT/'deployment/finish-existing-delivery-status.json',state)
            time.sleep(45)
        state.update(status='bounded_wait_expired',finished_at=time.time())
        save(ROOT/'deployment/finish-existing-delivery-status.json',state)


if __name__ == '__main__':
    main()
