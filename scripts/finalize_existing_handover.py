"""Finish the already authorized handover after download and disk copy finish.

No payload hashes are calculated. The receipt coverage check reads only local
JSON metadata, never archive payloads. Does not start experiments or stop rental.
"""
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
D = ROOT/'deployment'
META = D/'final-handover-metadata'
BACKUP = Path('/Volumes/TOSHIBA EXT/dag-pilot-rental-backup-20261027T130933Z-9769b4')
STATUS = D/'handover-finalization-status.json'

def read(path): return json.loads(path.read_text())

def main():
    state={'pid':os.getpid(),'started_at':time.time(),'status':'waiting_for_transfer',
           'content_hash_computed':False,'external_delivery_complete':False,'rental_shutdown':False}
    def save():
        state['updated_at']=time.time()
        tmp=STATUS.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2));tmp.replace(STATUS)
    save()
    while True:
        stage=read(D/'archive-prefetch-status.json');bundle=read(D/'archive-bundle-status.json')
        if bundle['status']=='failed':raise RuntimeError(bundle.get('error','External bundle failed'))
        if stage.get('catalog_caught_up') and not state.get('all_project_archives_local'):
            catalog=read(META/'final-project-catalog.json')
            baseline=read(D/'result-archive-backup-status.json')
            received={n:r['bytes'] for n,r in baseline['completed'].items()}
            received.update(stage['completed'])
            if any(received.get(x['archive'])!=x['archive_bytes'] for x in catalog['archives']):
                raise RuntimeError('Receipt coverage does not cover the sealed archive catalog')
            state['all_project_archives_local']=True
            state['all_project_archives_local_at']=state.get('all_project_archives_local_at',time.time())
            state['catalog_archive_bytes']=sum(x['archive_bytes'] for x in catalog['archives'])
            state['catalog_archive_count']=len(catalog['archives'])
            state['status']='waiting_for_external_disk'
        state['bundle_written_bytes']=bundle.get('written_bytes',0)
        save()
        if bundle['status']=='completed' and state.get('all_project_archives_local'):
            break
        time.sleep(10)
                                                                           
    if stage.get('pid') and stage['pid']!=os.getpid():
        cmd=subprocess.run(['ps','-p',str(stage['pid']),'-o','command='],capture_output=True,text=True).stdout
        if 'prefetch_rental_archives.py' in cmd:os.kill(stage['pid'],signal.SIGTERM)
    state['status']='preparing_final_supplement';save()
    files={Path(x['name']).relative_to('pilot-small') for x in read(ROOT/'outputs/final-current-code-archive-members.json')
           if (ROOT/Path(x['name']).relative_to('pilot-small')).is_file()}
    for folder in ['scripts','src','docs','tests','monitor','configs']:
        for p in (ROOT/folder).rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts and p.suffix in {'.py','.md','.txt','.html','.css','.js','.yaml','.yml','.sh'}:
                files.add(p.relative_to(ROOT))
    code=META/'pilot-small-code-20261028-final.tar.gz'
    with tarfile.open(code,'w:gz',compresslevel=1) as archive:
        for p in sorted(files):archive.add(ROOT/p,arcname='pilot-small/'+p.as_posix(),recursive=False)
    code_receipt={'archive':code.name,'bytes':code.stat().st_size,'files':len(files),'created_at':time.time(),
                  'content_hash_computed':False}
    (META/'final-code-receipt.json').write_text(json.dumps(code_receipt,indent=2))
    supplement=META/'final-handover-supplement.tar.gz'
    with tarfile.open(supplement,'w:gz',compresslevel=1) as archive:
        archive.add(code,arcname='code-revisions/'+code.name,recursive=False)
        archive.add(META/'final-project-metadata.tar.gz',arcname='final-project-metadata.tar.gz',recursive=False)
        archive.add(ROOT/'docs/RENTAL_RESTORE.md',arcname='RENTAL_RESTORE.md',recursive=False)
        for p in META.glob('*.json'):archive.add(p,arcname='reports/final-handover/'+p.name,recursive=False)
        for p in (D/'result-archive-transfers').glob('*.json'):
            archive.add(p,arcname='manifests/transfer-receipts/project-results/'+p.name,recursive=False)
        for p in (D/'archive-staging').glob('*.tar.gz.json'):
            archive.add(p,arcname='manifests/transfer-receipts/ssd-prefetch/'+p.name,recursive=False)
        for name in ['final-catalog-sealed.json','archive-bundle-status.json','archive-prefetch-status.json',
                     'transfer-disk-stall-incident.json','result-archive-backup-status.json']:
            archive.add(D/name,arcname='reports/final-handover/'+name,recursive=False)
    state.update(status='copying_final_supplement',supplement_bytes=supplement.stat().st_size);save()
    if not os.path.ismount('/Volumes/TOSHIBA EXT'):raise RuntimeError('External volume is not mounted')
    target=BACKUP/'final-handover-supplement.tar.gz'
    partial=target.with_suffix('.gz.part')
    shutil.copyfile(supplement,partial)
    if partial.stat().st_size!=supplement.stat().st_size:raise RuntimeError('Supplement copy was incomplete')
    partial.replace(target)
    index=read(META/'existing-DELIVERY_INDEX.json')
    index['status']='all_existing_frozen_artifacts_received'
    index['full_delivery_complete']=True
    index['completed_at']=time.time()
    index['delivery_scope']='All existing frozen rental artifacts; additional experiments remain deferred by user request'
    index['code'].update(archive='code-revisions/'+code.name,container='final-handover-supplement.tar.gz',
                         final_code_archive_update_required=False,content_hash_computed=False,bytes=code.stat().st_size)
    index['code'].pop('sha256',None)
    index['project'].update(catalog_snapshot_is_final=True,catalog_snapshot_at=catalog['updated_at'],
        received_archive_bytes=state['catalog_archive_bytes'],received_archive_count=state['catalog_archive_count'],
        standalone_archives=len(baseline['completed']),final_archives_bundle='remote/final-archives-bundle.tar',
        bundled_archives=len(bundle['members']),bundled_archive_bytes=bundle['archive_bytes'],
        final_metadata_container='final-handover-supplement.tar.gz -> final-project-metadata.tar.gz',
        receipt_snapshot={'final':True,'completed':state['catalog_archive_count'],
                          'content_hash_policy':'User disabled new content hashing; historical checked receipts retained'})
    index['known_incomplete']=['Expanded experiment budget was frozen before all planned evaluations were run',
        'A new machine independent offline restoration has not been run; see existing recovery evidence and historical missing-checkpoint report']
    index['transfer_policy']={'new_content_hashing':False,'checks':'authenticated SSH successful exit and exact length',
        'bundle_finalization':'closed tar stream and rename completed','final_supplement':'final-handover-supplement.tar.gz'}
    note=('所有已存在的冻结实验资料已回传。新资料按用户要求未做内容哈希校验。\n'
          '最后一批项目归档：remote/final-archives-bundle.tar\n'
          '最后源码、完整成员清单及传输收据：final-handover-supplement.tar.gz\n'
          '恢复前先解开补充包，再按其中的 RENTAL_RESTORE.md 展开最后归档包和完整成员清单。\n'
          '未继续新增实验；未自动关闭租用实例。\n')
    (BACKUP/'FINAL_HANDOVER_README.txt').write_text(note)
    tmp=BACKUP/'DELIVERY_INDEX.json.tmp';tmp.write_text(json.dumps(index,ensure_ascii=False,indent=2));tmp.replace(BACKUP/'DELIVERY_INDEX.json')
    (META/'completed-DELIVERY_INDEX.json').write_text(json.dumps(index,ensure_ascii=False,indent=2))
    state['status']='final_disk_flush';save()
    subprocess.run(['/bin/sync'],check=True)
    state.update(status='completed',external_delivery_complete=True,finished_at=time.time(),
                 final_disk_sync_returncode=0,rental_shutdown=False)
    save()

if __name__=='__main__':
    try:main()
    except BaseException as error:
        state=read(STATUS) if STATUS.exists() else {}
        state.update(status='failed',error=str(error),updated_at=time.time(),external_delivery_complete=False)
        STATUS.write_text(json.dumps(state,indent=2))
        raise
