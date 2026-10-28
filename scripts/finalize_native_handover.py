"""Finish direct file delivery. All inputs are local; no content hash scans."""
import json
import os
from pathlib import Path
import subprocess
import tarfile
import time

ROOT=Path(__file__).resolve().parents[1]
D=ROOT/'deployment'
META=D/'final-handover-metadata'
BACKUP=Path('/Volumes/TOSHIBA EXT/dag-pilot-rental-backup-20261027T130933Z-9769b4')
STATUS=D/'handover-finalization-status.json'

def read(path):return json.loads(path.read_text())

def main():
    state={'pid':os.getpid(),'started_at':time.time(),'status':'waiting_for_external_disk',
           'method':'native_copy_existing_files','all_project_archives_local':True,
           'external_delivery_complete':False,'content_hash_computed':False,'rental_shutdown':False}
    def save():
        state['updated_at']=time.time()
        temp=STATUS.with_suffix('.tmp');temp.write_text(json.dumps(state,indent=2));temp.replace(STATUS)
    def copy(source,target):
        target.parent.mkdir(parents=True,exist_ok=True)
        partial=target.with_name(target.name+'.native-copy.part')
        state['current_file']=str(target.relative_to(BACKUP));save()
        subprocess.run(['/bin/cp','-X',str(source),str(partial)],check=True)
        if partial.stat().st_size!=source.stat().st_size:raise RuntimeError('Incomplete file copy: '+str(target))
        partial.replace(target)
    save()
    while True:
        transfer=read(D/'archive-direct-copy-status.json')
        if transfer['status']=='failed':raise RuntimeError(transfer.get('error','Native copy failed'))
        state['copied_bytes']=transfer['copied_bytes'];save()
        if transfer['status']=='completed':break
        time.sleep(10)
    catalog=read(META/'final-project-catalog.json')
    baseline=read(D/'result-archive-backup-status.json')
    received={name:row['bytes'] for name,row in baseline['completed'].items()}
    received.update(transfer['completed'])
    if any(received.get(x['archive'])!=x['archive_bytes'] for x in catalog['archives']):
        raise RuntimeError('Completed-copy receipts do not cover the final catalog')
    if not os.path.ismount('/Volumes/TOSHIBA EXT'):raise RuntimeError('External disk is not mounted')
    state['status']='copying_final_code_and_records';save()
                                                                                                 
    files={Path(x['name']).relative_to('pilot-small') for x in read(ROOT/'outputs/final-current-code-archive-members.json')
           if (ROOT/Path(x['name']).relative_to('pilot-small')).is_file()}
    for folder in ['scripts','src','docs','tests','monitor','configs']:
        for p in (ROOT/folder).rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts and p.suffix in {'.py','.md','.txt','.html','.css','.js','.yaml','.yml','.sh'}:
                files.add(p.relative_to(ROOT))
    code=META/'pilot-small-code-20261028-final.tar.gz'
    with tarfile.open(code,'w:gz',compresslevel=1) as archive:
        for p in sorted(files):archive.add(ROOT/p,arcname='pilot-small/'+p.as_posix(),recursive=False)
    copy(code,BACKUP/'code-revisions'/code.name)
    copy(META/'final-project-metadata.tar.gz',BACKUP/'manifests/project-results/final-project-metadata.tar.gz')
    copy(ROOT/'docs/RENTAL_RESTORE.md',BACKUP/'RENTAL_RESTORE.md')
    records={'created_at':time.time(),'final_catalog':catalog,'native_copy':transfer,
             'download_completion':read(D/'all-project-archives-local.json'),
             'final_metadata_receipt':read(META/'final-project-metadata-receipt.json'),
             'historical_project_transfer_receipts':{},'ssd_download_receipts':{},
             'new_content_hashing':False}
    for p in (D/'result-archive-transfers').glob('*.json'):
        records['historical_project_transfer_receipts'][p.name]=read(p)
    for p in (D/'archive-staging').glob('*.tar.gz.json'):
        records['ssd_download_receipts'][p.name]=read(p)
    record_file=META/'final-transfer-records.json'
    record_file.write_text(json.dumps(records,ensure_ascii=False,separators=(',',':')))
    copy(record_file,BACKUP/'reports/final-transfer-records.json')
    index=read(META/'existing-DELIVERY_INDEX.json')
    index.update(status='all_existing_frozen_artifacts_received',full_delivery_complete=True,completed_at=time.time(),
        delivery_scope='All existing frozen rental artifacts; additional experiments remain deferred by user request')
    index['code'].update(archive='code-revisions/'+code.name,final_code_archive_update_required=False,
                         content_hash_computed=False,bytes=code.stat().st_size)
    index['code'].pop('sha256',None)
    index['project'].update(catalog_snapshot_is_final=True,catalog_snapshot_at=catalog['updated_at'],
        received_archive_bytes=sum(x['archive_bytes'] for x in catalog['archives']),
        received_archive_count=len(catalog['archives']),standalone_archives=len(catalog['archives']),
        final_metadata_container='manifests/project-results/final-project-metadata.tar.gz',
        final_transfer_receipts='reports/final-transfer-records.json',
        receipt_snapshot={'final':True,'completed':len(catalog['archives']),
                          'content_hash_policy':'New content hashing disabled by user; historical checked receipts retained'})
    index['known_incomplete']=['Expanded experiment budget was frozen before all planned evaluations were run',
        'A new machine independent offline restoration has not been run; see recovery evidence and historical missing-checkpoint report']
    index['transfer_policy']={'new_content_hashing':False,'copy':'native cp of unchanged original archive files',
        'completion_evidence':'successful system-copy exits and exact lengths',
        'temporary_bundle':'remote/final-archives-bundle.tar.part is an abandoned partial; not a delivery input'}
    note=('现有冻结实验资料已回传。最后 152 个原始归档通过系统直接复制，没有重新压缩或打大包。\n'
          '项目归档：remote/project-results-archives。最新代码：'+index['code']['archive']+'。\n'
          '恢复前，在本备份根目录运行：tar -xzf manifests/project-results/final-project-metadata.tar.gz\n'
          '然后按 RENTAL_RESTORE.md 恢复。新增传输未做内容哈希扫描；历史校验收据保留。\n'
          'remote/final-archives-bundle.tar.part 为已放弃的大包临时文件，不参与恢复。\n'
          '未自动关闭租用实例。\n')
    local_note=META/'FINAL_HANDOVER_README.txt';local_note.write_text(note)
    copy(local_note,BACKUP/'FINAL_HANDOVER_README.txt')
    local_index=META/'completed-DELIVERY_INDEX.json';local_index.write_text(json.dumps(index,ensure_ascii=False,indent=2))
    copy(local_index,BACKUP/'DELIVERY_INDEX.json')
    state['status']='final_disk_flush';save()
    subprocess.run(['/bin/sync'],check=True)
    state.update(status='completed',external_delivery_complete=True,finished_at=time.time(),final_disk_sync_returncode=0)
    save()

if __name__=='__main__':
    try:main()
    except BaseException as error:
        state=read(STATUS) if STATUS.exists() else {}
        state.update(status='failed',error=str(error),updated_at=time.time(),external_delivery_complete=False)
        STATUS.write_text(json.dumps(state,indent=2))
        raise
