"""Copy existing archive files with macOS cp, without repacking or hashing."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
D=ROOT/'deployment'
CACHE=D/'archive-staging'
DEST=Path('/Volumes/TOSHIBA EXT/dag-pilot-rental-backup-20261027T130933Z-9769b4/remote/project-results-archives')
STATUS=D/'archive-direct-copy-status.json'

def main():
    stage=json.loads((D/'archive-prefetch-status.json').read_text())
    baseline=json.loads((D/'result-archive-backup-status.json').read_text())
    if stage.get('status')!='completed' or stage.get('active'):
        raise RuntimeError('The local download must be complete')
    previous=json.loads(STATUS.read_text()) if STATUS.exists() else {}
    state={'pid':os.getpid(),'started_at':time.time(),'status':'waiting_for_old_writer',
        'method':'macOS /bin/cp -X; original archive bytes; no repacking or content hashes',
        'target_bytes':sum(stage['completed'].values()),'target_files':len(stage['completed']),
        'completed':previous.get('completed',{}),'current_file':None,'child_pid':None,
        'baseline_complete_bytes':baseline['received_archive_bytes'],
        'baseline_files':baseline['received_archives'],'content_hash_computed':False,
        'per_file_fsync':False,'source_files_retained':True}
    def save():
        state['updated_at']=time.time()
        state['copied_bytes']=sum(state['completed'].values())
        temp=STATUS.with_suffix('.tmp');temp.write_text(json.dumps(state,indent=2));temp.replace(STATUS)
    save()
                                                                                 
    old=json.loads((D/'archive-bundle-status.json').read_text())
    while True:
        result=subprocess.run(['ps','-p',str(old['pid']),'-o','command='],capture_output=True,text=True)
        if 'scripts/bundle_prefetched_archives.py' not in result.stdout:break
        state['waiting_for_pid']=old['pid'];save();time.sleep(3)
    state.pop('waiting_for_pid',None)
    if not os.path.ismount('/Volumes/TOSHIBA EXT'):
        raise RuntimeError('External volume is not mounted')
    state['status']='copying';save()
    with (D/'archive-direct-copy.log').open('ab') as log:
        for name,size in stage['completed'].items():
            if state['completed'].get(name)==size:continue
            source=CACHE/name
            target=DEST/name
            partial=target.with_suffix(target.suffix+'.native-copy.part')
            state.update(status='copying',current_file=name,current_file_bytes=size,current_file_started_at=time.time())
            save()
            child=subprocess.Popen(['/bin/cp','-X',str(source),str(partial)],stdout=log,stderr=log)
            state['child_pid']=child.pid;save()
            while child.poll() is None:
                                                                                     
                save();time.sleep(2)
            if child.returncode:
                raise RuntimeError('System copy failed for '+name+' (exit '+str(child.returncode)+')')
            if partial.stat().st_size!=size:
                raise RuntimeError('System copy ended with an incomplete destination length: '+name)
            partial.replace(target)
            state['completed'][name]=size
            state['child_pid']=None
            save()
    state.update(status='completed',finished_at=time.time(),current_file=None)
    save()

if __name__=='__main__':
    try:main()
    except BaseException as error:
        state=json.loads(STATUS.read_text()) if STATUS.exists() else {}
        state.update(status='failed',error=str(error),updated_at=time.time())
        STATUS.write_text(json.dumps(state,indent=2))
        raise
