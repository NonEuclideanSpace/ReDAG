"""Write cached remaining archives as one sequential external-disk tar file."""
import io
import json
import os
from pathlib import Path
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / 'deployment/archive-staging'
STATUS = ROOT / 'deployment/archive-bundle-status.json'
PREFETCH = ROOT / 'deployment/archive-prefetch-status.json'
BASE = ROOT / 'deployment/result-archive-backup-status.json'
DEST = Path('/Volumes/TOSHIBA EXT/dag-pilot-rental-backup-20261027T130933Z-9769b4/remote/final-archives-bundle.tar')

def main():
    state = {'pid':os.getpid(), 'started_at':time.time(), 'status':'waiting_for_existing_writers',
             'destination':str(DEST), 'written_bytes':0, 'archive_bytes':0, 'members':{},
             'content_hash_computed':False, 'final_disk_sync_pending':True}
    last_save = 0
    def save(force=False):
        nonlocal last_save
        if not force and time.monotonic()-last_save < 2:
            return
        state['updated_at'] = time.time()
        tmp = STATUS.with_suffix('.tmp')
        tmp.write_text(json.dumps(state,indent=2))
        tmp.replace(STATUS)
        last_save = time.monotonic()
    save(True)
    baseline = json.loads(BASE.read_text())
    waiting = list(baseline['active'])
    while waiting:
        for name in waiting[:]:
            receipt = json.loads((ROOT/'deployment/result-archive-transfers'/(name+'.json')).read_text())
            if receipt.get('status') == 'completed':
                baseline['completed'][name] = {'archive':name,'bytes':receipt['expected_bytes'],
                    'sha256':receipt['expected_sha256_from_manifest'], 'content_checksum_verified':False,
                    'verification':'authenticated_ssh_and_exact_length_only'}
                waiting.remove(name)
            elif receipt.get('status') == 'failed':
                raise RuntimeError('Existing external writer failed: '+name)
        state['waiting_for'] = waiting[:]
        save(True)
        if waiting:
            time.sleep(3)
    baseline['received_archive_bytes'] = sum(r['bytes'] for r in baseline['completed'].values())
    baseline['received_archives'] = len(baseline['completed'])
    baseline.update(active=[],status='remaining_archives_in_sequential_bundle',updated_at=time.time())
    tmp = BASE.with_suffix('.tmp');tmp.write_text(json.dumps(baseline,indent=2));tmp.replace(BASE)
    state['baseline_complete_bytes'] = baseline['received_archive_bytes']
    state['baseline_files'] = baseline['received_archives']
    if not os.path.ismount('/Volumes/TOSHIBA EXT'):
        raise RuntimeError('External volume is not mounted')
    partial = DEST.with_suffix('.tar.part')
    if partial.exists() or DEST.exists():
        raise RuntimeError('Preserving existing bundle; explicit resume required')

    class Output:
        def __init__(self, raw): self.raw = raw
        def write(self, data):
            n = self.raw.write(data)
            state['written_bytes'] += n
            save()
            return n
        def tell(self): return self.raw.tell()
        def flush(self): self.raw.flush()

    state['status'] = 'writing'
    with partial.open('wb',buffering=8*1024*1024) as raw:
        with tarfile.open(fileobj=Output(raw),mode='w|',bufsize=4*1024*1024) as bundle:
            while True:
                stage = json.loads(PREFETCH.read_text())
                ready = [n for n in stage['completed'] if n not in state['members']]
                for name in ready:
                    state['current_archive'] = name
                    save(True)
                    source = CACHE/name
                    receipt = json.loads((CACHE/(name+'.json')).read_text())
                    if receipt.get('status') != 'completed':
                        continue
                    bundle.add(source,arcname=name,recursive=False)
                    bundle.add(CACHE/(name+'.manifest.json'),arcname=name+'.manifest.json',recursive=False)
                    bundle.add(CACHE/(name+'.json'),arcname='transfer-receipts/'+name+'.json',recursive=False)
                    state['members'][name] = receipt['expected_bytes']
                    state['archive_bytes'] += receipt['expected_bytes']
                    save(True)
                if stage.get('catalog_caught_up') and set(stage['completed']) <= set(state['members']):
                                                                                               
                    seal = ROOT/'deployment/final-catalog-sealed.json'
                    if seal.exists():
                        sealed = json.loads(seal.read_text())
                        if sealed.get('catalog_archive_bytes') == stage.get('known_archive_bytes'):
                            metadata = json.dumps({'members':state['members'],'catalog':sealed,
                                'content_hash_computed':False},indent=2).encode()
                            info=tarfile.TarInfo('BUNDLE_INDEX.json');info.size=len(metadata);info.mtime=int(time.time())
                            bundle.addfile(info,io.BytesIO(metadata))
                            break
                state['status'] = 'waiting_for_download' if not ready else 'writing'
                save(True)
                time.sleep(2)
        state['status']='closing_external_bundle';save(True)
    partial.replace(DEST)
    state.update(status='completed',finished_at=time.time(),bundle_file_bytes=DEST.stat().st_size)
    save(True)

if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        if STATUS.exists():
            state=json.loads(STATUS.read_text());state.update(status='failed',error=str(error),updated_at=time.time())
            STATUS.write_text(json.dumps(state,indent=2))
        raise
