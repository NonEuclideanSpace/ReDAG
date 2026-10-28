import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--bytes', type=int, required=True)
    parser.add_argument('--mtime', type=int, required=True)
    parser.add_argument('--sha256', required=True)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--control', action='append', required=True)
    parser.add_argument('--status', type=Path, required=True)
    parser.add_argument('--staging', action='store_true')
    args = parser.parse_args()
    volume = Path('/Volumes/TOSHIBA EXT')
    cache_root = Path(__file__).resolve().parents[1] / 'deployment/archive-staging'
    staging_ok = args.staging and args.destination.parent == cache_root
    if not staging_ok and (not os.path.ismount(volume) or not args.destination.is_relative_to(volume)):
        raise RuntimeError('Requested external disk is not mounted')
    destination = args.destination
    partial = destination.with_suffix(destination.suffix + '.direct.part')
    assembled = destination.with_suffix(destination.suffix + '.assembled.tmp')
    parts = destination.parent / (destination.name + '.parts')
    if not partial.exists() and assembled.exists():
        assembled.replace(partial)
    if not partial.exists() and parts.exists():
        with partial.open('wb') as output:
            index = 0
            while True:
                chunk = parts / f'{index:06d}.part'
                expected = min(4194304, args.bytes - index * 4194304)
                if expected <= 0 or not chunk.is_file() or chunk.stat().st_size != expected:
                    break
                with chunk.open('rb') as source:
                    while data := source.read(1048576):
                        output.write(data)
                index += 1
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > args.bytes:
        raise RuntimeError('Existing partial is larger than the expected source')
    state = {'status':'running','pid':os.getpid(),'started_at':time.time(),
        'source':args.source,'destination':str(destination),'expected_bytes':args.bytes,
        'initial_bytes':offset,'received_bytes':offset,'assembled_bytes':offset,
        'transport_mode':'direct_no_content_hash','source_checksum_verified':False,
        'expected_sha256_from_manifest':args.sha256,'verification':'authenticated_ssh_and_exact_length_only',
        'content_hash_computed':False,'per_file_fsync':False,'final_disk_sync_pending':True,'chunk_failures':[]}

    def save():
        state['updated_at'] = time.time()
        state['average_bytes_per_second'] = (state['received_bytes']-offset)/max(1,time.time()-state['started_at'])
        temporary = args.status.with_suffix('.tmp')
        temporary.write_text(json.dumps(state,indent=2)+'\n')
        temporary.replace(args.status)

    save()
    ssh = ['ssh','-S',args.control[0],'-o','ControlMaster=no','-o','ProxyCommand=false',
        '-o','BatchMode=yes','-o','ConnectTimeout=8','-o','ServerAliveInterval=15',
        '-o','ServerAliveCountMax=3','-p','29410','root@117.50.181.105']
    command = 'dd if='+shlex.quote(args.source)+' iflag=skip_bytes,count_bytes skip='+str(offset)+' count='+str(args.bytes-offset)+' bs=1048576 status=none'
    child = None
    try:
        cached = cache_root / destination.name
        cache_receipt = cache_root / (destination.name + '.json')
        cached_ready = False
        if not args.staging and cache_receipt.exists():
            receipt = json.loads(cache_receipt.read_text())
            cached_ready = (receipt.get('status') == 'completed' and receipt.get('received_bytes') == args.bytes
                and receipt.get('expected_sha256_from_manifest') == args.sha256
                and cached.is_file() and cached.stat().st_size == args.bytes)
        if offset < args.bytes and cached_ready:
            state['input_mode'] = 'local_ssd_staging'
            with cached.open('rb') as source, partial.open('ab') as output:
                source.seek(offset)
                last_save = time.monotonic()
                while data := source.read(1048576):
                    output.write(data)
                    state['received_bytes'] += len(data)
                    state['assembled_bytes'] = state['received_bytes']
                    if time.monotonic()-last_save >= 3:
                        save()
                        last_save = time.monotonic()
            state['transfer_returncode'] = receipt['transfer_returncode']
            state['source_receipt'] = str(cache_receipt)
        elif offset < args.bytes:
            with destination.with_suffix(destination.suffix+'.direct.log').open('ab') as errors:
                child = subprocess.Popen(ssh+[command],stdout=subprocess.PIPE,stderr=errors)
                with partial.open('ab') as output:
                    last_save = time.monotonic()
                    while data := child.stdout.read(1048576):
                        if state['received_bytes']+len(data)>args.bytes:
                            raise RuntimeError('Source sent more bytes than the frozen archive')
                        output.write(data)
                        state['received_bytes'] += len(data)
                        state['assembled_bytes'] = state['received_bytes']
                        if time.monotonic()-last_save>=3:
                            output.flush()
                            save()
                            last_save = time.monotonic()
                    output.flush()
                returncode = child.wait(timeout=30)
                state['transfer_returncode'] = returncode
                if returncode:
                    raise RuntimeError('SSH transfer returned '+str(returncode))
        else:
            state['transfer_returncode'] = 0
            state['reused_full_length_partial'] = True
        if partial.stat().st_size != args.bytes:
            raise RuntimeError('Transfer ended before the expected file length')
        os.utime(partial,(args.mtime,args.mtime))
        partial.replace(destination)
        state.update(status='completed',finished_at=time.time(),received_bytes=args.bytes,assembled_bytes=args.bytes)
        save()
    except BaseException as error:
        if child is not None and child.poll() is None:
            child.terminate()
        state.update(status='failed',error=type(error).__name__+': '+str(error))
        save()
        raise


if __name__ == '__main__':
    main()
