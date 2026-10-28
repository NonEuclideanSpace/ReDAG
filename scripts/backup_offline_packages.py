import argparse
from concurrent.futures import ThreadPoolExecutor
from email.parser import Parser
import fcntl
import hashlib
from html.parser import HTMLParser
import itertools
import json
import os
from pathlib import Path
import re
import threading
import time
from urllib.parse import quote, unquote, urljoin, urlsplit
import urllib.request
import zipfile


def write_json(path, value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n')
    temporary.replace(path)


def normalize(name):
    return re.sub(r'[-_.]+','-',name).lower()


def tags(value):
    fields=value.split('-')
    return {'-'.join(parts) for parts in itertools.product(*(field.split('.') for field in fields))} if len(fields)==3 else set()


def installed_tags(package):
    return set().union(*(tags(line[5:].strip()) for line in (package.get('wheel') or '').splitlines() if line.startswith('Tag: ')))


def wheel_matches(filename, package):
    if not filename.endswith('.whl'):
        return False
    fields=filename[:-4].rsplit('-',3)
    return len(fields)==4 and bool(tags('-'.join(fields[-3:])) & installed_tags(package))


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links=[]
    def handle_starttag(self, tag, attrs):
        if tag=='a':
            self.links.extend(value for key,value in attrs if key=='href')


class RateLimit:
    def __init__(self, rate):
        self.rate=rate
        self.next=time.monotonic()
        self.lock=threading.Lock()
    def consume(self, size):
        with self.lock:
            now=time.monotonic()
            self.next=max(now,self.next)+size/self.rate
            delay=self.next-now
        time.sleep(delay)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--origins',type=Path,required=True)
    parser.add_argument('--destination',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--mib-per-second',type=float,default=0.5)
    args=parser.parse_args()
    if not 1<=args.workers<=4 or args.mib_per_second<=0:
        raise ValueError('Use one to four workers and a positive transfer rate')
    destination=args.destination.resolve()
    volume=Path('/Volumes/TOSHIBA EXT')
    if not os.path.ismount(volume) or volume not in destination.parents:
        raise RuntimeError('The expected external volume is not mounted or selected')
    device=volume.stat().st_dev
    destination.mkdir(parents=True,exist_ok=True)
    lockfile=(destination/'.download.lock').open('w')
    fcntl.flock(lockfile,fcntl.LOCK_EX|fcntl.LOCK_NB)
    origins=json.loads(args.origins.read_text())
    (destination/'installed-package-origins.json').write_bytes(args.origins.read_bytes())
    lock=threading.Lock()
    rate=RateLimit(args.mib_per_second*1024*1024)
    started=time.time()
    jobs=[]
    for environment, data in origins.items():
        conda_names={normalize(package['name']):package for package in data['conda_packages']}
        for package in data['packages']:
            direct=package.get('direct_url') or {}
            name=normalize(package['name'])
            if name in conda_names and conda_names[name]['version']==package['version']:
                continue
            jobs.append({'environment':environment,'kind':'wheel','name':package['name'],'version':package['version'],'package':package,'status':'pending'})
        for package in data['conda_packages']:
            jobs.append({'environment':environment,'kind':'conda','name':package['name'],'version':package['version'],'package':package,'status':'pending'})
    jobs.sort(key=lambda row:(0 if (row['package'].get('direct_url') or {}).get('url','').startswith('https:') else 1 if row['kind']=='conda' else 2,row['environment'],row['name']))
    files={}
    history=[]
    stop=threading.Event()
    def update(row,**values):
        with lock:
            row.update(values)
    def publish():
        with lock:
            rows=[{key:value for key,value in row.items() if key!='package'} for row in jobs]
            states=[dict(value) for value in files.values()]
        now=time.time()
        total=sum(row.get('bytes',0) for row in states)
        history.append((now,total))
        while len(history)>2 and history[0][0]<now-60: history.pop(0)
        state={'status':'running' if not stop.is_set() else 'completed_with_gaps' if any(row['status'] in ('unavailable','archive_required','failed') for row in rows) else 'sources_downloaded','pid':os.getpid(),'started_at':started,'updated_at':now,'workers':args.workers,'rate_limit_bytes_per_second':rate.rate,'bytes':total,'expected_known_bytes':sum(row.get('expected_bytes') or 0 for row in states),'recent_bytes_per_second':(total-history[0][1])/max(.001,now-history[0][0]),'files_verified':sum(row['status']=='verified' for row in states),'files_total':len(states),'rows':rows,'files':states,'complete_offline_restore_verified':False,'environment_archive_delivery':False,'original_installed_index_artifact_identity_claimed':False}
        write_json(destination/'status.json',state)
        write_json(Path(__file__).resolve().parents[1]/'deployment/offline-package-backup-status.json',state)
    def monitor():
        while not stop.wait(5): publish()
    def request(url,method='GET'):
        return urllib.request.urlopen(urllib.request.Request(url,method=method,headers={'User-Agent':'Exact-Linux-environment-recovery/1.0'}),timeout=20)
    def resolve(row):
        package=row['package']
        if row['kind']=='conda':
            if package.get('subdir') not in {'linux-64','noarch'}:
                raise ValueError('Unexpected conda architecture')
            algorithm='sha256' if package.get('sha256') else 'md5'
            if not package.get(algorithm):raise ValueError('Missing native conda hash')
            return {'filename':package['fn'],'url':package['url'],'hash_algorithm':algorithm,'native_hash':package[algorithm],'expected_bytes':None,'identity_source':'installed_conda_record','kind':'conda'}
        direct=package.get('direct_url') or {}
        url=direct.get('url')
        if url:
            if not url.startswith('https://'):
                update(row,status='archive_required',reason='Installed local source or editable build requires its actual archived implementation')
                return None
            hashes=direct.get('archive_info',{}).get('hashes',{})
            if not hashes.get('sha256'):
                value=direct.get('archive_info',{}).get('hash','')
                if value.startswith('sha256='):hashes['sha256']=value.split('=',1)[1]
            filename=unquote(urlsplit(url).path.rsplit('/',1)[-1])
            if not hashes.get('sha256') or not wheel_matches(filename,package):
                raise ValueError('Direct URL lacks a native SHA256 or exact installed Linux wheel tags')
            return {'filename':filename,'url':url,'hash_algorithm':'sha256','native_hash':hashes['sha256'],'expected_bytes':None,'identity_source':'installed_direct_url','kind':'wheel'}
        if package['name'] in ('torch','torchvision') and package['version'].endswith('+cu132'):
            index='https://download.pytorch.org/whl/cu132/'+package['name']+'/'
            with request(index) as response: contents=response.read().decode()
            links=Links();links.feed(contents)
            matches=[]
            for link in links.links:
                target=urljoin(index,link);parts=urlsplit(target)
                filename=unquote(parts.path.rsplit('/',1)[-1])
                expected_prefix=package['name'].replace('-','_')+'-'+package['version']+'-'
                if filename.startswith(expected_prefix) and wheel_matches(filename,package) and parts.fragment.startswith('sha256='):
                    matches.append({'filename':filename,'url':target.split('#',1)[0],'hash_algorithm':'sha256','native_hash':parts.fragment.split('=',1)[1],'expected_bytes':None,'identity_source':'official_torch_exact_version_and_installed_tags','kind':'wheel'})
            if len(matches)!=1:raise ValueError('Official CUDA wheel exact match missing or ambiguous')
            return matches[0]
        with request('https://pypi.org/pypi/'+quote(package['name'],safe='')+'/'+quote(package['version'],safe='')+'/json') as response:
            metadata=json.load(response)
        matches=[item for item in metadata['urls'] if item['packagetype']=='bdist_wheel' and wheel_matches(item['filename'],package)]
        if len(matches)!=1:raise ValueError('Exact installed wheel tag match missing or ambiguous: '+str(len(matches)))
        item=matches[0]
        return {'filename':item['filename'],'url':item['url'],'hash_algorithm':'sha256','native_hash':item['digests']['sha256'],'expected_bytes':item['size'],'identity_source':'pypi_exact_version_and_installed_wheel_tags','kind':'wheel'}
    def download(row,entry):
        key=entry['kind']+'/'+entry['filename']
        with lock:
            if key in files:
                if files[key]['native_hash']!=entry['native_hash']:raise ValueError('Conflicting native package hashes')
                row.update(status='shared_source',file=key)
                return
            item={**entry,'path':key,'bytes':0,'status':'connecting'}
            files[key]=item
            row.update(status='downloading',file=key)
        target=destination/key
        target.parent.mkdir(parents=True,exist_ok=True)
        partial=target.with_name(target.name+'.part')
        proof=target.with_name(target.name+'.verified.json')
        if target.exists() and proof.exists():
            previous=json.loads(proof.read_text())
            if previous['native_hash']==entry['native_hash'] and previous['bytes']==target.stat().st_size:
                update(item,status='verified',bytes=target.stat().st_size,expected_bytes=target.stat().st_size)
                update(row,status='verified')
                return
        for attempt in range(2):
            try:
                if not os.path.ismount(volume) or volume.stat().st_dev!=device:raise RuntimeError('External disk identity changed')
                offset=partial.stat().st_size if partial.exists() else 0
                digest=hashlib.new(entry['hash_algorithm'])
                if offset:
                    with partial.open('rb') as stream:
                        while data:=stream.read(1024*1024):digest.update(data)
                headers={'Range':f'bytes={offset}-'} if offset else {}
                with urllib.request.urlopen(urllib.request.Request(entry['url'],headers=headers),timeout=25) as response:
                    if offset and (response.status!=206 or not response.headers.get('Content-Range','').startswith(f'bytes {offset}-')):raise RuntimeError('Server refused exact partial resume')
                    length=response.headers.get('Content-Length')
                    expected=offset+int(length) if length else entry['expected_bytes']
                    if entry['expected_bytes'] and expected and entry['expected_bytes']!=expected:raise ValueError('Expected package length changed')
                    update(item,status='downloading',expected_bytes=expected,bytes=offset,attempt=attempt+1)
                    with partial.open('ab' if offset else 'wb') as stream:
                        while data:=response.read(65536):
                            rate.consume(len(data));stream.write(data);digest.update(data);offset+=len(data)
                            update(item,bytes=offset,last_byte_at=time.time())
                if expected and offset!=expected:raise ValueError('Incomplete package response')
                if digest.hexdigest()!=entry['native_hash']:raise ValueError('Native package hash mismatch')
                if entry['kind']=='wheel':
                    with zipfile.ZipFile(partial) as wheel:
                        names=[name for name in wheel.namelist() if name.endswith('.dist-info/METADATA')]
                        if len(names)!=1:raise ValueError('Wheel metadata is ambiguous')
                        metadata=Parser().parsestr(wheel.read(names[0]).decode())
                        if normalize(metadata['Name'])!=normalize(row['name']) or metadata['Version']!=row['version']:raise ValueError('Wheel name/version differs from installed identity')
                partial.replace(target)
                write_json(proof,{**entry,'bytes':offset,'verified_at':time.time(),'verification':'native_hash_streamed_once_during_download','installed_metadata_version_verified':entry['kind']=='wheel'})
                update(item,status='verified',bytes=offset,expected_bytes=offset)
                update(row,status='verified')
                return
            except Exception as error:
                update(item,status='failed',error=f'{type(error).__name__}: {error}')
                if isinstance(error,ValueError):break
        update(row,status='failed',error=item.get('error'))
    def work(row):
        update(row,status='resolving')
        try:
            entry=resolve(row)
            if entry:download(row,entry)
        except Exception as error:update(row,status='unavailable',reason=f'{type(error).__name__}: {error}')
    publish()
    monitor_thread=threading.Thread(target=monitor,daemon=True);monitor_thread.start()
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            list(executor.map(work,jobs))
    finally:
        stop.set();monitor_thread.join();publish()


if __name__=='__main__':main()
