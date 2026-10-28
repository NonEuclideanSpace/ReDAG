from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import plistlib
from pathlib import Path
import subprocess
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
HISTORY = deque(maxlen=600)
NETWORK_HISTORY = deque(maxlen=600)
DATA = {'status':'starting','history':[]}
DISK = {}
LOCK = threading.Lock()

def sample_disk():
    global DISK
    previous=None
    last_write=time.time()
    while True:
        try:
            rows=plistlib.loads(subprocess.check_output(['ioreg','-a','-r','-c','IOBlockStorageDriver','-l','-d','2'],timeout=4))
            for row in rows:
                if any(c.get('BSD Name')=='disk4' for c in row.get('IORegistryEntryChildren',[])):
                    stats=row['Statistics'];now=time.time();written=stats['Bytes (Write)']
                    speed=max(0,written-previous[1])/max(.1,now-previous[0]) if previous else None
                    if previous and written>previous[1]:last_write=now
                    DISK={'device':'disk4','sampled_at':now,'last_write_progress_at':last_write,
                          'write_speed':speed,'write_errors':stats.get('Errors (Write)',0)}
                    previous=(now,written)
        except Exception:
            pass
        time.sleep(2)


def sample():
    global DATA
    previous_total = 0
    last_progress_at = time.time()
    while True:
        try:
            state = json.loads((ROOT/'deployment/result-archive-backup-status.json').read_text())
            now = time.time()
            complete = state.get('received_archive_bytes',state.get('verified_bytes',0))
            active = []
            native = {}
            native_path = ROOT/'deployment/archive-direct-copy-status.json'
            if native_path.exists():
                native=json.loads(native_path.read_text())
            for name in state.get('active',[]):
                try:
                    row = json.loads((ROOT/'deployment/result-archive-transfers'/(name+'.json')).read_text())
                except (OSError,ValueError):
                    continue
                active.append({'name':name,'received':row.get('received_bytes',0),'total':row.get('expected_bytes',0),
                    'status':row.get('status'),'updated_at':row.get('updated_at',0)})
            if native:
                complete=native['baseline_complete_bytes']+native.get('copied_bytes',0)
                active=[]
                if native.get('current_file'):
                    active=[{'name':native['current_file'],'received':0,'total':native.get('current_file_bytes',0),
                             'status':native['status'],'updated_at':native.get('updated_at',0)}]
            total = max(previous_total,complete+sum(x['received'] for x in active))
            bundle = {}
            bundle_status = ROOT/'deployment/archive-bundle-status.json'
            if bundle_status.exists() and not native:
                bundle = json.loads(bundle_status.read_text())
                if bundle.get('baseline_complete_bytes'):
                    total = max(total,bundle['baseline_complete_bytes']+bundle.get('written_bytes',0))
            if total > previous_total:
                last_progress_at = now
            previous_total = total
            HISTORY.append((now,total))
            def rate(seconds):
                candidates = [x for x in HISTORY if x[0]>=now-seconds]
                first = candidates[0]
                return (total-first[1])/max(1,now-first[0])
            current,minute = rate(10),rate(60)
            live = False
            try:
                os.kill(state['pid'],0)
                live = now-state.get('updated_at',0)<35
            except (OSError,KeyError):
                pass
            estimate = 257342193128
            config = ROOT/'monitor/transfer-config.json'
            if config.exists():
                estimate = json.loads(config.read_text()).get('estimated_final_archive_bytes',estimate)
            remaining = max(0,estimate-total)
            progress_age = now-last_progress_at
            label = '正在写入外接盘' if live and active else '等待下一批打包' if live else '传输状态已过期'
            if active and progress_age >= 30:
                label = '外接盘写入停滞 · 正在处理'
            if bundle.get('status') in ['writing','waiting_for_download','closing_external_bundle','completed']:
                label = {'writing':'正在连续写入外接盘','waiting_for_download':'等待下载下一批',
                    'closing_external_bundle':'外接盘完成最后写入','completed':'归档包已写入，整理交付中'}[bundle['status']]
                live = now-bundle.get('updated_at',0)<35
                if bundle['status']=='writing' and progress_age>=30:
                    label='外接盘写入停滞 · 正在处理'
            if native:
                live=now-native.get('updated_at',0)<35
                label={'copying':'系统直接复制原文件','waiting_for_old_writer':'等待旧写入结束，随后直接复制',
                       'completed':'原始归档复制完成，正在收尾','failed':'复制遇到问题'}.get(native['status'],native['status'])
            prefetch = {}
            staging_status = ROOT/'deployment/archive-prefetch-status.json'
            if staging_status.exists():
                stage = json.loads(staging_status.read_text())
                network_total = stage.get('received_bytes',0)
                for name in stage.get('active',[]):
                    receipt = ROOT/'deployment/archive-staging'/(name+'.json')
                    if receipt.exists():
                        network_total += json.loads(receipt.read_text()).get('received_bytes',0)
                NETWORK_HISTORY.append((now,network_total))
                recent = [x for x in NETWORK_HISTORY if x[0]>=now-10]
                network_speed = (network_total-recent[0][1])/max(1,now-recent[0][0])
                recent_minute = [x for x in NETWORK_HISTORY if x[0]>=now-60]
                network_minute = (network_total-recent_minute[0][1])/max(1,now-recent_minute[0][0])
                prefetch = {'received':network_total,'target':stage.get('target_bytes',0),
                    'speed':network_speed,'minute_speed':network_minute,'active':stage.get('active',[]),
                    'catalog_caught_up':stage.get('catalog_caught_up',False),
                    'remaining':max(0,stage.get('target_bytes',0)-network_total)}
                if network_speed > 100000:
                    label = '正在下载到内置盘 · 外接盘' + ('写入停滞' if progress_age>=30 else '正在写入')
            finalization = {}
            finalization_status = ROOT/'deployment/handover-finalization-status.json'
            if finalization_status.exists():
                finalization = json.loads(finalization_status.read_text())
                if finalization.get('status') == 'completed' and finalization.get('external_delivery_complete'):
                    label = '全部交付完成 · 可以关闭租用实例'
                    remaining = 0
                    live = True
                elif finalization.get('all_project_archives_local'):
                    label = ('项目归档全部到本地 · 系统直接复制到外接盘' if native and native.get('status')=='copying'
                             else '项目归档全部到本地 · '+('外接盘写入停滞' if progress_age>=30 else '正在完成外接盘交付'))
                if finalization.get('status') == 'failed':
                    label = '交付收尾出现问题 · '+str(finalization.get('error',''))
            if native.get('status')=='manual_handoff':
                label='已交接手动复制 · 自动任务已停止'
                active=[]
                live=False
            value = {'sampled_at':now,'status':label,'live':live,'receiver_age':now-state.get('updated_at',0),
                'speed':current,'minute_speed':minute,'sample_seconds':now-HISTORY[0][0],
                'received':total,'complete':complete,'files':native['baseline_files']+len(native['completed']) if native else state.get('received_archives',state.get('verified_archives',0)),
                'remaining_estimate':remaining,'expected_estimate':estimate,
                'eta_seconds':remaining/minute if minute>100000 and now-HISTORY[0][0]>20 else None,
                'active':active,'failures':len(state.get('failures',[])),
                'stalled':bool(progress_age>=30 and (now-DISK.get('last_write_progress_at',now)>30 if native else (active or bundle.get('status')=='writing'))),
                'progress_age':progress_age,'prefetch':prefetch,'bundle':bundle,'native_copy':native,
                'disk':dict(DISK),'finalization':finalization,
                'hash_checking':False,'history':[{'t':t,'bytes':b} for t,b in HISTORY],
                'destination':'TOSHIBA EXT / dag-pilot-rental-backup-20261027T130933Z-9769b4'}
            with LOCK:
                DATA = value
        except Exception as error:
            with LOCK:
                DATA = dict(DATA,status='读取状态失败',live=False,error=str(error),sampled_at=time.time())
        time.sleep(2)


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.split('?')[0] == '/api/transfer':
            with LOCK:
                payload = json.dumps(DATA).encode()
            content_type = 'application/json; charset=utf-8'
        elif self.path.split('?')[0] in ['/','/transfer']:
            payload = (Path(__file__).with_name('transfer.html')).read_bytes()
            content_type = 'text/html; charset=utf-8'
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header('Content-Type',content_type)
        self.send_header('Cache-Control','no-store')
        self.send_header('Content-Length',str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self,*args):
        pass


threading.Thread(target=sample_disk,daemon=True).start()
threading.Thread(target=sample,daemon=True).start()
ThreadingHTTPServer(('127.0.0.1',8877),Handler).serve_forever()
