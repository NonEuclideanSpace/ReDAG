import json
import os
import time

_last = [0.0]


def report(done, total=None, unit='items', force=False, **extra):
    path = os.environ.get('REDAG_PROGRESS_FILE')
    if not path:
        return
    now = time.time()
    if not force and now - _last[0] < 5 and (total is None or done < total):
        return
    _last[0] = now
    row = dict(done=done, total=total, unit=unit, time=now, **extra)
    tmp = f'{path}.{os.getpid()}.tmp'
    try:
        with open(tmp, 'w', encoding='utf-8') as stream:
            json.dump(row, stream)
        os.replace(tmp, path)
    except OSError:
        pass
