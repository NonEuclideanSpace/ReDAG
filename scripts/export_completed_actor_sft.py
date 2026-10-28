import argparse
import gzip
import json
import os
from pathlib import Path
import tarfile
import time

from sample_delivery_compression import DigestSink, low_priority, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', default='runs/actor-full-200')
    parser.add_argument('--data', default='data/actor-full-sft')
    parser.add_argument('--output', default='delivery/priority-checkpoints/actor-full-step200-complete.tar.gz')
    parser.add_argument('--manifest', default='deployment/actor-full-step200-export.json')
    args = parser.parse_args()
    root = Path.cwd().resolve()
    run = Path(args.run).resolve()
    data = Path(args.data).resolve()
    output = Path(args.output).resolve()
    if not all(p.is_relative_to(root) for p in (run, data, output)):
        raise ValueError('Export paths must belong to this project')
    completed = json.loads((data / 'training_completed.json').read_text())
    rows = [json.loads(line) for line in (run / 'sft_actor/metrics.jsonl').read_text().splitlines()]
    target = completed['updates']
    if completed['status'] != 'completed' or [r['step'] for r in rows] != list(range(1, target + 1)):
        raise ValueError('Complete committed training metrics are required')
    checkpoint = Path(rows[-1]['checkpoint']).resolve()
    if checkpoint.parent != run / 'sft_actor/checkpoints':
        raise ValueError('Final checkpoint must belong to this training run')
    commit = json.loads((checkpoint / 'COMMITTED.json').read_text())
    if commit.get('status') != 'COMMITTED':
        raise ValueError('Final checkpoint is not committed')
    policy = Path(completed['policy']).resolve()
    if not policy.is_relative_to(run):
        raise ValueError('Final policy must belong to this training run')
    files = {p for p in policy.rglob('*') if p.is_file()}
    files.update(checkpoint / n for n in ('COMMITTED.json', 'actor.pt', 'training_state.pt'))
    files.update(p for p in data.rglob('*') if p.is_file())
    files.add(run / 'sft_actor/metrics.jsonl')
    files.update(p for p in (run / 'sft_actor').glob('*.json*') if p.is_file())
    files = sorted(files)
    before = {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in files}
    if output.exists():
        raise FileExistsError('Completed export already exists')
    output.parent.mkdir(parents=True, exist_ok=True)
    low_priority()
    started = time.time()
    temporary = output.with_suffix(output.suffix + '.part')
    sink = DigestSink(temporary)
    try:
        with gzip.GzipFile(filename='', mode='wb', fileobj=sink, compresslevel=1, mtime=0) as stream:
            with tarfile.open(fileobj=stream, mode='w|') as archive:
                for path in files:
                    archive.add(path, arcname=str(path.relative_to(root)), recursive=False)
        sink.finish()
        after = {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in files}
        if before != after:
            raise RuntimeError('Completed training files changed during export')
        temporary.replace(output)
    except BaseException:
        if not sink.stream.closed:
            sink.stream.close()
        raise
    write_json(Path(args.manifest), {
        'status': 'completed', 'source': str(output), 'bytes': sink.size,
        'mtime': int(output.stat().st_mtime), 'sha256': sink.digest.hexdigest(),
        'step': target, 'checkpoint': str(checkpoint), 'files': before,
        'includes_optimizer_scheduler_rng_and_sampler': True,
        'includes_final_policy': True, 'source_files_unchanged': True,
        'started_at': started, 'finished_at': time.time(),
    })


if __name__ == '__main__':
    main()
