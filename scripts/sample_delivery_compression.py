import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def low_priority():
    os.nice(19)
    if Path('/usr/bin/ionice').exists():
        subprocess.run(['/usr/bin/ionice', '-c', '3', '-p', str(os.getpid())], check=True)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


class DigestSink:
    def __init__(self, path):
        self.stream = path.open('wb')
        self.digest = hashlib.sha256()
        self.size = 0

    def write(self, data):
        self.digest.update(data)
        self.size += len(data)
        return self.stream.write(data)

    def flush(self):
        self.stream.flush()

    def finish(self):
        self.stream.flush()
        os.fsync(self.stream.fileno())
        self.stream.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, default=Path('/workspace/agentic-image-pilot-small'))
    parser.add_argument('--source', type=Path)
    parser.add_argument('--pattern', default='runs/renderer-expanded-200/bundles/*.pt')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--read-mib-per-second', type=float, default=16)
    args = parser.parse_args()
    if args.read_mib_per_second <= 0:
        raise ValueError('Read rate must be positive')
    project = args.project.resolve()
    source = args.source
    if source is None:
        candidates = sorted(path for path in project.glob(args.pattern) if path.is_file())
        if not candidates:
            raise FileNotFoundError('No source matched the exact bundle pattern')
        source = next((p for p in candidates if 16 * 2**20 <= p.stat().st_size <= 32 * 2**20), candidates[0])
    elif not source.is_absolute():
        source = project / source
    source = source.resolve(strict=True)
    if not source.is_relative_to(project) or source.is_relative_to(project / 'models'):
        raise ValueError('Sample must be a project result, not a base model')
    output = args.output or project / 'delivery/results-compression-sample'
    output.mkdir(parents=True, exist_ok=True)
    target = output / (source.name + '.gz')
    if target.exists():
        raise FileExistsError('Sample output already exists; select a new output directory')
    low_priority()
    before = source.stat()
    start = time.monotonic()
    cpu = time.process_time()
    raw_hash = hashlib.sha256()
    count = 0
    sink = DigestSink(target.with_suffix(target.suffix + '.part'))
    try:
        with source.open('rb') as stream, gzip.GzipFile(filename='', mode='wb', fileobj=sink, compresslevel=1, mtime=0) as compressed:
            while chunk := stream.read(1024 * 1024):
                count += len(chunk)
                raw_hash.update(chunk)
                compressed.write(chunk)
                time.sleep(max(0, count / (args.read_mib_per_second * 2**20) - (time.monotonic() - start)))
        sink.finish()
    except BaseException:
        sink.stream.close()
        raise
    after = source.stat()
    sink.stream.name and Path(sink.stream.name).replace(target)
    stable = (before.st_size, before.st_mtime_ns, before.st_ino) == (after.st_size, after.st_mtime_ns, after.st_ino)
    report = {'source': str(source), 'source_bytes': before.st_size, 'read_bytes': count,
        'source_mtime_ns_before': before.st_mtime_ns, 'source_mtime_ns_after': after.st_mtime_ns,
        'source_stable': stable, 'source_stream_sha256': raw_hash.hexdigest(),
        'archive': str(target), 'archive_bytes': sink.size, 'archive_sha256': sink.digest.hexdigest(),
        'compression': 'gzip_level_1_single_thread', 'ratio': sink.size / max(1, count),
        'saved_bytes': count - sink.size, 'wall_seconds': time.monotonic() - start,
        'cpu_seconds': time.process_time() - cpu, 'read_mib_per_second_limit': args.read_mib_per_second,
        'source_deleted': False, 'full_return_completed': False}
    write_json(output / 'sample-report.json', report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
