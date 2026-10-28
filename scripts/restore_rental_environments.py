import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import subprocess
import sys
import tarfile
import time


def inspect_archive(path, expected):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while data := stream.read(8 * 1024 * 1024):
            digest.update(data)
    if path.stat().st_size != expected['archive_bytes'] or digest.hexdigest() != expected['sha256']:
        raise ValueError('Archive identity mismatch: ' + str(path))
    roots = [PurePosixPath(value.lstrip('/')) for value in expected['roots']]
    with tarfile.open(path) as archive:
        for member in archive:
            name = PurePosixPath(member.name)
            if name.is_absolute() or '..' in name.parts or not any(name == root or root in name.parents for root in roots):
                raise ValueError('Archive member outside declared roots: ' + member.name)
            if member.isdev() or member.isfifo():
                raise ValueError('Unsupported device or FIFO in environment archive')
            if member.islnk():
                target = PurePosixPath(member.linkname)
                if target.is_absolute() or '..' in target.parts or not any(target == root or root in target.parents for root in roots):
                    raise ValueError('Hard link outside declared roots')
    return {'file': path.name, 'bytes': expected['archive_bytes'], 'sha256_verified': True, 'member_roots_verified': True}


def verify_runtime(manifest):
    result = {}
    for name, inventory in manifest['inventory'].items():
        executable = inventory['executable']
        imports = ['torch', 'transformers', 'diffusers', 'peft', 'bitsandbytes', 'causal_conv1d', 'fla'] if name == 'py312' else ['torch', 'torchvision', 'transformers', 'groundingdino', 'sam2']
        code = "import importlib,json,sys; modules=" + repr(imports) + "; loaded={name:str(getattr(importlib.import_module(name),'__version__','imported')) for name in modules}; import torch; print(json.dumps({'executable':sys.executable,'versions':loaded,'cuda_build':torch.version.cuda,'cuda_available':torch.cuda.is_available()}))"
        process = subprocess.run([executable, '-c', code], cwd=manifest['project'], text=True, capture_output=True, timeout=180, env={**os.environ, 'PYTHONPATH': str(Path(manifest['project'])/'src'), 'TOKENIZERS_PARALLELISM': 'false'})
        result[name] = {'returncode': process.returncode, 'stdout': process.stdout[-12000:], 'stderr': process.stderr[-12000:]}
    ocr = Path(manifest['project']) / '.runtime/ocr/bin/tesseract'
    if ocr.exists():
        process = subprocess.run([str(ocr), '--version'], text=True, capture_output=True, timeout=30)
        result['ocr'] = {'returncode': process.returncode, 'stdout': process.stdout, 'stderr': process.stderr}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    parser.add_argument('--restore', action='store_true')
    parser.add_argument('--verify-runtime', action='store_true')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    directory = args.directory.resolve()
    manifest = json.loads((directory/'manifest.json').read_text())
    if manifest['status'] != 'complete':
        raise ValueError('The environment package is incomplete')
    report = {'started_at': time.time(), 'archives': [], 'restored': False, 'runtime_verified': False}
    for name, expected in manifest['archives'].items():
        report['archives'].append(inspect_archive(directory/expected['file'], expected))
    report['unresolved_non_system_symlinks'] = manifest['unresolved_non_system_symlinks']
    if args.restore or args.verify_runtime:
        if platform.system() != 'Linux' or platform.machine() != manifest['machine']:
            raise RuntimeError('Restore requires Linux with the original CPU architecture')
        if manifest['unresolved_non_system_symlinks']:
            raise RuntimeError('Resolve declared external environment links before restoration')
    if args.restore:
        if os.geteuid() != 0:
            raise RuntimeError('Absolute-prefix restoration requires root on the destination Linux host')
        for expected in manifest['archives'].values():
            for root in expected['roots']:
                path = Path(root)
                if path.exists() or path.is_symlink():
                    raise FileExistsError('Restore only to an unused original prefix: ' + root)
        for expected in manifest['archives'].values():
            subprocess.run(['tar', '--extract', '--gzip', '--file', str(directory/expected['file']), '--directory', '/', '--same-permissions', '--no-same-owner', '--keep-old-files'], check=True)
        report['restored'] = True
    if args.verify_runtime:
        report['runtime'] = verify_runtime(manifest)
        report['runtime_verified'] = all(value['returncode'] == 0 for value in report['runtime'].values())
    report['completed_at'] = time.time()
    output = args.report or directory/'restore-verification.json'
    output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'report': str(output), 'archive_hashes_verified': True, 'restored': report['restored'], 'runtime_verified': report['runtime_verified']}))
    if args.verify_runtime and not report['runtime_verified']:
        sys.exit(1)


if __name__ == '__main__':
    main()
