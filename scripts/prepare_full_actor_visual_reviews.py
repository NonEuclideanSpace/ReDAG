import argparse
import json
from pathlib import Path
import re
import tarfile
import textwrap

from PIL import Image, ImageDraw

from agentic_image.core.artifacts import ArtifactStore


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', default='runs/full-actor-demonstrations')
    parser.add_argument('--batch', required=True)
    parser.add_argument('--limit', type=int, default=12)
    parser.add_argument('--state', default='deployment/full-actor-visual-packed.json')
    parser.add_argument('--seed-manifest', default='outputs/training-repair-inspection-v1/manifest.json')
    args = parser.parse_args()
    if not re.fullmatch(r'[a-zA-Z0-9_-]+', args.batch) or not 1 <= args.limit <= 32:
        raise ValueError('A bounded batch and simple name are required')
    state_path = Path(args.state)
    packed = json.loads(state_path.read_text()) if state_path.exists() else {}
    if Path(args.seed_manifest).exists():
        for row in json.loads(Path(args.seed_manifest).read_text()):
            previous = packed.get(row['task_id'])
            if previous is not None and previous != row['images']:
                raise ValueError('A seeded review changed its actual images')
            packed[row['task_id']] = row['images']
    selected = []
    for bundle_path in sorted(Path(args.run).glob('shard-*/tasks/*/state_bundle.json')):
        folder = bundle_path.parent
        completed = folder / 'completed.json'
        if not completed.exists():
            continue
        done = json.loads(completed.read_text())
        if done['counts']['failed_tasks']:
            continue
        records = read_rows(folder / 'actor_records.jsonl')
        if not any(row['phase'] == 1 or row['verifier_provenance'].get('visual_success_label') is True for row in records):
            continue
        bundle = json.loads(bundle_path.read_text())
        stages = {row['stage']: row for row in bundle['private_outcomes']}
        if not {'initial', 'injected', 'recovery'} <= stages.keys():
            raise ValueError('A visual-positive family lacks the three actual stages')
        images = {'SOURCE': bundle['public_source_image_id'], 'INITIAL REQUEST': stages['initial']['image_id'],
                  'CONTROLLED WRONG REQUEST': stages['injected']['image_id'], 'RECOVERY': stages['recovery']['image_id']}
        if bundle['task_id'] in packed:
            if packed[bundle['task_id']] != images:
                raise ValueError('An already packed family changed its images')
            continue
        selected.append({'task_id': bundle['task_id'], 'family_id': bundle['family_id'],
            'state_id': bundle['state_id'], 'public_request': bundle['public_request'], 'images': images,
            'source_bundle': str(bundle_path.resolve()), 'diagnostic_scope': 'Visual admission of existing proposed SFT positives only; original scores and calibration remain unchanged'})
        if len(selected) == args.limit:
            break
    if not selected:
        print(json.dumps({'status': 'no_new_completed_cases', 'total_packed': len(packed)}))
        return
    output = Path('outputs/full-actor-visual-review') / args.batch
    package = Path('deployment') / f'visual-sft-{args.batch}.tar.gz'
    output.mkdir(parents=True, exist_ok=False)
    if package.exists():
        raise FileExistsError(package)
    store = ArtifactStore('data/artifacts')
    for index, row in enumerate(selected):
        canvas = Image.new('RGB', (2048, 640), 'white')
        draw = ImageDraw.Draw(canvas)
        for column, (label, image_id) in enumerate(row['images'].items()):
            image = store.get_image(image_id).convert('RGB')
            image.thumbnail((512, 512))
            canvas.paste(image, (column * 512 + (512 - image.width) // 2, 35 + (512 - image.height) // 2))
            draw.text((column * 512 + 8, 8), label, fill='black')
        caption = row['task_id'] + ' | ' + row['public_request']
        draw.multiline_text((10, 560), '\n'.join(textwrap.wrap(caption, 150)), fill='black')
        name = f'case-{index:03d}.jpg'
        canvas.save(output / name, quality=92)
        row['contact'] = name
    save(output / 'manifest.json', selected)
    with tarfile.open(package, 'w:gz') as archive:
        archive.add(output, arcname=args.batch)
    packed.update({row['task_id']: row['images'] for row in selected})
    save(state_path, packed)
    print(json.dumps({'status': 'prepared', 'cases': len(selected), 'archive': str(package),
                      'bytes': package.stat().st_size, 'total_packed': len(packed)}))


if __name__ == '__main__':
    main()
