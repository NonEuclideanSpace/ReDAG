import json
from pathlib import Path
import random
import tarfile
import textwrap

from PIL import Image, ImageDraw

from agentic_image.core.artifacts import ArtifactStore


def main():
    root = Path('runs/development64-fixed-private/scores')
    methods = ('baseline', 'sft200')
    scores = {}
    for method in methods:
        rows = [json.loads(path.read_text()) for path in (root / method).glob('*.json')]
        scores[method] = {(row['index'], row['eval_seed']): row for row in rows if row.get('status') == 'scored'}
    keys = sorted(scores['baseline'].keys() & scores['sft200'].keys())
    selected = [key for key in keys if scores['baseline'][key]['success'] != scores['sft200'][key]['success']]
    output = Path('outputs/renderer-fixed-discordance-review')
    output.mkdir(exist_ok=False)
    store = ArtifactStore('data/artifacts')
    manifest, private = [], []
    generator = random.Random(409611)
    for number, key in enumerate(selected):
        pair = {method: scores[method][key] for method in methods}
        left, right = pair.values()
        for field in ('family_id', 'task_id', 'bundle_id', 'source_image_id', 'condition_recipe'):
            if left[field] != right[field]:
                raise ValueError('Discordant images do not share their original conditions')
        order = list(methods)
        generator.shuffle(order)
        roles = {'SOURCE': left['source_image_id'], 'A': pair[order[0]]['image_id'], 'B': pair[order[1]]['image_id']}
        case_id = f'case-{number:03d}'
        canvas = Image.new('RGB', (1536, 640), 'white')
        draw = ImageDraw.Draw(canvas)
        for column, (role, image_id) in enumerate(roles.items()):
            image = store.get_image(image_id).convert('RGB')
            image.thumbnail((512, 512))
            canvas.paste(image, (column * 512 + (512 - image.width) // 2, 35 + (512 - image.height) // 2))
            draw.text((column * 512 + 8, 8), role, fill='black')
        request = left['plan']['global_text']
        draw.multiline_text((10, 560), '\n'.join(textwrap.wrap(request, 140)), fill='black')
        name = case_id + '.jpg'
        canvas.save(output / name, quality=92)
        manifest.append({'case_id': case_id, 'task_id': left['task_id'], 'family_id': left['family_id'],
            'request': request, 'images': roles, 'contact': name,
            'scope': 'Selected discordant fixed-render outputs; diagnosis only, not an unbiased success estimate'})
        private.append({'case_id': case_id, 'index': key[0], 'seed': key[1], 'A': order[0], 'B': order[1],
            'original_success': {method: pair[method]['success'] for method in methods}})
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    Path('deployment/renderer-fixed-discordance-key.json').write_text(json.dumps(private, indent=2) + '\n')
    package = Path('deployment/renderer-fixed-discordance-review.tar.gz')
    with tarfile.open(package, 'w:gz') as archive:
        archive.add(output, arcname=output.name)
    print(json.dumps({'cases': len(manifest), 'original_paired_observations': len(keys), 'bytes': package.stat().st_size,
        'archive': str(package), 'new_gpu_renders': 0, 'scores_changed': False}))


if __name__ == '__main__':
    main()
