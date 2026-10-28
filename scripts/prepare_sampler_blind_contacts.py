import argparse
import json
from pathlib import Path
import textwrap

from PIL import Image, ImageDraw, ImageFont


parser = argparse.ArgumentParser()
parser.add_argument('--packet', default='runs/sampler-expanded-200/sampler-blind-review')
parser.add_argument('--reviewed', action='append', default=[])
args = parser.parse_args()
root = Path(args.packet).resolve()
records = {}
for path in sorted((root / 'manifests').glob('*.jsonl')):
    for line in path.read_text().splitlines():
        row = json.loads(line)
        previous = records.get(row['review_id'])
        if previous and previous != row:
            raise ValueError('An anonymous review identity changed across transfers')
        records[row['review_id']] = row
reviewed = set()
for path in args.reviewed:
    if Path(path).is_file():
        reviewed.update(json.loads(line)['review_id'] for line in Path(path).read_text().splitlines() if line.strip())
rows = [row for key, row in sorted(records.items()) if key not in reviewed]
folder = root / 'contacts'
folder.mkdir(parents=True, exist_ok=True)
font_path = Path('/System/Library/Fonts/Supplemental/Arial.ttf')
font = ImageFont.truetype(str(font_path), 17) if font_path.is_file() else ImageFont.load_default(size=17)
index = []
for offset in range(0, len(rows), 2):
    group = rows[offset:offset + 2]
    canvas = Image.new('RGB', (1024, 604 * len(group)), 'white')
    draw = ImageDraw.Draw(canvas)
    for position, row in enumerate(group):
        y = position * 604
        draw.text((10, y + 6), row['review_id'] + ' | Source left | Candidate right', font=font, fill='black')
        request = '\n'.join(textwrap.wrap(row['request'], width=112))
        draw.multiline_text((10, y + 29), request, font=font, fill='black', spacing=2)
        for column, role in enumerate(('source_path', 'candidate_path')):
            path = root / row[role]
            with Image.open(path) as image:
                image = image.convert('RGB')
                image.thumbnail((512, 512))
                canvas.paste(image, (column * 512 + (512 - image.width) // 2, y + 88 + (512 - image.height) // 2))
        index.append({**row, 'source_path': str((root / row['source_path']).resolve()),
                      'candidate_path': str((root / row['candidate_path']).resolve())})
    name = group[0]['review_id'] + '_' + group[-1]['review_id'] + '.jpg'
    canvas.save(folder / name, quality=90, optimize=True)
(root / 'available.records.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in index))
print(json.dumps({'available_unreviewed':len(index),'contacts':(len(index)+1)//2,'folder':str(folder),'labels_created':0}),flush=True)
