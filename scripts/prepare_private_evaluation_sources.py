import json
from pathlib import Path
import textwrap

from PIL import Image, ImageDraw, ImageFont

from agentic_image.core.artifacts import ArtifactStore
from agentic_image.core.types import PublicTask


base = Path('data/private_eval_v1')
base.mkdir(parents=True, exist_ok=True)
store = ArtifactStore('data/artifacts')
font = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf', 16)
small = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf', 13)
for cohort, path in [('development64', 'data/evaluation/development64/public_tasks.jsonl'), ('training128', 'data/public/expanded_training128.tasks.jsonl')]:
    tasks = [PublicTask.model_validate(json.loads(line)) for line in Path(path).read_text().splitlines()]
    folder = base / cohort
    folder.mkdir(exist_ok=True)
    (folder / 'source_contacts').mkdir(exist_ok=True)
    rows = []
    for index, task in enumerate(tasks):
        ref = store.get_ref(task.user_request.source_image_id)
        image_path = store.root / ref.relative_path
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        rows.append({'index': index, 'task_id': task.task_id, 'family_id': task.family_id, 'split_id': task.split_id, 'request': task.user_request.text, 'source_image_id': ref.artifact_id, 'source_value_id': ref.value_id, 'source_path': str(image_path), 'source_hw': list(ref.shape[:2]), 'source_frame': ref.frame.model_dump(mode='json') if ref.frame else None, 'input_scope': 'original_public_request_and_source_only'})
    destination = folder / 'sources.frozen.jsonl'
    payload = ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows)
    if destination.exists() and destination.read_text() != payload:
        raise ValueError('Frozen source bindings may not change')
    destination.write_text(payload)
    for start in range(0, len(rows), 4):
        sheet = Image.new('RGB', (1040, 1200), 'white')
        draw = ImageDraw.Draw(sheet)
        for offset, row in enumerate(rows[start:start + 4]):
            x, y = offset % 2 * 520, offset // 2 * 600
            draw.text((x + 8, y + 6), f"{cohort} {row['index']:03d} | {row['source_hw']}", fill='black', font=font)
            draw.multiline_text((x + 8, y + 32), '\n'.join(textwrap.wrap(row['request'], width=64)), fill='black', font=small, spacing=3)
            image = Image.open(row['source_path']).convert('RGB')
            image.thumbnail((500, 470))
            sheet.paste(image, (x + 8, y + 120))
        sheet.save(folder / 'source_contacts' / f'{start:03d}-{start + 3:03d}.jpg', quality=93)
    print(json.dumps({'cohort': cohort, 'sources': len(rows), 'target_images_opened': 0, 'output': str(folder)}))
