import argparse
import io
import json
from pathlib import Path
import tarfile
from PIL import Image, ImageDraw

parser = argparse.ArgumentParser()
parser.add_argument('--start', type=int, required=True)
parser.add_argument('--stop', type=int, required=True)
args = parser.parse_args()
root = Path('data/calibration/ai_reserved')
rows = [row for row in map(json.loads, (root/'batch_600.jsonl').read_text().splitlines()) if args.start <= row['calibration_index'] < args.stop]
directory = root/'contact_sheets_remote'
directory.mkdir(exist_ok=True)
paths = []
for offset in range(0, len(rows), 4):
    group = rows[offset:offset+4]
    canvas = Image.new('RGB', (1024, 552*len(group)), 'white')
    draw = ImageDraw.Draw(canvas)
    for y, row in enumerate(group):
        for x, role in enumerate(('source','target')):
            record = next(image for image in row['images'] if image['candidate_role']==role)
            with Image.open(Path('data/artifacts')/record['relative_path']) as source:
                image = source.convert('RGB')
                image.thumbnail((512,512))
                canvas.paste(image, (x*512+(512-image.width)//2, y*552+40+(512-image.height)//2))
            draw.text((x*512+12,y*552+12),f"{row['calibration_index']:03d} | {role} | {row['sample_id']}",fill='black')
    destination = directory/f"{group[0]['calibration_index']:03d}-{group[-1]['calibration_index']:03d}.jpg"
    canvas.save(destination,quality=70,optimize=True)
    paths.append(destination)
archive = root/f'contacts_{args.start:03d}_{args.stop-1:03d}.tar.gz'
with tarfile.open(archive,'w:gz') as stream:
    for path in paths:stream.add(path,arcname=str(path))
report = {'start':args.start,'stop':args.stop,'families':len(rows),'actual_source_images':2*len(rows),'contact_sheets':len(paths),'archive_bytes':archive.stat().st_size,'private_artifacts_reencoded_for_review':True,'labels_created':0}
(root/f'contacts_{args.start:03d}_{args.stop-1:03d}.report.json').write_text(json.dumps(report,indent=2)+'\n')
if archive.stat().st_size > 5_000_000: raise RuntimeError('Contact bundle exceeds five MB')
print(json.dumps(report),flush=True)
