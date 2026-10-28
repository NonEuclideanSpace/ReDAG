from concurrent.futures import ThreadPoolExecutor
import io
import json
from pathlib import Path
from urllib.parse import urlencode
import urllib.request
from PIL import Image, ImageDraw

root=Path('data/calibration/ai_reserved')
directory=root/'near_duplicate_review'
directory.mkdir(exist_ok=True)
expected={185:(208,'task_obj_swap_joint_mask_339308'),186:(398,'task_obj_swap_rand_mask_624029'),189:(524,'task_obj_swap_joint_mask_509856')}
url='https://datasets-server.huggingface.co/rows?'+urlencode({'dataset':'TIGER-Lab/OmniEdit-Filtered-1.2M','config':'default','split':'dev','offset':180,'length':10})
with urllib.request.urlopen(url,timeout=45) as response:payload=json.load(response)
metadata=[]
for entry in payload['rows']:
    index=entry['row_idx']
    if index not in expected:continue
    calibration_index,sample=expected[index]
    item=entry['row']
    if item['omni_edit_id']!=sample or entry.get('truncated_cells'):raise ValueError('Official row identity differs')
    for role,column in [('source','src_img'),('target','edited_img')]:
        image_url=item[column]['src']
        if '/--/82455c6cd66db7f0e5bfce8d7a236441af59d6df/--/' not in image_url:raise ValueError('Viewer revision differs')
        metadata.append({'calibration_index':calibration_index,'official_row_index':index,'official_sample_id':'omniedit:'+sample,'official_role':role,'url':image_url,'file':str(directory/f'{calibration_index:03d}_official_{role}.jpg')})
assert len(metadata)==6

def fetch(record):
    with urllib.request.urlopen(record['url'],timeout=35) as response:payload=response.read()
    with Image.open(io.BytesIO(payload)) as image:image.load()
    Path(record['file']).write_bytes(payload)
    return len(payload)

with ThreadPoolExecutor(max_workers=4) as pool:sizes=list(pool.map(fetch,metadata))
for index in [208,398,524]:
    canvas=Image.new('RGB',(1024,1104),'white');draw=ImageDraw.Draw(canvas)
    for y,scope in enumerate(['calibration','official']):
        for x,role in enumerate(['source','target']):
            path=root/'local_images'/f'{index:03d}_{role}.jpg' if scope=='calibration' else directory/f'{index:03d}_official_{role}.jpg'
            with Image.open(path) as source:
                image=source.convert('RGB');image.thumbnail((512,512));canvas.paste(image,(x*512+(512-image.width)//2,y*552+40+(512-image.height)//2))
            draw.text((x*512+12,y*552+12),f'{index} | {scope} | {role}',fill='black')
    canvas.save(directory/f'{index:03d}_comparison.jpg',quality=95)
(directory/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
print(json.dumps({'official_images':6,'bytes':sum(sizes),'comparisons':3,'quarantine_changed':False}),flush=True)
