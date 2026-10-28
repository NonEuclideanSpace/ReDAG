import html
import json
from pathlib import Path


runs = Path(__file__).resolve().parents[1] / 'runs'
output = runs / 'renderer-multiseed'
output.mkdir(exist_ok=True)
tasks = json.loads((runs / 'renderer-baseline/results.json').read_text())
registrations = [
    (11, runs / 'renderer-baseline', runs / 'renderer-global-ai-evaluation'),
    (23, output / 'seed-23/before', output / 'seed-23/after'),
    (47, output / 'seed-47/before', output / 'seed-47/after'),
]
samples = [{'index': row['index'], 'request': row['request'], 'task_id': row['task_id'], 'source': f'../renderer-baseline/previews/sources/{row["index"]:03d}.jpg', 'seeds': []} for row in tasks]
status = []
for seed, before_dir, after_dir in registrations:
    sides = {}
    for phase, directory in [('before', before_dir), ('after', after_dir)]:
        if not (directory / 'results.json').exists():
            sides[phase] = []
            continue
        metadata = json.loads((directory / 'metadata.json').read_text())
        assert metadata['seed'] == seed
        assert metadata['steps'] == 50 and metadata['eta'] == 0.3 and metadata['cfg_scale'] == 4.0
        assert metadata['optimizer_updates'] == (0 if phase == 'before' else 20)
        rows = json.loads((directory / 'results.json').read_text())
        for index, row in enumerate(rows):
            assert all(row[key] == tasks[index][key] for key in ('index', 'task_id', 'family_id', 'request', 'source_image_id', 'plan'))
            assert row['result']['metadata']['seed'] == seed
        sides[phase] = rows
    paired = 0
    for index, sample in enumerate(samples):
        item = {'seed': seed}
        for phase, directory in [('before', before_dir), ('after', after_dir)]:
            rows = sides[phase]
            image = directory / 'previews' / f'{index:03d}.jpg'
            if index < len(rows) and image.is_file():
                item[phase] = {'image': '../' + str(image.relative_to(runs)), 'seconds': rows[index]['seconds']}
        if 'before' in item and 'after' in item:
            before, after = sides['before'][index], sides['after'][index]
            assert before['result']['bundle_id'] == after['result']['bundle_id'] == after['before_bundle_id']
            assert after['source_condition_reused_exactly'] is True
            paired += 1
        sample['seeds'].append(item)
    status.append({'seed': seed, 'paired_preview_count': paired, 'expected_pairs': 16})
payload = json.dumps(samples, ensure_ascii=False).replace('</', '<\\/')
buttons = ''.join(f'<option value="{row["index"]}">{row["index"]:02d} · {html.escape(row["request"])}</option>' for row in samples)
status_text = ' · '.join(f'种子 {row["seed"]}：{row["paired_preview_count"]}/16 对' for row in status)
document = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>三种子渲染前后对照</title><style>
*{box-sizing:border-box}body{margin:0;background:#f4f6f8;color:#182331;font-family:system-ui,-apple-system,sans-serif;line-height:1.5}main{max-width:1500px;margin:auto;padding:24px}h1{font-size:27px;margin:0 0 10px}h2{font-size:19px;margin:0 0 10px}p{margin:8px 0}.intro,.source,.seed{background:white;border:1px solid #dce2e7;border-radius:9px;padding:18px;margin-bottom:18px}select{max-width:100%;width:900px;font-size:15px;padding:10px;margin:12px 0}.layout{display:grid;grid-template-columns:260px 1fr;gap:18px}.source{height:fit-content;position:sticky;top:15px}.source img{width:100%;height:auto}.pair{display:grid;grid-template-columns:1fr 1fr;gap:16px}figure{margin:0}figcaption{font-size:14px;margin-bottom:8px;color:#4c5c6c}.pair img{display:block;width:100%;aspect-ratio:1;object-fit:contain;background:#edf0f3}.pending{display:grid;place-items:center;min-height:200px;background:#edf0f3;color:#596675}.note{font-size:13px;color:#5d6976}a{color:#245786}.identity{overflow-wrap:anywhere;font-size:12px;color:#5d6976}@media(max-width:760px){main{padding:12px}.layout{grid-template-columns:1fr}.source{position:static;max-width:320px}.pair{gap:8px}.intro,.source,.seed{padding:12px}}</style><main><section class="intro"><h1>同一请求，三个种子的训练前后对照</h1><p>每组使用相同原图、原请求和配对条件包。训练前与训练 20 步后的渲染都固定为 50 步、eta 0.3、CFG 4；仅切换种子 11、23、47。</p><p>以下是实际渲染 JPEG 预览，不使用官方目标图。这 16 条请求仅用于评测。新增种子尚未经过逐图 AI 审阅，不能沿用种子 11 的观察作为新结论。</p><p class="note">本地已载入：'''+status_text+'''。缺少本地预览表示尚未取回，不代表实例没有完成。</p><p><a href="../renderer-global-ai-evaluation/index.html">查看种子 11 的逐项 AI 观察</a> · <a href="local_comparison_status.json">素材核验状态</a></p><label for="sample">选择请求</label><br><select id="sample">'''+buttons+'''</select></section><div class="layout"><aside class="source"><h2>原始图片</h2><a id="source-link" target="_blank"><img id="source-image" alt="真实原始图片"></a><p id="identity" class="identity"></p></aside><div id="pairs"></div></div></main><script>
const samples='''+payload+''';const selector=document.getElementById('sample');const pairs=document.getElementById('pairs');
function render(){const sample=samples[Number(selector.value)];const source=document.getElementById('source-image');source.src=sample.source;document.getElementById('source-link').href=sample.source;document.getElementById('identity').textContent=sample.task_id;pairs.replaceChildren();for(const entry of sample.seeds){const card=document.createElement('section');card.className='seed';const title=document.createElement('h2');title.textContent='种子 '+entry.seed;card.append(title);const grid=document.createElement('div');grid.className='pair';for(const phase of ['before','after']){const figure=document.createElement('figure');const caption=document.createElement('figcaption');caption.textContent=(phase==='before'?'训练前':'训练 20 步后')+(entry[phase]?' · '+entry[phase].seconds.toFixed(2)+' 秒':'');figure.append(caption);if(entry[phase]){const link=document.createElement('a');link.href=entry[phase].image;link.target='_blank';const image=document.createElement('img');image.src=entry[phase].image;image.alt=sample.index+' / seed '+entry.seed+' / '+phase;link.append(image);figure.append(link)}else{const pending=document.createElement('div');pending.className='pending';pending.textContent='本地预览待取回';figure.append(pending)}grid.append(figure)}card.append(grid);pairs.append(card)}}selector.addEventListener('change',render);render();</script></html>'''
(output / 'index.html').write_text(document)
(output / 'local_comparison_status.json').write_text(json.dumps({'seeds': status, 'tasks': 16, 'official_targets_used': False, 'new_seed_visual_review_complete': False, 'pair_alignment_verified': True}, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(status))
