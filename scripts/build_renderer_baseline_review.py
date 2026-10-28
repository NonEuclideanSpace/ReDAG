import html
import json
from pathlib import Path


root = Path(__file__).resolve().parents[1] / 'runs' / 'renderer-baseline'
rows = json.loads((root / 'results.json').read_text())
summary = json.loads((root / 'summary.json').read_text())
mapping = {row['index']: row for row in json.loads((root / 'source_mapping.json').read_text())}
observation_path = root / 'ai_observations.json'
observations = json.loads(observation_path.read_text()) if observation_path.exists() else {}
escape = html.escape
cards = []
for row in rows:
    index = row['index']
    paths = mapping[index]
    note = observations.get(str(index))
    note_html = (
        '<aside class="observation"><b>AI 初步观察 · 未经人工审核</b><p>'
        + escape(note) + '</p></aside>' if note else ''
    )
    search = escape((row['request'] + ' ' + row['task_id'] + ' ' + (note or '')).lower(), quote=True)
    cards.append(f'''<article id="case-{index:03d}" data-search="{search}" data-observed="{str(bool(note)).lower()}">
      <header class="case-header"><span class="number">{index + 1:02d}</span><div><h2>{escape(row['request'])}</h2><p>{escape(row['task_id'])}</p></div><span class="time">{row['seconds']:.1f} 秒</span></header>
      <div class="comparison"><figure><figcaption>真实原图 <span>source</span></figcaption><a href="{escape(paths['local_source'])}" target="_blank" rel="noopener"><img src="{escape(paths['local_source'])}" alt="第 {index + 1} 条请求的原图" loading="lazy"></a></figure><figure><figcaption>基线输出 <span>未训练 renderer</span></figcaption><a href="{escape(paths['local_output'])}" target="_blank" rel="noopener"><img src="{escape(paths['local_output'])}" alt="第 {index + 1} 条请求的渲染输出" loading="lazy"></a></figure></div>
      {note_html}
      <details><summary>查看输入与运行记录</summary><dl><dt>Source artifact</dt><dd>{escape(row['source_image_id'])}</dd><dt>Source value</dt><dd>{escape(paths['source_value_id'])}</dd><dt>Output artifact</dt><dd>{escape(row['result']['image_id'])}</dd><dt>完整原图（实例）</dt><dd>{escape(paths['source_path'])}</dd><dt>完整输出 PNG（实例）</dt><dd>{escape(paths['output_path'])}</dd><dt>完整渲染轨迹（实例）</dt><dd>{escape(row['result']['trajectory_id'])}</dd></dl></details>
    </article>''')

document = '''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>真实原图与渲染基线 · 16 条请求</title>
<style>
:root{color-scheme:light;--ink:#17242f;--muted:#5a6a78;--line:#dce4e9;--blue:#225da6}*{box-sizing:border-box}body{margin:0;background:#f2f5f7;color:var(--ink);font:16px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}main{max-width:1160px;margin:auto;padding:40px 24px 72px}h1{font-size:32px;line-height:1.3;margin:8px 0 16px;letter-spacing:-.5px}h2{font-size:20px;line-height:1.4;margin:0;font-weight:650}p{margin:8px 0}.eyebrow{font-size:13px;font-weight:700;letter-spacing:.1em;color:var(--blue)}.intro{max-width:920px;color:var(--muted)}.facts{display:flex;flex-wrap:wrap;gap:10px;margin:22px 0}.facts span{background:white;border:1px solid var(--line);padding:7px 12px;border-radius:6px;font-size:14px}.notice{padding:16px 18px;border-left:4px solid #d69a29;background:#fff9e9;border-radius:4px;color:#594821;font-size:14px}.toolbar{position:sticky;top:0;z-index:2;display:flex;align-items:center;gap:16px;padding:14px 0;background:#f2f5f7f5;backdrop-filter:blur(8px);margin-top:12px}.toolbar input[type=search]{font:inherit;background:white;border:1px solid #c4d1dc;border-radius:6px;padding:9px 12px;flex:1;min-width:120px}.toolbar label{font-size:14px;white-space:nowrap}.count{color:var(--muted);font-size:14px;white-space:nowrap}article{background:white;border:1px solid var(--line);border-radius:10px;overflow:hidden;margin:16px 0 30px;scroll-margin-top:80px}.case-header{display:flex;align-items:flex-start;gap:14px;padding:20px}.case-header>div{flex:1;min-width:0}.case-header p{color:var(--muted);font-size:12px;overflow-wrap:anywhere;margin:6px 0 0}.number{font-size:14px;color:var(--blue);background:#edf4fc;border-radius:6px;padding:5px 9px;font-weight:700}.time{font-size:12px;white-space:nowrap;color:var(--muted);margin-top:5px}.comparison{display:grid;grid-template-columns:1fr 1fr;gap:1px;background:var(--line);border-top:1px solid var(--line);border-bottom:1px solid var(--line)}figure{margin:0;background:#f8fafb;min-width:0}figcaption{background:#fff;padding:10px 16px;font-size:14px;font-weight:650;border-bottom:1px solid var(--line)}figcaption span{color:var(--muted);font-size:12px;font-weight:400;margin-left:9px}figure a{display:block;height:440px}img{display:block;width:100%;height:100%;object-fit:contain}.observation{margin:16px 20px;padding:12px 16px;border:1px solid #cdddeb;background:#f4f8fc;border-radius:6px;font-size:14px}.observation b{font-size:12px;color:#315d82}.observation p{margin:5px 0 0}details{padding:12px 20px;color:var(--muted);font-size:12px}summary{cursor:pointer}dl{display:grid;grid-template-columns:150px 1fr;gap:8px 16px;margin:14px 0 4px}dt{font-weight:600}dd{margin:0;overflow-wrap:anywhere;font-family:ui-monospace,monospace}footer{color:var(--muted);font-size:13px}a{color:var(--blue)}[hidden]{display:none!important}@media(max-width:700px){main{padding:24px 12px 48px}h1{font-size:26px}.case-header{padding:16px 12px;gap:10px}h2{font-size:17px}.time{display:none}figure a{height:300px}.comparison{grid-template-columns:1fr}.toolbar{gap:9px;flex-wrap:wrap}.toolbar input[type=search]{width:100%}dl{grid-template-columns:1fr;gap:4px}dd{margin-bottom:10px}.observation{margin:12px}}
</style></head><body><main><div class="eyebrow">RENDERER BASELINE · REAL SOURCES</div><h1>16 条真实请求：原图与渲染基线</h1><p class="intro">并排查看真实 source、原始请求与 FLUX.2-klein-base-4B 的本次输出。渲染输入为原图和原请求，没有使用官方目标图；这批 renderer 的优化器更新次数为 0。</p>
<div class="facts">FACTS</div><div class="notice">图片为仅供页面显示的 JPEG 预览，最长边不超过 1024px；点击可单独查看。完整 PNG、原始 source 与轨迹保留在租用实例。AI 初步观察不代表人工审核、校准标签或视觉质量分数，也不会写入训练标签。</div>
<div class="toolbar"><input id="search" type="search" placeholder="筛选请求、任务 ID 或观察内容" aria-label="筛选样本"><label><input id="observed" type="checkbox"> 只看有 AI 观察的样本</label><span id="count" class="count">16 / 16</span></div><section id="cards">CARDS</section>
<footer><p>每张图均来自已完成的一次真实渲染。本页未启动任何训练或重新渲染。</p><p>运行证据：<a href="metadata.json">metadata.json</a> · <a href="results.json">results.json</a> · <a href="summary.json">summary.json</a> · <a href="source_mapping.json">source 对应记录</a> · <a href="ai_observations.json">AI 初步观察</a></p></footer></main>
<script>
const cards=Array.from(document.querySelectorAll('article'));const search=document.getElementById('search');const observed=document.getElementById('observed');function update(){const query=search.value.trim().toLowerCase();let count=0;for(const card of cards){const visible=card.dataset.search.includes(query)&&(!observed.checked||card.dataset.observed==='true');card.hidden=!visible;if(visible)count++}document.getElementById('count').textContent=count+' / '+cards.length}search.addEventListener('input',update);observed.addEventListener('change',update);
</script></body></html>'''
facts = [f'{len(rows)} 条真实渲染', f'{summary["steps"]} steps', f'CFG {summary["cfg_scale"]}', f'η {summary["eta"]}', f'seed {summary["seed"]}', f'总渲染 {summary["render_seconds"]:.1f} 秒']
document = document.replace('FACTS', ''.join(f'<span>{escape(text)}</span>' for text in facts)).replace('CARDS', '\n'.join(cards))
(root / 'index.html').write_text(document)
print(root / 'index.html')
