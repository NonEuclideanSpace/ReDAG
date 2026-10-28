import html
import json
from collections import Counter
from pathlib import Path


root = Path(__file__).resolve().parents[1] / 'runs' / 'renderer-global-ai-evaluation'
reviews = [json.loads(line) for line in (root / 'ai_visual_review.jsonl').read_text().splitlines()]
before = json.loads((root.parent / 'renderer-baseline' / 'results.json').read_text())
after = json.loads((root / 'results.json').read_text())
metadata = json.loads((root / 'metadata.json').read_text())
policy = json.loads((root / 'policy_load.json').read_text())
baseline_metadata = json.loads((root.parent / 'renderer-baseline' / 'metadata.json').read_text())
assert metadata['seed'] == baseline_metadata['seed'] == 11
assert len(reviews) == len(before) == len(after) == 16
assert [row['index'] for row in reviews] == list(range(16))
for index, (old, new) in enumerate(zip(before, after)):
    assert all(old[key] == new[key] for key in ('index', 'task_id', 'family_id', 'request', 'source_image_id', 'plan'))
    assert old['result']['bundle_id'] == new['before_bundle_id'] == new['result']['bundle_id']
    assert old['result']['metadata']['seed'] == new['result']['metadata']['seed'] == 11
    assert all((root / path).is_file() for path in reviews[index]['visual_materials'].values())
summary = {
    'scope': 'qualitative_ai_visual_before_after_review',
    'review_type': 'ai',
    'human_reviewed': False,
    'calibrated': False,
    'used_for_training': False,
    'reviewed': 16,
    'seeds': [11],
    'training_updates': 20,
    'fixed_source_plan_bundle_and_seed': True,
    'overall_change_counts': dict(Counter(row['overall_change'] for row in reviews)),
    'clear_preservation_regression_indices': [row['index'] for row in reviews if row['preservation']['change'] == 'worse'],
    'before_render_seconds': sum(row['seconds'] for row in before),
    'after_render_seconds': sum(row['seconds'] for row in after),
    'limitations': ['Single seed and 16 tasks do not establish generalization or statistical significance.', 'JPEG previews were visually inspected; no human review or judge calibration was performed.', 'Rendering duration differs between runs and does not establish a model speedup.'],
}
(root / 'ai_visual_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
labels = {'worse': '保持性退步', 'similar': '总体相近', 'better': '观察到改善', 'mixed': '有得有失', 'uncertain': '不确定'}
cards = []
for row in reviews:
    index = row['index']
    figures = []
    for key, label in [('source', '原始图片'), ('before', f'训练前 · {row["before_seconds"]:.2f} 秒'), ('after', f'训练 20 步后 · {row["after_seconds"]:.2f} 秒')]:
        path = html.escape(row['visual_materials'][key], quote=True)
        figures.append(f'<figure><figcaption>{label}</figcaption><a href="{path}" target="_blank"><img src="{path}" alt="{index:02d} {label}" loading="lazy"></a></figure>')
    evidence = ''.join(f'<p><b>{label}</b> {html.escape(row[key]["evidence"])}</p>' for key, label in [('edit_completion', '编辑完成度：'), ('preservation', '保持性：'), ('distortion', '异常与失真：')])
    cards.append(f'<article id="sample-{index:02d}" data-change="{row["overall_change"]}"><header><span class="number">{index:02d}</span><h2>{html.escape(row["request"])}</h2><span class="tag {row["overall_change"]}">{labels[row["overall_change"]]}</span></header><div class="images">{"".join(figures)}</div><p class="observation">{html.escape(row["summary"])}</p><details><summary>查看逐项 AI 观察与样本身份</summary>{evidence}<p class="identity">Task: {html.escape(row["task_id"])}<br>Family: {html.escape(row["family_id"])}<br>固定条件: {html.escape(row["fixed_bundle_id"])}</p></details></article>')
document = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>渲染器训练前后 · 16 例真实比较</title><style>
*{box-sizing:border-box}body{margin:0;background:#f3f5f7;color:#182331;font-family:system-ui,-apple-system,sans-serif;line-height:1.55}main{max-width:1540px;margin:auto;padding:30px 22px}h1{font-size:28px;margin:0 0 10px}p{margin:10px 0}a{color:#245786}.intro,article{background:white;border:1px solid #dce2e7;border-radius:10px;padding:20px;margin:0 0 20px}.intro strong{color:#913b22}.muted,.identity{font-size:13px;color:#596575}.controls{display:flex;gap:10px;flex-wrap:wrap;margin-top:16px}button{border:1px solid #aebbc6;background:white;border-radius:6px;padding:8px 13px;cursor:pointer}button[aria-pressed=true]{background:#203d56;color:white}article header{display:flex;align-items:center;gap:12px;margin-bottom:16px}h2{font-size:17px;font-weight:600;flex:1;margin:0}.number{font-variant-numeric:tabular-nums;color:#687989}.tag{font-size:12px;padding:4px 9px;border-radius:5px;background:#edf2f5;white-space:nowrap}.tag.worse{background:#f9e5df;color:#8c3720}.images{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}figure{margin:0;min-width:0}figcaption{font-size:13px;color:#4e6071;margin:0 0 6px}img{display:block;width:100%;height:auto;aspect-ratio:1;object-fit:contain;background:#edf0f3}.observation{font-size:15px;font-weight:500}details{font-size:14px}summary{cursor:pointer;color:#365b79}.identity{overflow-wrap:anywhere}article[hidden]{display:none}@media(max-width:700px){main{padding:16px 10px}.intro,article{padding:12px}.images{gap:5px}figcaption{font-size:11px}h1{font-size:23px}article header{flex-wrap:wrap}h2{font-size:15px}img{aspect-ratio:1}h2{min-width:65%}}</style><main><section class="intro"><h1>渲染器训练前后：16 例固定对照</h1><p>相同的官方开发集请求、原始图片、条件包、种子 11；每张 50 步，eta 0.3，CFG 4。左至右为原图、未训练适配器、20 次真实梯度更新后的适配器。</p><p><strong>AI 逐图观察：4 例保持性明显退步，12 例总体相近，未观察到明确的总体改善。</strong>夏日场景丢失树干、加花重绘整张照片、舞者原姿态改变、牛脸与背景细节改变是主要问题。普通马与单前腿要求仍未完成。</p><p>这些是 AI 对实际预览图的初步观察；没有人工审核或校准评分，不构成统计显著的泛化结论。这 16 例未用于训练，页面没有提供官方目标图作为模型输入。</p><p class="muted">训练后适配器真实加载核验：160 个张量完全相等，LoRA B 非零元素 11,796,480。16 张渲染累计：训练前 339.53 秒，训练后 324.58 秒；此运行时间差不能单独归因于训练。JPEG 仅供显示，原始 PNG 与完整轨迹保留在实例上。</p><p><a href="ai_visual_review.jsonl">逐例观察 JSONL</a> · <a href="ai_visual_summary.json">观察摘要</a> · <a href="policy_load.json">真实权重加载证据</a> · <a href="summary.json">运行报告</a></p><div class="controls"><button data-filter="all" aria-pressed="true">全部 16 例</button><button data-filter="worse" aria-pressed="false">保持性退步 4 例</button><button data-filter="similar" aria-pressed="false">总体相近 12 例</button></div></section>'''+''.join(cards)+'''</main><script>document.querySelectorAll('[data-filter]').forEach(button=>button.addEventListener('click',()=>{document.querySelectorAll('[data-filter]').forEach(other=>other.setAttribute('aria-pressed',String(other===button)));document.querySelectorAll('article').forEach(card=>card.hidden=button.dataset.filter!=='all'&&card.dataset.change!==button.dataset.filter)}));</script></html>'''
(root / 'index.html').write_text(document)
print(json.dumps(summary, ensure_ascii=False))
