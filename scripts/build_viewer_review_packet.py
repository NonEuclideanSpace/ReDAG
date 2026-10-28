from html import escape
import json
from pathlib import Path

import numpy as np
from PIL import Image

from agentic_image.core.artifacts import ArtifactStore
from agentic_image.data.annotation import prepare_annotation_packet
from agentic_image.data.io import read_jsonl, write_jsonl, write_report


def main():
    root = Path("data")
    store = ArtifactStore(root / "artifacts")
    tasks = {row["task_id"]: row for row in read_jsonl(root / "public/protocol_pilot_tasks.jsonl")}
    candidates = []
    for row in read_jsonl(root / "review/viewer_candidates_pending.jsonl"):
        if row["sample_id"] not in tasks:
            continue
        task = tasks[row["sample_id"]]
        if row["source_image_id"] != task["user_request"]["source_image_id"]:
            raise ValueError("Review source differs from the actual protocol task image")
        row["family_id"] = task["family_id"]
        row["split_id"] = "train"
        candidates.append(row)
    path = root / "review/viewer_candidates.jsonl"
    write_jsonl(path, candidates)
    output = root / "review/viewer_candidate_packet"
    report = prepare_annotation_packet(path, output, store, limit=len(candidates))
    sections = []
    dimensions = (
        ("task_completion", "目标图完成了原始请求"),
        ("mask", "白色编辑区域可信"),
        ("identity", "主体身份和应保留内容正确"),
        ("registration", "源图与目标图可按同一画布对应"),
        ("original_request_coverage", "没有遗漏原始请求要求"),
    )
    for index, row in enumerate(candidates):
        figures = []
        for role, label, identifier in (
            ("source", "公开源图", row["source_image_id"]),
            ("target", "私有目标图", row["target_image_id"]),
        ):
            image = store.get_image(identifier)
            shape = image.size
            image.thumbnail((512, 512))
            image.save(output / "previews" / f"{index:04d}_{role}.jpg", quality=88)
            figures.append(
                f'<figure><img loading="lazy" src="previews/{index:04d}_{role}.jpg"><figcaption>{label} · {shape[0]}×{shape[1]}</figcaption></figure>'
            )
        mask_id = row.get("edit_mask_id")
        if mask_id:
            mask = store.get_array(mask_id)
            image = Image.fromarray(np.uint8(np.clip(mask, 0, 1) * 255))
            image.thumbnail((512, 512))
            image.save(output / "previews" / f"{index:04d}_mask.png")
            figures.append(
                f'<figure><img loading="lazy" src="previews/{index:04d}_mask.png"><figcaption>官方透明区域反解 · 白色为编辑区 · 原尺寸{mask.shape[1]}×{mask.shape[0]}</figcaption></figure>'
            )
        fields = []
        for key, label in dimensions:
            options = "".join(
                f'<label><input type="radio" name="r{index}_{key}" value="{value}">{text}</label>'
                for value, text in (("true", "正确"), ("false", "错误"), ("unknown", "无法确认"))
            )
            fields.append(f'<div class="rating"><span>{label}</span><div>{options}</div></div>')
        sections.append(
            f'<section data-sample="{escape(row["sample_id"])}"><h2>{index + 1}. {escape(row["sample_id"])}</h2><p class="request">{escape(row["instruction"])}</p><div class="images">{"".join(figures)}</div>{"".join(fields)}<label>备注<textarea data-notes rows="2"></textarea></label></section>'
        )
    script = """const keys=['task_completion','mask','identity','registration','original_request_coverage'];document.querySelector('#export').onclick=()=>{const reviewer=document.querySelector('#reviewer').value.trim();if(!reviewer){alert('请填写审核者姓名');return;}const rows=[...document.querySelectorAll('section')].map((s,i)=>{const ratings=Object.fromEntries(keys.map(k=>{const v=s.querySelector(`[name="r${i}_${k}"]:checked`)?.value;return[k,v==='true'?true:v==='false'?false:null];}));return{sample_id:s.dataset.sample,reviewer,reviewed:Object.values(ratings).every(v=>v!==null),ratings,notes:s.querySelector('[data-notes]').value,scope:'independent_candidate_review_before_shared_plan',reviewed_at:new Date().toISOString()};});const blob=new Blob([rows.map(r=>JSON.stringify(r)).join('\\n')+'\\n'],{type:'application/x-ndjson'});const link=document.createElement('a');link.href=URL.createObjectURL(blob);link.download='candidate_human_reviews.jsonl';link.click();setTimeout(()=>URL.revokeObjectURL(link.href),1000);};"""
    html = f'<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>小模型候选数据审阅</title><style>body{{font:16px system-ui;margin:24px;background:#f5f6f8;color:#18242d}}header,section{{background:white;padding:24px;margin:20px 0;border-radius:10px}}h1{{font-size:26px}}h2{{font-size:19px}}p{{line-height:1.6}}.images{{display:flex;gap:16px;flex-wrap:wrap}}figure{{margin:0;flex:1;min-width:240px}}img{{width:100%;max-width:512px;max-height:512px;object-fit:contain;background:#eee}}figcaption{{font-size:13px;margin:8px 0 20px}}.request{{font-size:19px}}.rating{{display:flex;justify-content:space-between;border-top:1px solid #ddd;padding:12px 0;gap:20px}}.rating label{{margin-right:16px}}textarea{{display:block;width:100%;margin-top:8px}}button{{padding:10px 18px;cursor:pointer}}input[type=text]{{padding:10px}}.status{{background:#fff4d6;padding:16px}}@media(max-width:700px){{.rating{{display:block}}.rating div{{margin-top:10px}}}}</style><header><h1>32 个真实候选：源图、目标图与修正后的官方编辑区域</h1><p class="status">当前全部未审核。图片来自固定数据版本的官方查看器，经过重新编码，使用新的图像标识。旧配准结论未沿用；白色区域只表示官方透明通道的解码结果，不表示已通过人工检查。</p><p>本页供独立候选标注。共享 PLAN、独立评估原子和最终配对审核仍待完成；本页的确认不会自动解锁生成器训练。公开协议训练独立进行。逐项选择“正确 / 错误 / 无法确认”，填入审核者后导出 JSONL。</p><input id="reviewer" type="text" placeholder="审核者姓名"><button id="export">导出审核结果</button></header>{"".join(sections)}<script>{script}</script></html>'
    (output / "index.html").write_text(html)
    report.update(
        status="awaiting_human_candidate_review",
        official_alpha_masks_decoded=len(candidates),
        human_audit_accepted=False,
        independent_evaluation_atoms_pending=True,
        shared_plan_audit_pending=True,
        paired_target_leakage_audit_pending=True,
    )
    write_report(output / "report.json", report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
