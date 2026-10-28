import argparse
from hashlib import blake2b
from html import escape
import json
from pathlib import Path
import shutil

from agentic_image.core.types import PlanSpec, PrivateEvaluation


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_jsonl(path, rows):
    Path(path).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def write_report(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


DIMENSIONS = (
    ("task_completion", "目标图完成了原始请求"),
    ("mask", "白色编辑区域可信"),
    ("identity", "主体身份和应保留内容正确"),
    ("registration", "源图与目标图可按同一画布对应"),
    ("original_request_coverage", "独立评估条目完整覆盖原始请求，且无新增要求"),
    ("plan_compatible", "下方具体 PLAN 与目标图、原始请求相容"),
    ("evaluation_approved", "独立评估条目的谓词和预期值逐项正确"),
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--proposals", required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    output = Path(arguments.output)
    if output.exists():
        raise FileExistsError("Review packets are immutable; choose a fresh output directory")
    proposals_dir = Path(arguments.proposals)
    candidates = list(read_jsonl("data/review/viewer_candidates.jsonl"))
    proposals = {row["sample_id"]: row for row in read_jsonl(proposals_dir / "proposals.jsonl")}
    compiled = {row["sample_id"]: row for row in read_jsonl(proposals_dir / "compiled_plans.jsonl")}
    isolation = json.loads(Path("data/review/viewer_candidate_target_isolation.json").read_text())
    isolation_rows = {row["sample_id"]: row for row in isolation["results"]}
    original = {row["sample_id"]: row for row in read_jsonl("data/review/viewer_candidate_packet/annotations.jsonl")}
    output.mkdir(parents=True)
    shutil.copytree("data/review/viewer_candidate_packet/previews", output / "previews")
    keys, annotations, sections = [], [], []
    for index, candidate in enumerate(candidates):
        sample = candidate["sample_id"]
        proposal = proposals.get(sample, {})
        recipe = compiled.get(sample)
        annotation = original[sample]
        evaluation = proposal.get("evaluation")
        plan = proposal.get("plan")
        if evaluation:
            PrivateEvaluation.model_validate(evaluation)
            annotation["evaluation"] = evaluation
        if plan:
            PlanSpec.model_validate(plan)
        if recipe and recipe["plan"] != plan:
            raise ValueError("Compiled plan differs from the exact proposed plan")
        ready = bool(plan and evaluation and recipe and not isolation_rows[sample]["quarantined"])
        annotation.update(verified=False, semantic_compatible=False, registration_verified=False,
                          human_reviewed=False, task_definition_verified=False)
        key = {"sample_id": sample, "family_id": candidate["family_id"],
               "source_image_id": candidate["source_image_id"],
               "target_image_id": candidate["target_image_id"],
               "edit_mask_id": candidate["edit_mask_id"], "request": candidate["instruction"],
               "plan": plan, "evaluation": evaluation, "compiled_plan": recipe,
               "isolation": isolation_rows[sample], "review_ready": ready,
               "human_reviewed": False, "actor_teacher_trajectory": False,
               "proposal_teacher": proposal.get("teacher_model"),
               "proposal_teacher_revision": proposal.get("teacher_revision"),
               "automated_coverage": proposal.get("automated_coverage"),
               "proposal_errors": proposal.get("errors", []),
               "source_dataset": candidate["source_dataset"]}
        key["review_revision"] = blake2b(json.dumps(key, sort_keys=True).encode(), digest_size=16).hexdigest()
        keys.append(key)
        annotations.append(annotation)
        figures = "".join(
            f'<figure><img loading="lazy" src="previews/{index:04d}_{role}.{extension}"><figcaption>{label}</figcaption></figure>'
            for role, extension, label in (("source", "jpg", "公开源图"),
                                          ("target", "jpg", "私有目标图"),
                                          ("mask", "png", "官方编辑区：白色表示允许编辑"))
        )
        atoms = "".join(
            f'<tr><td>{escape(atom["raw_span"])}</td><td>{escape(atom["predicate"])}</td><td>{escape(json.dumps(atom["expected_value"], ensure_ascii=False))}</td><td>{escape(atom["checker_route"])}</td></tr>'
            for atom in (evaluation or {}).get("eval_atoms", [])
        )
        requirements = "".join(f'<li>{escape(row["text"])}</li>' for row in (plan or {}).get("requirements", []))
        fields = []
        for name, label in DIMENSIONS:
            choices = "".join(f'<label><input type="radio" name="r{index}_{name}" value="{value}" {"disabled" if not ready else ""}>{text}</label>'
                              for value, text in (("true", "正确"), ("false", "错误"), ("unknown", "无法确认")))
            fields.append(f'<div class="rating"><span>{label}</span><div>{choices}</div></div>')
        state = "待人工审核：模型建议与实际编译已具备" if ready else "资料不完整或被隔离：禁止接受，请查看错误记录"
        sections.append(f'<section><h2>{index + 1}. {escape(sample)}</h2><p class="status">{state}</p><p class="request">{escape(candidate["instruction"])}</p><div class="images">{figures}</div><h3>具体 PLAN · 仅依据公开源图和请求</h3><p>{escape((plan or {}).get("global_text", "尚无有效提案"))}</p><ul>{requirements}</ul><h3>独立评估条目 · 单独调用生成</h3><table><thead><tr><th>原文跨度</th><th>检查要求</th><th>预期值</th><th>检查类型</th></tr></thead><tbody>{atoms}</tbody></table><details><summary>完整 PLAN、评估条目与编译证据</summary><pre>{escape(json.dumps(key, indent=2, ensure_ascii=False))}</pre></details>{"".join(fields)}<label>审核备注<textarea data-notes rows="2"></textarea></label></section>')
    data = json.dumps(keys, ensure_ascii=False).replace("<", "\\u003c")
    names = json.dumps([name for name, _ in DIMENSIONS])
    script = """const records=__DATA__;const dimensions=__DIMENSIONS__;document.querySelector('#export').onclick=()=>{const reviewer=document.querySelector('#reviewer').value.trim();if(!reviewer){alert('请填写实际审核者姓名');return;}const rows=[...document.querySelectorAll('section')].map((section,i)=>{const ratings=Object.fromEntries(dimensions.map(name=>{const value=section.querySelector(`[name="r${i}_${name}"]:checked`)?.value;return[name,value==='true'?true:value==='false'?false:null];}));return{sample_id:records[i].sample_id,review_revision:records[i].review_revision,scope:'independent_source_only_plan_and_annotation_review_v1',reviewer,reviewed:records[i].review_ready&&Object.values(ratings).every(value=>value!==null),ratings,notes:section.querySelector('[data-notes]').value,reviewed_at:new Date().toISOString()};});const blob=new Blob([rows.map(row=>JSON.stringify(row)).join('\\n')+'\\n'],{type:'application/x-ndjson'});const link=document.createElement('a');link.href=URL.createObjectURL(blob);link.download='viewer_human_reviews.jsonl';link.click();setTimeout(()=>URL.revokeObjectURL(link.href),1000);};""".replace("__DATA__", data).replace("__DIMENSIONS__", names)
    page = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>真实配对候选审核</title><style>body{font:16px system-ui;background:#f3f5f7;color:#16222b;margin:24px}header,section{padding:24px;background:#fff;margin:24px 0;border-radius:12px}p{line-height:1.6}h1{font-size:28px}h2{font-size:22px}.images{display:flex;gap:16px;flex-wrap:wrap}figure{flex:1;min-width:220px;margin:0}img{max-width:100%;height:300px;object-fit:contain;background:#eee}figcaption{padding:8px 0}.status{background:#fff1c9;padding:12px}.request{font-size:20px}table{border-collapse:collapse;width:100%;margin:16px 0}td,th{border:1px solid #ddd;text-align:left;padding:10px;vertical-align:top}.rating{display:flex;justify-content:space-between;gap:18px;border-top:1px solid #ddd;padding:12px 0}.rating label{margin-right:14px}textarea{width:100%;display:block;margin-top:8px}pre{white-space:pre-wrap;word-break:break-word;background:#f5f6f8;padding:16px}button,input[type=text]{padding:12px;margin-right:12px}header{position:relative}@media(max-width:700px){.rating{display:block}.rating div{margin-top:10px}table{font-size:13px}}</style><header><h1>32 组真实候选：具体 PLAN 与独立评估条目</h1><p class="status">当前全部未获人工审核。自动建议由冻结 8B 模型从公开源图与原始请求分别生成；它没有看到目标图或编辑区域。PLAN 已使用实际 4B 配套编码器编译，但不是控制器的教师轨迹。评估阈值仍未校准。</p><p>请逐项检查源图、目标图、编辑区域、具体 PLAN 和评估条目，再选“正确 / 错误 / 无法确认”。本页只接受这些明确展示内容的审核。导出结果可导入独立标注；后续若控制器重新生成 PLAN，仍须审查对应配对结果。该操作不会自动解锁完整训练或替代 600 家族的裁判校准。</p><input id="reviewer" type="text" placeholder="实际审核者姓名"><button id="export">导出逐项审核结果</button></header>' + "".join(sections) + '<script>' + script + '</script></html>'
    ready_count = sum(key["review_ready"] for key in keys)
    page = page.replace("32 组真实候选：具体 PLAN 与独立评估条目",
                        f"32 组真实候选：{ready_count} 组资料可审核，{len(keys) - ready_count} 组缺少独立标注")
    (output / "index.html").write_text(page)
    write_jsonl(output / "keys.jsonl", keys)
    write_jsonl(output / "annotations.pending.jsonl", annotations)
    write_report(output / "isolation.json", isolation)
    report = {"status": "awaiting_human_review", "samples": len(keys),
              "review_ready": sum(key["review_ready"] for key in keys),
              "human_approved": 0, "calibrated": False, "actor_teacher_trajectories": 0,
              "proposals_path": str(proposals_dir.resolve()),
              "candidate_path": str(Path("data/review/viewer_candidates.jsonl").resolve()),
              "scope": "independent_annotation_and_specific_compiled_plan_review",
              "final_actor_generated_pair_audit_complete": False}
    write_report(output / "report.json", report)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
