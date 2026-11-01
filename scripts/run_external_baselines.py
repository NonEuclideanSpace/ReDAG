import argparse
import json
from pathlib import Path
import re
import sys

import numpy as np
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from redag.edit.compositor import paste_window_strictly_local
from redag.graph.dag import CANVAS
from redag.graph.schema import ControllerAction
from redag.harness.environment import requirement_statement
from redag.learn.sft_trainer import read_records
from redag.models.editor import FluxEditor
from redag.models.planner import preserve_previous_conditions
from redag.verify.grounder import load_image
from redag.verify.verifier import TwoStageVerifier as Verifier, export_view, load_vlm_scorer

EDIT_THINKER_PROMPT = """# Edit Evaluation and Prompt Refinement System

You are an expert image editing evaluator and prompt engineer. Your task is to:
1. Score the edited image from two perspectives and output the result in JSON format.
2. If you think the edited image is not good enough, generate an optimized rewritten prompt that addresses the original shortcomings; if you think it is good enough, output the original rewritten prompt.

## Input Information
You have been presented with two images in sequence:
- Original Image: The input image before editing.
- Edited Image: The resulting image after applying the rewritten prompt.

Original User Instruction: "{original_instruction}"
Rewritten Prompt: "{rewritten_prompt}"

Output format:
<think>
Detailed explanation of evaluation and new rewritten prompt.
</think>
<score>
{{
    "semantic": 0,
    "quality": 0
}}
</score>
<answer>
Improved rewritten prompt
</answer>"""

REPLAN_PROMPT = """You are an expert AI image editing assistant. Your task is to carefully analyze a user's editing instruction and input image, reason step by step, and then decompose the necessary actions into global and local edits.

### Rules
1. Global Edits: Affect the entire image's style, lighting, color grading, or overall composition.
2. Local Edits: Target specific objects or areas. These instructions go into the `<region>` tag.
3. Hint Quality: The `hint` text MUST be a concise, visually descriptive instruction for its specific region.
4. Edge Case - No Global Edits: If no global edits are necessary, `<gen_image>` MUST be placeholder 'keep remaining part of image unchanged.'.

### Output Format
<think>Reasoning process</think><gen_image>Global edit instruction</gen_image><region>[{"bbox_2d": [10,150,150,210], "point_2d": [30,175], "hint": "change the color of this one apple to blue"}]</region>

The user instruction: """

GENREVISIT_PROMPT = """You are a verifier-grounded multimodal image retry planner.
Given the task instruction, the atomic constraints, and the candidate history with each candidate's attempt_id, GM score, passed constraints, and failed constraints, return a single JSON object selecting the best historical candidate to edit from and a concise editing instruction that fixes the failed constraints while preserving the passed constraints:
{"action": "edit_image", "source_attempt_id": 0, "instruction": "..."}
Return only valid JSON."""


def tensor_to_pil(image):
    arr = (image.detach().cpu().permute(1, 2, 0).numpy() * 255).round().clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def load_mllm(model_id, device):
    import os
    from transformers import AutoModelForImageTextToText, AutoProcessor
    repo_map = {
        'RePlan-Qwen2.5-VL-7B': 'TainU/RePlan-Qwen2.5-VL-7B',
        'GenRevisit-RL': 'GenRevisit/GenRevisit-RL',
        'EditThinker-8B': 'appletea2333/EditThinker-8B',
    }
    model_path = Path(model_id)
    for key, repo_id in repo_map.items():
        if key in str(model_id):
            index_file = model_path / 'model.safetensors.index.json'
            complete = False
            if index_file.exists():
                idx_data = json.loads(index_file.read_text())
                shards = set(idx_data.get('weight_map', {}).values())
                complete = bool(shards) and all((model_path / s).exists() for s in shards)
            if not complete:
                os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
                from huggingface_hub import snapshot_download
                target_dir = str(model_path.resolve()) if model_path.is_symlink() else str(model_path)
                snapshot_download(repo_id=repo_id, local_dir=target_dir, max_workers=8)
            break
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(device).eval()
    model.requires_grad_(False)
    return processor, model


@torch.no_grad()
def generate_mllm_text(processor, model, images, text, device, max_new_tokens=384):
    content = [{'type': 'image', 'image': tensor_to_pil(img)} for img in images]
    content.append({'type': 'text', 'text': text})
    messages = [{'role': 'user', 'content': content}]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors='pt',
    ).to(device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    gen = out[0, inputs['input_ids'].shape[1]:]
    return processor.decode(gen, skip_special_tokens=True).strip()


def score_all_nodes(verifier, image, dag):
    scores = {}
    for nid, node in dag.nodes.items():
        scores[nid] = float(verifier.score(image, node.question))
    return scores


def weighted_mean_score(dag, scores):
    total_w = sum(n.weight for n in dag.nodes.values()) or 1.0
    return sum(dag.nodes[nid].weight * scores.get(nid, 0.0) for nid in dag.nodes) / total_w


def geometric_mean_score(dag, scores):
    if not dag.nodes:
        return 0.0
    prod = 1.0
    for nid in dag.nodes:
        prod *= max(1e-4, float(scores.get(nid, 0.0)))
    return float(prod ** (1.0 / len(dag.nodes)))


def normalize_box_2d(bbox_2d, height, width):
    if not isinstance(bbox_2d, (list, tuple)) or len(bbox_2d) != 4:
        return CANVAS
    x1, y1, x2, y2 = [float(v) for v in bbox_2d]
    if max(x1, y1, x2, y2) > 1.5:
        if max(x1, y1, x2, y2) <= 1000.0 and (width != 1000 or height != 1000):
            x1, x2 = x1 / 1000.0, x2 / 1000.0
            y1, y2 = y1 / 1000.0, y2 / 1000.0
        else:
            x1, x2 = x1 / width, x2 / width
            y1, y2 = y1 / height, y2 / height
    x1, x2 = max(0.0, min(1.0, min(x1, x2))), max(0.0, min(1.0, max(x1, x2)))
    y1, y2 = max(0.0, min(1.0, min(y1, y2))), max(0.0, min(1.0, max(y1, y2)))
    if x2 - x1 < 0.05 or y2 - y1 < 0.05:
        return CANVAS
    return (round(x1, 4), round(y1, 4), round(x2, 4), round(y2, 4))


def run_single_session(args, record, editor, verifier, mllm_bundle):
    image = load_image(record['image'])
    dag = None
    gen_calls = 0
    ver_calls = 0
    steps_per_turn = max(1, args.max_steps)

    for turn_idx, turn in enumerate(record['turns']):
        dag = preserve_previous_conditions(turn['graph'], dag, turn.get('superseded_ids', turn.get('supersedes')))
        user_prompt = turn['user_prompt']
        turn_seed = args.seed + turn_idx * 100

        if args.baseline in ('direct_flux', 'mt_opsd'):
            action = ControllerAction('modify_attr', CANVAS, user_prompt)
            crop = editor.render(image, action, seed=turn_seed)
            image = paste_window_strictly_local(image, crop, CANVAS, 0)
            gen_calls += 1

        elif args.baseline == 'best_of_n':
            best_img, best_score = image, -1.0
            for k in range(args.candidates_per_turn):
                action = ControllerAction('modify_attr', CANVAS, user_prompt)
                crop = editor.render(image, action, seed=turn_seed + k)
                cand = paste_window_strictly_local(image, crop, CANVAS, 0)
                gen_calls += 1
                scores = score_all_nodes(verifier, cand, dag)
                ver_calls += len(dag.nodes)
                s = weighted_mean_score(dag, scores)
                if s > best_score:
                    best_score, best_img = s, cand
            image = best_img

        elif args.baseline == 'vision_director':
            # VisionDirector-style: global-to-local flat goal ordering + rollback if overall checklist score does not improve
            cur_scores = score_all_nodes(verifier, image, dag)
            ver_calls += len(dag.nodes)
            cur_mean = weighted_mean_score(dag, cur_scores)
            order_rank = {'background': 0, 'count': 1, 'prohibition': 1, 'entity': 2, 'attribute': 3, 'relation': 4}
            unsat = [nid for nid, node in dag.nodes.items() if cur_scores.get(nid, 0.0) < node.threshold]
            unsat.sort(key=lambda nid: (order_rank.get(dag.nodes[nid].node_type, 5), -dag.nodes[nid].weight))
            for step_i, nid in enumerate(unsat[:steps_per_turn]):
                goal_inst = f'{user_prompt}. Make sure that {requirement_statement(dag.nodes[nid])}.'
                action = ControllerAction('modify_attr', CANVAS, goal_inst)
                crop = editor.render(image, action, seed=turn_seed + step_i)
                cand = paste_window_strictly_local(image, crop, CANVAS, 0)
                gen_calls += 1
                cand_scores = score_all_nodes(verifier, cand, dag)
                ver_calls += len(dag.nodes)
                cand_mean = weighted_mean_score(dag, cand_scores)
                if cand_mean > cur_mean + 1e-4:
                    image, cur_scores, cur_mean = cand, cand_scores, cand_mean

        elif args.baseline == 'genrevisit':
            processor, mllm = mllm_bundle
            # Initial turn edit
            action = ControllerAction('modify_attr', CANVAS, user_prompt)
            crop = editor.render(image, action, seed=turn_seed)
            cand0 = paste_window_strictly_local(image, crop, CANVAS, 0)
            gen_calls += 1
            scores0 = score_all_nodes(verifier, cand0, dag)
            ver_calls += len(dag.nodes)
            archive = [{'id': 0, 'image': cand0, 'scores': scores0, 'gm': geometric_mean_score(dag, scores0)}]
            for step_i in range(1, min(3, steps_per_turn)):
                best_item = max(archive, key=lambda x: x['gm'])
                if all(best_item['scores'][nid] >= dag.nodes[nid].threshold for nid in dag.nodes):
                    break
                hist_desc = []
                for item in archive:
                    passed = [nid for nid, s in item['scores'].items() if s >= dag.nodes[nid].threshold]
                    failed = [requirement_statement(dag.nodes[nid]) for nid, s in item['scores'].items() if s < dag.nodes[nid].threshold]
                    hist_desc.append({'attempt_id': item['id'], 'gm': round(item['gm'], 4), 'passed': passed, 'failed': failed})
                prompt_text = f'{GENREVISIT_PROMPT}\nTask: {user_prompt}\nHistory: {json.dumps(hist_desc)}'
                raw_reply = generate_mllm_text(processor, mllm, [item['image'] for item in archive[-2:]], prompt_text, args.device)
                src_id = best_item['id']
                refined_inst = user_prompt
                m = re.search(r'\{.*\}', raw_reply, re.S)
                if m:
                    try:
                        parsed = json.loads(m.group(0))
                        src_id = int(parsed.get('source_attempt_id', src_id))
                        refined_inst = str(parsed.get('instruction') or user_prompt)
                    except (ValueError, TypeError):
                        pass
                src_img = next((it['image'] for it in archive if it['id'] == src_id), best_item['image'])
                act = ControllerAction('modify_attr', CANVAS, refined_inst)
                crop = editor.render(src_img, act, seed=turn_seed + step_i)
                cand = paste_window_strictly_local(src_img, crop, CANVAS, 0)
                gen_calls += 1
                sc = score_all_nodes(verifier, cand, dag)
                ver_calls += len(dag.nodes)
                archive.append({'id': len(archive), 'image': cand, 'scores': sc, 'gm': geometric_mean_score(dag, sc)})
            image = max(archive, key=lambda x: x['gm'])['image']

        elif args.baseline == 'edit_thinker':
            processor, mllm = mllm_bundle
            orig_turn_img = image
            rewritten = user_prompt
            for step_i in range(min(3, steps_per_turn)):
                act = ControllerAction('modify_attr', CANVAS, rewritten)
                crop = editor.render(image, act, seed=turn_seed + step_i)
                image = paste_window_strictly_local(image, crop, CANVAS, 0)
                gen_calls += 1
                prompt_text = EDIT_THINKER_PROMPT.format(original_instruction=user_prompt, rewritten_prompt=rewritten)
                reply = generate_mllm_text(processor, mllm, [orig_turn_img, image], prompt_text, args.device)
                score_m = re.search(r'<score>\s*(\{.*?\})\s*</score>', reply, re.S)
                ans_m = re.search(r'<answer>\s*(.*?)\s*</answer>', reply, re.S)
                if score_m:
                    try:
                        sc_obj = json.loads(score_m.group(1))
                        if float(sc_obj.get('semantic', 0)) >= 8 and float(sc_obj.get('quality', 0)) >= 8:
                            break
                    except (ValueError, TypeError):
                        pass
                if ans_m and ans_m.group(1).strip():
                    rewritten = ans_m.group(1).strip()

        elif args.baseline == 'replan':
            processor, mllm = mllm_bundle
            reply = generate_mllm_text(processor, mllm, [image], REPLAN_PROMPT + user_prompt, args.device)
            reg_m = re.search(r'<region>\s*(\[.*?\])\s*</region>', reply, re.S)
            gen_m = re.search(r'<gen_image>\s*(.*?)\s*</gen_image>', reply, re.S)
            regions = []
            if reg_m:
                try:
                    regions = json.loads(reg_m.group(1))
                except ValueError:
                    regions = []
            h, w = image.shape[-2:]
            executed_any = False
            for r_i, reg in enumerate(regions[:3]):
                hint = str(reg.get('hint', '')).strip()
                if not hint or 'unchanged' in hint.lower():
                    continue
                box = normalize_box_2d(reg.get('bbox_2d'), h, w)
                act = ControllerAction('modify_attr', box, hint)
                crop = editor.render(image, act, seed=turn_seed + r_i)
                image = paste_window_strictly_local(image, crop, box, 12 if box != CANVAS else 0)
                gen_calls += 1
                executed_any = True
            if not executed_any:
                global_inst = gen_m.group(1).strip() if gen_m and gen_m.group(1).strip() else user_prompt
                if 'unchanged' in global_inst.lower():
                    global_inst = user_prompt
                act = ControllerAction('modify_attr', CANVAS, global_inst)
                crop = editor.render(image, act, seed=turn_seed)
                image = paste_window_strictly_local(image, crop, CANVAS, 0)
                gen_calls += 1

    return image, dag, gen_calls, ver_calls


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', required=True, choices=(
        'direct_flux', 'mt_opsd', 'best_of_n', 'vision_director', 'genrevisit', 'edit_thinker', 'replan',
    ))
    parser.add_argument('--data', default='/root/autodl-tmp/ReDAG/data/final/eval_main.jsonl')
    parser.add_argument('--prefix', default='exp1')
    parser.add_argument('--shards', nargs='+', type=int, default=[0, 1, 2, 10])
    parser.add_argument('--num_shards', type=int, default=12)
    parser.add_argument('--editor_model', default='/root/autodl-tmp/models/black-forest-labs/FLUX.2-klein-base-9B')
    parser.add_argument('--editor_adapter', default=None)
    parser.add_argument('--verifier_model', default='/root/autodl-tmp/models/Qwen/Qwen3-VL-8B-Instruct')
    parser.add_argument('--mllm_model', default=None)
    parser.add_argument('--num_denoise_steps', type=int, default=28)
    parser.add_argument('--max_steps', type=int, default=3)
    parser.add_argument('--candidates_per_turn', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()

    if args.baseline == 'mt_opsd' and not args.editor_adapter:
        args.editor_adapter = '/root/autodl-tmp/models/MT-OPSD/flux2-klein-base-9b'

    records = read_records(args.data)
    editor = FluxEditor.from_pretrained(args.editor_model, args.device, torch.bfloat16, args.editor_adapter, False)
    editor.num_steps = args.num_denoise_steps

    verifier = None
    if args.baseline in ('best_of_n', 'vision_director', 'genrevisit'):
        verifier = Verifier(load_vlm_scorer(args.verifier_model, args.device))

    mllm_bundle = None
    if args.baseline in ('genrevisit', 'edit_thinker', 'replan'):
        default_mllm = {
            'genrevisit': '/root/autodl-tmp/models/GenRevisit/GenRevisit-RL',
            'edit_thinker': '/root/autodl-tmp/models/appletea2333/EditThinker-8B',
            'replan': '/root/autodl-tmp/models/TainU/RePlan-Qwen2.5-VL-7B',
        }[args.baseline]
        mllm_id = args.mllm_model or default_mllm
        if args.baseline == 'genrevisit' and verifier is not None and mllm_id == args.verifier_model:
            pass
        mllm_bundle = load_mllm(mllm_id, args.device)

    root = Path('/root/autodl-tmp/ReDAG/runs/eval')
    root.mkdir(parents=True, exist_ok=True)
    for shard in args.shards:
        out = root / f'{args.prefix}_{args.baseline}.s{shard}.cases.jsonl'
        folder = root / f'{args.prefix}_{args.baseline}.s{shard}_images'
        folder.mkdir(parents=True, exist_ok=True)
        done = set()
        if out.exists():
            for line in out.read_text(encoding='utf-8').splitlines()[1:]:
                try:
                    done.add(json.loads(line)['key'])
                except ValueError:
                    pass
        else:
            out.write_text(json.dumps({
                'fingerprint': {
                    'baseline': args.baseline,
                    'adapter': args.editor_adapter,
                    'steps': args.num_denoise_steps,
                    'data': str(args.data),
                }
            }) + '\n', encoding='utf-8')

        indices = [i for i in range(len(records)) if i % args.num_shards == shard]
        for index in indices:
            key = f'external|{args.baseline}|{index}|0'
            if key in done:
                continue
            record = records[index]
            image, dag, gen_calls, ver_calls = run_single_session(args, record, editor, verifier, mllm_bundle)
            identifier = f'case_{args.baseline}_s{shard}_{index:04d}'
            view = export_view(image, dag, folder, identifier + '_final')
            case = {
                'id': identifier,
                'group': f'external/{args.baseline}/{len(dag.nodes)}',
                'views': {view['id']: view},
                'final': view['id'],
                'steps': [],
                'execution': {
                    'generation_calls': gen_calls,
                    'verifier_calls': ver_calls,
                    'stop_reason': 'turns_done',
                },
            }
            with out.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps({'key': key, 'case': case, 'k': len(dag.nodes)}) + '\n')
            print(f'[{args.baseline}] {identifier} gen={gen_calls} ver={ver_calls}', flush=True)


if __name__ == '__main__':
    main()
