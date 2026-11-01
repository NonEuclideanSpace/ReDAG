import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import io
import json
from pathlib import Path
import re
import struct
import sys
import urllib.request
import zipfile
import zlib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from redag.models.planner import dag_from_dict, preserve_previous_conditions

COLOR_WORDS = {
    'red', 'blue', 'green', 'yellow', 'black', 'white', 'gray', 'grey', 'brown',
    'orange', 'pink', 'purple', 'beige', 'teal', 'silver', 'gold', 'golden',
    'blonde', 'dark', 'light', 'floral', 'colorful',
}
MATERIAL_WORDS = {
    'wooden', 'wood', 'metal', 'metallic', 'plastic', 'glass', 'ceramic', 'fabric',
    'cotton', 'leather', 'paper', 'stone', 'marble', 'rubber', 'plush', 'wool',
    'woven', 'latex', 'denim', 'silk', 'velvet', 'brick', 'concrete',
}
REL_TEXT = {
    'left': 'to the left of',
    'right': 'to the right of',
    'above': 'above',
    'below': 'below',
}


def clean_phrase(text):
    text = text.strip().strip('"').strip("'").strip('.').strip()
    return re.sub(r'\s+', ' ', text)


def strip_adjectives(phrase, drop_words):
    tokens = clean_phrase(phrase).split()
    kept = [t for t in tokens if t.lower() not in drop_words and t.lower() != 'and']
    return ' '.join(kept) if kept else clean_phrase(phrase)


def bracket_args(fmt):
    return [clean_phrase(m) for m in re.findall(r'\[([^\]]+)\]', fmt)]


def make_node(node_id, node_type, question, entity_phrase, depends_on=None, scope='local', weight=1.0, bbox=None):
    return {
        'node_id': node_id,
        'node_type': node_type,
        'question': question,
        'threshold': 0.7,
        'weight': weight,
        'scope': scope,
        'entity_phrase': clean_phrase(entity_phrase),
        'depends_on': list(depends_on or []),
        'bbox': bbox,
    }


def parse_turn_to_graph(turn_idx, turn_data, existing_entities, unchanged_objects, unchanged_boxes):
    tt = turn_data['task_type']
    fmt = turn_data.get('format_instructions', '')
    inst = clean_phrase(turn_data.get('instruction', ''))
    args = bracket_args(fmt)
    nodes = []
    comp_edges = []
    superseded_ids = []

    if turn_idx == 1:
        for u_idx, u_obj in enumerate(unchanged_objects[:2]):
            u_clean = clean_phrase(u_obj)
            nid = f'u{u_idx + 1}'
            bbox = unchanged_boxes.get(u_clean)
            nodes.append(make_node(
                nid, 'entity', f'Is there a {u_clean} in the image?',
                u_clean, depends_on=[], scope='local', weight=1.0, bbox=bbox,
            ))
            existing_entities[u_clean.lower()] = [nid]

    prefix = f't{turn_idx}'

    def ensure_entity(phrase, suffix='e1'):
        p_clean = clean_phrase(phrase)
        key = p_clean.lower()
        if key in existing_entities and existing_entities[key]:
            return existing_entities[key][0], p_clean
        nid = f'{prefix}_{suffix}'
        nodes.append(make_node(nid, 'entity', f'Is there a {p_clean} in the image?', p_clean))
        existing_entities[key] = [nid]
        return nid, p_clean

    def retire_entity(phrase):
        p_clean = clean_phrase(phrase).lower()
        for key, nids in list(existing_entities.items()):
            if key == p_clean or p_clean in key or key in p_clean:
                for nid in nids:
                    if turn_idx == 1:
                        nodes[:] = [n for n in nodes if n['node_id'] != nid]
                    elif nid not in superseded_ids:
                        superseded_ids.append(nid)
                existing_entities.pop(key, None)

    if tt == 'color_alter' and len(args) >= 2:
        raw_obj, new_color = args[0], args[1]
        retire_entity(raw_obj)
        base_obj = strip_adjectives(raw_obj, COLOR_WORDS)
        e_id, base_obj = ensure_entity(base_obj, 'e1')
        a_id = f'{prefix}_a1'
        nodes.append(make_node(
            a_id, 'attribute',
            f'If the {base_obj} is present, is the {base_obj} {new_color}?',
            base_obj, depends_on=[e_id], scope='local',
        ))
        comp_edges.append([e_id, a_id])
        existing_entities[base_obj.lower()].append(a_id)
        existing_entities[f'{new_color} {base_obj}'.lower()] = [e_id, a_id]

    elif tt == 'material_alter' and len(args) >= 2:
        raw_obj, new_mat = args[0], args[1]
        retire_entity(raw_obj)
        base_obj = strip_adjectives(raw_obj, MATERIAL_WORDS)
        e_id, base_obj = ensure_entity(base_obj, 'e1')
        a_id = f'{prefix}_a1'
        nodes.append(make_node(
            a_id, 'attribute',
            f'If the {base_obj} is present, is the {base_obj} made of {new_mat}?',
            base_obj, depends_on=[e_id], scope='local',
        ))
        comp_edges.append([e_id, a_id])
        existing_entities[base_obj.lower()].append(a_id)
        existing_entities[f'{new_mat} {base_obj}'.lower()] = [e_id, a_id]

    elif tt in ('subject_add', 'position_change') and len(args) >= 1:
        subj = args[0]
        if tt == 'position_change':
            retire_entity(subj)
        e1_id, subj_clean = ensure_entity(subj, 'e1')
        if len(args) >= 3:
            rel_raw, ref_obj = args[1].lower(), args[2]
            rel_phrase = REL_TEXT.get(rel_raw, rel_raw)
            e2_id, ref_clean = ensure_entity(ref_obj, 'e2')
            if e2_id == e1_id:
                e2_id = f'{prefix}_e2'
                nodes.append(make_node(e2_id, 'entity', f'Is there a {ref_clean} in the image?', ref_clean))
            r_id = f'{prefix}_r1'
            nodes.append(make_node(
                r_id, 'relation',
                f'If the {subj_clean} and the {ref_clean} are present, is the {subj_clean} {rel_phrase} the {ref_clean}?',
                subj_clean, depends_on=[e1_id, e2_id], scope='local',
            ))
            comp_edges.extend([[e1_id, r_id], [e2_id, r_id]])
            existing_entities[subj_clean.lower()].append(r_id)

    elif tt == 'count_change' and len(args) >= 2:
        subj, count_str = args[0], args[1]
        retire_entity(subj)
        e_id, subj_clean = ensure_entity(subj, 'e1')
        c_id = f'{prefix}_c1'
        nodes.append(make_node(
            c_id, 'count',
            f'Are there exactly {count_str} {subj_clean}s in the image?',
            subj_clean, depends_on=[e_id], scope='global',
        ))
        comp_edges.append([e_id, c_id])
        existing_entities[subj_clean.lower()].append(c_id)

    elif tt == 'subject_remove' and len(args) >= 1:
        subj = args[0]
        retire_entity(subj)
        p_id = f'{prefix}_p1'
        nodes.append(make_node(
            p_id, 'prohibition',
            f'Is there no {subj} in the image?',
            subj, depends_on=[], scope='global',
        ))
        existing_entities[f'no_{subj.lower()}'] = [p_id]

    elif tt == 'subject_replace' and len(args) >= 2:
        old_obj, new_obj = args[0], args[1]
        retire_entity(old_obj)
        e_id, new_clean = ensure_entity(new_obj, 'e1')
        p_id = f'{prefix}_p1'
        nodes.append(make_node(
            p_id, 'prohibition',
            f'Is there no {old_obj} in the image?',
            old_obj, depends_on=[], scope='global',
        ))
        existing_entities[new_clean.lower()] = [e_id]
    else:
        subj = args[0] if args else inst
        ensure_entity(subj, 'e1')

    graph = {
        'nodes': nodes,
        'comp_edges': comp_edges,
        'supp_edges': [],
        'int_edges': [],
    }
    return {
        'turn_idx': turn_idx - 1,
        'user_prompt': inst,
        'superseded_ids': superseded_ids,
        'graph': graph,
    }


def fetch_zip_entries(zip_url, target_sids):
    req = urllib.request.Request(zip_url, method='HEAD')
    resp = urllib.request.urlopen(req, timeout=30)
    final_url = resp.geturl()
    size = int(resp.headers.get('Content-Length'))
    tail_size = min(size, 1048576)
    tail = urllib.request.urlopen(
        urllib.request.Request(final_url, headers={'Range': f'bytes={size - tail_size}-{size - 1}'}),
        timeout=30,
    ).read()
    idx = 0
    matched = {}
    pat = re.compile(r'(?:^|/)(\d+)_input_raw_turn_?3\.(?:png|jpg|jpeg)$', re.I)
    while True:
        idx = tail.find(b'PK\x01\x02', idx)
        if idx == -1:
            break
        comp_method = struct.unpack('<H', tail[idx + 10:idx + 12])[0]
        csz = struct.unpack('<I', tail[idx + 20:idx + 24])[0]
        usz = struct.unpack('<I', tail[idx + 24:idx + 28])[0]
        nlen = struct.unpack('<H', tail[idx + 28:idx + 30])[0]
        elen = struct.unpack('<H', tail[idx + 30:idx + 32])[0]
        clen = struct.unpack('<H', tail[idx + 32:idx + 34])[0]
        off = struct.unpack('<I', tail[idx + 42:idx + 46])[0]
        name = tail[idx + 46:idx + 46 + nlen].decode('utf-8', errors='ignore')
        m = pat.search(name)
        if m and ('multipass' in name or 'singlepass' not in name):
            sid = int(m.group(1))
            if sid in target_sids:
                matched[sid] = (comp_method, csz, usz, off, name)
        idx += 46 + nlen + elen + clen

    def download_one(item):
        sid, (comp_method, csz, usz, off, name) = item
        r = urllib.request.Request(final_url, headers={'Range': f'bytes={off}-{off + csz + 256}'})
        chunk = urllib.request.urlopen(r, timeout=30).read()
        lnlen = struct.unpack('<H', chunk[26:28])[0]
        lelen = struct.unpack('<H', chunk[28:30])[0]
        payload = chunk[30 + lnlen + lelen:30 + lnlen + lelen + csz]
        if comp_method == 0:
            return sid, payload
        return sid, zlib.decompress(payload, -15)

    results = {}
    with ThreadPoolExecutor(max_workers=12) as pool:
        for sid, data in pool.map(download_one, matched.items()):
            results[sid] = data
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--edival_dir', default='/root/autodl-tmp/edival')
    parser.add_argument('--repo_root', default='/root/autodl-tmp/ReDAG')
    parser.add_argument('--extract_models', action='store_true')
    args = parser.parse_args()

    edival_dir = Path(args.edival_dir)
    repo_root = Path(args.repo_root)
    zf_eval = zipfile.ZipFile(edival_dir / 'eval_results.zip')
    zf_in = zipfile.ZipFile(edival_dir / 'input_images_resize_512.zip')

    names = zf_eval.namelist()
    target_models = [
        'GPT_image_1.5', 'GPT_image_1', 'Nano_Banana2', 'Nano_Banana',
        'Seedream4', 'FLUX.2-max', 'FLUX.1-Kontext-max', 'FLUX.1-Kontext-dev',
        'Qwen-Image-Edit', 'StepEdit', 'OmniGen',
    ]
    by_model = defaultdict(dict)
    for m in target_models:
        prefix = f'eval_results_agg/{m}/multipass/'
        for n in names:
            if n.startswith(prefix) and n.endswith('_input_raw.json'):
                sid = int(n.split('/')[-1].split('_')[0])
                by_model[m][sid] = n

    common_sids = sorted(set.intersection(*(set(by_model[m].keys()) for m in target_models)))
    ranked = []
    for sid in common_sids:
        d = json.loads(zf_eval.read(by_model['Qwen-Image-Edit'][sid]))
        types = [d[t]['task_type'] for t in ('1', '2', '3') if t in d]
        if len(types) != 3 or 'text_change' in types or 'background_change' in types:
            continue
        unch = len(d['1']['meta'].get('unchanged_objects', []))
        ranked.append((-unch, sid))
    ranked.sort()
    selected_100 = [sid for _, sid in ranked[:100]]

    img_dir = repo_root / 'data/final/edival100_images'
    img_dir.mkdir(parents=True, exist_ok=True)
    in_names = {}
    for n in zf_in.namelist():
        base = n.split('/')[-1]
        if '_input_raw.' in base:
            try:
                in_names[int(base.split('_')[0])] = n
            except ValueError:
                pass

    records = []
    for idx, sid in enumerate(selected_100):
        zip_entry = in_names[sid]
        ext = Path(zip_entry).suffix
        out_img = img_dir / f'{sid}_input_raw{ext}'
        if not out_img.exists():
            out_img.write_bytes(zf_in.read(zip_entry))

        d = json.loads(zf_eval.read(by_model['Qwen-Image-Edit'][sid]))
        unch_objs = d['1']['meta'].get('unchanged_objects', [])
        unch_boxes = {}

        existing_entities = {}
        turns = []
        cumulative_dag = None
        for t_idx in (1, 2, 3):
            turn_dict = parse_turn_to_graph(t_idx, d[str(t_idx)], existing_entities, unch_objs, unch_boxes)
            cumulative_dag = preserve_previous_conditions(turn_dict['graph'], cumulative_dag, turn_dict['superseded_ids'])
            turns.append(turn_dict)

        record = {
            'id': f'edival_{sid:04d}',
            'edival_sid': sid,
            'source': 'edival',
            'image': str(out_img),
            'turns': turns,
            'graph': turns[-1]['graph'],
        }
        records.append(record)

    out_jsonl = repo_root / 'data/final/eval_edival100.jsonl'
    with out_jsonl.open('w', encoding='utf-8') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f'Wrote {len(records)} validated 3-turn EdiVal records to {out_jsonl}')

    # Save official EdiVal metrics for the 100 selected sessions
    official_summary = {}
    for m in target_models:
        t1_if, t2_if, t3_if, t3_l1, t3_dino = [], [], [], [], []
        for sid in selected_100:
            d = json.loads(zf_eval.read(by_model[m][sid]))
            if1 = int(bool(d['1'].get('instruction_following', 0)))
            if2 = int(if1 and bool(d['2'].get('instruction_following', 0)))
            if3 = int(if2 and bool(d['3'].get('instruction_following', 0)))
            t1_if.append(if1)
            t2_if.append(if2)
            t3_if.append(if3)
            l1 = d['3'].get('object_l1_consistency_mean') or d['3'].get('background_l1_consistency') or 0.0
            dino = d['3'].get('object_dinov3_consistency_mean') or d['3'].get('background_dinov3_consistency') or 0.0
            t3_l1.append(float(l1))
            t3_dino.append(float(dino))
        avg = lambda xs: round(sum(xs) / len(xs) * 100, 2)
        official_summary[m] = {
            'n': len(selected_100),
            'T1_IF': avg(t1_if),
            'T2_CumIF': avg(t2_if),
            'T3_CumIF': avg(t3_if),
            'T3_L1': avg(t3_l1),
            'T3_DINO': avg(t3_dino),
            'T3_Overall': round((avg(t3_if) * avg(t3_l1)) ** 0.5, 2),
        }
    metrics_path = repo_root / 'runs/eval/edival100_official_metrics.json'
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps({'selected_sids': selected_100, 'models': official_summary}, indent=2))
    print(f'Wrote official metrics summary to {metrics_path}')

    if not args.extract_models:
        return

    zip_map = {
        'edival_gpt_image_1_5': 'GPT_Image_1_5_generation.zip',
        'edival_nano_banana2': 'Nano_Banana2_generation.zip',
        'edival_nano_banana': 'NanoBanana_generation.zip',
        'edival_seedream4': 'Seedream4_generation.zip',
        'edival_flux2_max': 'Flux2_max_generation.zip',
        'edival_flux_kontext_max': 'Flux_max_generation.zip',
        'edival_flux_kontext_dev': 'Flux_dev_generation.zip',
        'edival_qwen_image_edit': 'QWEN_generation.zip',
        'edival_step1x_edit': 'StepEdit_generation.zip',
    }
    target_sids = set(selected_100)
    eval_dir = repo_root / 'runs/eval'

    for model_label, zip_name in zip_map.items():
        url = f'https://hf-mirror.com/datasets/C-Tianyu/EdiVal/resolve/main/{zip_name}'
        print(f'Fetching 100 Turn-3 images for {model_label} from {zip_name}...')
        fetched = fetch_zip_entries(url, target_sids)
        folder = eval_dir / f'{model_label}.s0_images'
        folder.mkdir(parents=True, exist_ok=True)
        cases_path = eval_dir / f'{model_label}.s0.cases.jsonl'
        with cases_path.open('w', encoding='utf-8') as stream:
            stream.write(json.dumps({'fingerprint': {'baseline': model_label, 'dataset': 'eval_edival100'}}) + '\n')
            for idx, rec in enumerate(records):
                sid = rec['edival_sid']
                if sid not in fetched:
                    continue
                dag = None
                for turn in rec['turns']:
                    dag = preserve_previous_conditions(turn['graph'], dag, turn.get('superseded_ids'))
                identifier = f'case_{model_label}_s0_{idx:04d}'
                img_path = folder / f'{identifier}_final.png'
                img_path.write_bytes(fetched[sid])
                nodes_dict = {
                    nid: {
                        'question': node.question,
                        'threshold': node.threshold,
                        'weight': node.weight,
                        'scope': node.scope,
                        'depends_on': list(node.depends_on),
                        'bbox': list(node.bbox) if node.bbox is not None else None,
                    }
                    for nid, node in dag.nodes.items()
                }
                view = {'id': f'{identifier}_final', 'image': str(img_path), 'nodes': nodes_dict}
                case = {
                    'id': identifier,
                    'group': f'edival/{model_label}/{len(nodes_dict)}',
                    'views': {view['id']: view},
                    'final': view['id'],
                    'steps': [],
                    'execution': {'generation_calls': 3, 'verifier_calls': 0, 'stop_reason': 'turns_done'},
                }
                key = f'edival|{model_label}|{idx}|0'
                stream.write(json.dumps({'key': key, 'case': case, 'k': len(nodes_dict)}) + '\n')
        print(f'Wrote {len(fetched)} cases to {cases_path}')


if __name__ == '__main__':
    main()
