import argparse
import json
import re
from collections import Counter

COLORS = ['blue', 'red', 'green', 'yellow', 'purple', 'orange', 'pink']
COLOR_WORDS = set(COLORS) | {'white', 'black', 'grey', 'gray', 'brown', 'silver', 'gold', 'golden', 'teal', 'beige', 'navy', 'cyan', 'magenta',
                             'turquoise', 'maroon', 'violet', 'tan', 'cream', 'ivory', 'bronze', 'colored', 'coloured', 'color', 'colour',
                             'colorful', 'multicolored', 'dark', 'bright', 'pale'}
MATERIALS = ['wooden', 'metallic', 'ceramic', 'glass']
MATERIAL_WORDS = set(MATERIALS) | {'wood', 'metal', 'steel', 'iron', 'brass', 'copper', 'plastic', 'leather', 'fabric', 'cloth', 'stone',
                                   'marble', 'brick', 'concrete', 'porcelain', 'clay', 'bamboo', 'wicker', 'rattan', 'rubber', 'paper', 'cardboard'}
SIZES = ['large', 'small']
SIZE_WORDS = set(SIZES) | {'big', 'tiny', 'huge', 'enormous', 'giant', 'massive', 'little', 'miniature', 'compact', 'tall', 'short',
                           'wide', 'narrow', 'oversized', 'medium', 'sized', 'size'}
POSITIONS = [('left side', 'on the left side of the scene', 'move the {phrase} to the left side of the scene'),
             ('right side', 'on the right side of the scene', 'move the {phrase} to the right side of the scene'),
             ('foreground', 'in the foreground of the scene', 'move the {phrase} closer to the foreground of the scene')]
POSITION_WORDS = {'left', 'right', 'center', 'centre', 'middle', 'foreground', 'background', 'top', 'bottom', 'corner', 'side', 'front', 'back'}

PERSON_WORDS = {'person', 'people', 'man', 'men', 'woman', 'women', 'girl', 'girls', 'boy', 'boys', 'child', 'children', 'kid', 'kids', 'player',
                'players', 'lady', 'guy', 'baby', 'skier', 'surfer', 'rider', 'batter', 'catcher', 'pitcher', 'officer', 'worker', 'chef',
                'skateboarder', 'snowboarder', 'cyclist', 'biker', 'tennis', 'athlete', 'human', 'adult', 'toddler', 'teenager', 'couple',
                'crowd', 'audience', 'spectator', 'spectators', 'he', 'she', 'his', 'her', 'someone', 'face', 'hand', 'hands'}
YES_NO = ('is ', 'are ', 'does ', 'do ', 'has ', 'have ', 'can ', 'was ', 'were ', 'will ')
EDGE_KEYS = ('comp_edges', 'supp_edges', 'int_edges')


def words(text):
    return set(re.findall(r"[a-z']+", text.lower()))


def strip_vocabulary(text, vocab):
    tokens = re.findall(r"[\w'-]+|[^\w\s]", text)
    removed = [token.lower().strip('-') in vocab for token in tokens]
    kept = []
    for index, token in enumerate(tokens):
        if removed[index]:
            continue
        if token.lower() in ('and', 'or', ',', '/'):
            beside = (index > 0 and removed[index - 1]) or (index + 1 < len(tokens) and removed[index + 1])
            if beside or not kept or kept[-1].lower() in ('a', 'an', 'the', 'is', 'are', 'there'):
                continue
        kept.append(token)
    text = ''
    for token in kept:
        text += token if not text or re.match(r'[^\w]', token) else ' ' + token
    return text.strip()


def strip_colors(text):
    return strip_vocabulary(text, COLOR_WORDS)


def prune(nodes, edges):
    changed = True
    while changed:
        changed = False
        for nid in list(nodes):
            if any(parent not in nodes for parent in nodes[nid].get('depends_on', [])):
                del nodes[nid]
                changed = True
    for key in EDGE_KEYS:
        edges[key] = [edge for edge in edges[key] if edge[0] in nodes and edge[1] in nodes]


def dependents(nodes, edges, entity):
    return ({n['node_id'] for n in nodes.values() if entity in n.get('depends_on', [])}
            | {child for parent, child in edges['comp_edges'] + edges['supp_edges'] if parent == entity})


def extract_initial_candidates(record):
    nodes, edges = {}, {key: [] for key in EDGE_KEYS}
    for turn in record['turns']:
        for node in turn['graph']['nodes']:
            if re.match(r'^i\d+$', node['node_id']) and node['question'].strip().lower().startswith(YES_NO):
                nodes[node['node_id']] = json.loads(json.dumps(node))
        for key in EDGE_KEYS:
            edges[key] += [list(edge) for edge in turn['graph'].get(key, [])]
    prune(nodes, edges)
    candidates = [n for n in nodes.values() if n['node_type'] == 'entity' and not words(n['entity_phrase'] + ' ' + n['question']) & PERSON_WORDS]
    candidates = [n for n in sorted(candidates, key=lambda n: -len(dependents(nodes, edges, n['node_id']))) if len(dependents(nodes, edges, n['node_id'])) >= 2]
    return nodes, edges, candidates


def build_variant_record(index, record, candidate_id, kind, stats, suffix=''):
    base_nodes, base_edges, _ = extract_initial_candidates(record)
    if candidate_id not in base_nodes:
        return None
    nodes = json.loads(json.dumps(base_nodes))
    edges = json.loads(json.dumps(base_edges))
    entity = nodes[candidate_id]

    if kind == 'recolor':
        vocab = COLOR_WORDS
    elif kind == 'material':
        vocab = MATERIAL_WORDS
    elif kind == 'resize':
        vocab = SIZE_WORDS
    elif kind == 'relocate':
        vocab = POSITION_WORDS
    else:
        raise ValueError(f'Unknown stress kind: {kind}')

    old_phrase, phrase = entity['entity_phrase'].strip(), strip_vocabulary(entity['entity_phrase'].strip(), vocab)
    if not phrase or words(phrase) <= vocab:
        stats['phrase_lost'] += 1
        return None
    if old_phrase.lower() != phrase.lower():
        for node in nodes.values():
            node['question'] = re.sub(re.escape(old_phrase), phrase, node['question'], flags=re.I)
            node['entity_phrase'] = re.sub(re.escape(old_phrase), phrase, node['entity_phrase'], flags=re.I)
    stripped_q = strip_vocabulary(entity['question'], vocab)
    if not stripped_q.lower().startswith(YES_NO):
        stats['phrase_lost'] += 1
        return None
    entity['question'], entity['entity_phrase'] = stripped_q, phrase
    related = dependents(nodes, edges, entity['node_id'])
    phrase_words = words(phrase) - {'a', 'an', 'the', 'of', 'in', 'on'}
    drop = {nid for nid in related if words(nodes[nid]['question'] + ' ' + nodes[nid]['entity_phrase']) & vocab}
    drop |= {nid for nid, n in nodes.items() if nid not in related and nid != entity['node_id'] and n['node_type'] != 'entity'
             and words(n['question']) & vocab and words(n['question']) & phrase_words}
    for nid in drop:
        del nodes[nid]
    prune(nodes, edges)
    if entity['node_id'] not in nodes or len(dependents(nodes, edges, entity['node_id'])) < 2:
        stats['too_few_dependents'] += 1
        return None

    used = set().union(*(words(n['question'] + ' ' + n['entity_phrase']) & vocab for n in nodes.values()))
    verb = 'Are' if entity['question'].lower().startswith('are ') else 'Is'

    if kind == 'recolor':
        free = [color for color in COLORS if color not in used]
        if not free:
            stats['no_free_color'] += 1
            return None
        target_val = free[index % len(free)]
        question = f'{verb} the {phrase} {target_val}?'
        prompt = f'Change the color of the {phrase} to {target_val}. Keep everything else in the image exactly as it is.'
    elif kind == 'material':
        free = [mat for mat in MATERIALS if mat not in used]
        if not free:
            stats['no_free_material'] += 1
            return None
        target_val = free[index % len(free)]
        question = f'{verb} the {phrase} {target_val}?'
        prompt = f'Replace the material of the {phrase} so that the {phrase} is {target_val}. Keep everything else in the image exactly as it is.'
    elif kind == 'resize':
        free = [sz for sz in SIZES if sz not in used]
        if not free:
            stats['no_free_size'] += 1
            return None
        target_val = free[index % len(free)]
        action_verb = 'Enlarge' if target_val == 'large' else 'Shrink'
        question = f'{verb} the {phrase} {target_val}?'
        prompt = f'{action_verb} the {phrase} so that the {phrase} is {target_val}. Keep everything else in the image exactly as it is.'
    elif kind == 'relocate':
        free = [pos for pos in POSITIONS if not (words(pos[0]) & used)]
        if not free:
            stats['no_free_position'] += 1
            return None
        label, q_tail, p_tmpl = free[index % len(free)]
        target_val = label
        question = f'{verb} the {phrase} {q_tail}?'
        prompt = f'{p_tmpl.format(phrase=phrase).capitalize()}. Keep everything else in the image exactly as it is.'

    nodes['s1'] = {'node_id': 's1', 'node_type': 'attribute', 'question': question, 'entity_phrase': phrase, 'bbox': None,
                   'weight': 1.0, 'threshold': 0.7, 'expected_count': None, 'scope': 'local', 'depends_on': [entity['node_id']]}
    edges['comp_edges'].append([entity['node_id'], 's1'])
    stats['records'] += 1
    stats[f'kind_{kind}'] += 1
    stats['nodes'] += len(nodes)
    rec_id = f'stress_{index}' if not suffix else f'stress_{index}_{suffix}'
    return {'id': rec_id, 'image': record['image'], 'k': len(nodes), 'kind': f'stress_{kind}', 'session_id': record.get('session_id'),
            'source': index, 'parent': entity['node_id'], 'turns': [{'user_prompt': prompt, 'superseded_ids': [], 'graph': {'nodes': list(nodes.values()), **edges}}]}


def recolor_record(index, record, stats):
    _, _, candidates = extract_initial_candidates(record)
    if not candidates:
        stats['no_candidate'] += 1
        return None
    return build_variant_record(index, record, candidates[0]['node_id'], 'recolor', stats)


def audit(record):
    graph = record['turns'][0]['graph']
    nodes = {n['node_id']: n for n in graph['nodes']}
    entity, target = nodes[record['parent']], nodes['s1']
    kind = record.get('kind', 'stress_recolor').replace('stress_', '')
    vocab = {'recolor': COLOR_WORDS, 'material': MATERIAL_WORDS, 'resize': SIZE_WORDS, 'relocate': POSITION_WORDS}[kind]
    target_tokens = words(target['question']) & vocab
    related = dependents(nodes, graph, record['parent']) - {'s1'}
    issues = []
    if words(entity['question'] + ' ' + entity['entity_phrase']) & (vocab | PERSON_WORDS):
        issues.append(f'parent mentions {kind} vocabulary or a person')
    issues += [f'child mentions {kind} vocabulary: {nodes[nid]["question"]}' for nid in related if words(nodes[nid]['question']) & vocab]
    issues += [f'target {kind} token reused: {n["question"]}' for nid, n in nodes.items() if nid != 's1' and words(n['question']) & target_tokens]
    issues += [f'not a yes/no question: {n["question"]}' for n in nodes.values() if not n['question'].strip().lower().startswith(YES_NO)]
    return entity['question'], target['question'], [nodes[nid]['question'] for nid in sorted(related)], issues


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/final/eval_main.jsonl')
    parser.add_argument('--out', default='data/final/eval_stress.jsonl')
    parser.add_argument('--kinds', nargs='+', choices=('recolor', 'material', 'resize', 'relocate'), default=['recolor'])
    parser.add_argument('--all_entities', action='store_true')
    args = parser.parse_args()
    stats, records = Counter(), []
    for index, line in enumerate(open(args.data, encoding='utf-8')):
        raw = json.loads(line)
        if args.kinds == ['recolor'] and not args.all_entities:
            record = recolor_record(index, raw, stats)
            if record is not None:
                records.append(record)
        else:
            _, _, candidates = extract_initial_candidates(raw)
            if not candidates:
                stats['no_candidate'] += 1
                continue
            chosen = candidates if args.all_entities else candidates[:1]
            for c_idx, cand in enumerate(chosen):
                for kind in args.kinds:
                    suffix = f'{kind}_e{c_idx}' if args.all_entities or len(args.kinds) > 1 else ''
                    rec = build_variant_record(index + c_idx, raw, cand['node_id'], kind, stats, suffix=suffix)
                    if rec is not None:
                        records.append(rec)
    with open(args.out, 'w', encoding='utf-8') as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')
    failed = 0
    for record in records:
        parent, target, children, issues = audit(record)
        failed += bool(issues)
    print(dict(stats), f'total={len(records)} audit failures={failed}')


if __name__ == '__main__':
    main()
