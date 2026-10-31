import argparse
import json
import re
from collections import Counter

COLORS = ['blue', 'red', 'green', 'yellow', 'purple', 'orange', 'pink']
COLOR_WORDS = set(COLORS) | {'white', 'black', 'grey', 'gray', 'brown', 'silver', 'gold', 'golden', 'teal', 'beige', 'navy', 'cyan', 'magenta',
                             'turquoise', 'maroon', 'violet', 'tan', 'cream', 'ivory', 'bronze', 'colored', 'coloured', 'color', 'colour',
                             'colorful', 'multicolored', 'dark', 'bright', 'pale'}
PERSON_WORDS = {'person', 'people', 'man', 'men', 'woman', 'women', 'girl', 'girls', 'boy', 'boys', 'child', 'children', 'kid', 'kids', 'player',
                'players', 'lady', 'guy', 'baby', 'skier', 'surfer', 'rider', 'batter', 'catcher', 'pitcher', 'officer', 'worker', 'chef',
                'skateboarder', 'snowboarder', 'cyclist', 'biker', 'tennis', 'athlete', 'human', 'adult', 'toddler', 'teenager', 'couple',
                'crowd', 'audience', 'spectator', 'spectators', 'he', 'she', 'his', 'her', 'someone', 'face', 'hand', 'hands'}
YES_NO = ('is ', 'are ', 'does ', 'do ', 'has ', 'have ', 'can ', 'was ', 'were ', 'will ')
EDGE_KEYS = ('comp_edges', 'supp_edges', 'int_edges')


def words(text):
    return set(re.findall(r"[a-z']+", text.lower()))


def strip_colors(text):
    tokens = re.findall(r"[\w'-]+|[^\w\s]", text)
    removed = [token.lower().strip('-') in COLOR_WORDS for token in tokens]
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


def recolor_record(index, record, stats):
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
    if not candidates:
        stats['no_candidate'] += 1
        return None
    entity = candidates[0]
    old_phrase, phrase = entity['entity_phrase'].strip(), strip_colors(entity['entity_phrase'].strip())
    if not phrase or words(phrase) <= COLOR_WORDS:
        stats['phrase_lost'] += 1
        return None
    if old_phrase.lower() != phrase.lower():
        for node in nodes.values():
            node['question'] = re.sub(re.escape(old_phrase), phrase, node['question'], flags=re.I)
            node['entity_phrase'] = re.sub(re.escape(old_phrase), phrase, node['entity_phrase'], flags=re.I)
    entity['question'], entity['entity_phrase'] = strip_colors(entity['question']), phrase
    related = dependents(nodes, edges, entity['node_id'])
    phrase_words = words(phrase) - {'a', 'an', 'the', 'of', 'in', 'on'}
    drop = {nid for nid in related if words(nodes[nid]['question'] + ' ' + nodes[nid]['entity_phrase']) & COLOR_WORDS}
    drop |= {nid for nid, n in nodes.items() if nid not in related and nid != entity['node_id'] and n['node_type'] != 'entity'
             and words(n['question']) & COLOR_WORDS and words(n['question']) & phrase_words}
    for nid in drop:
        del nodes[nid]
    prune(nodes, edges)
    if entity['node_id'] not in nodes or len(dependents(nodes, edges, entity['node_id'])) < 2:
        stats['too_few_dependents'] += 1
        return None
    used = set().union(*(words(n['question'] + ' ' + n['entity_phrase']) & COLOR_WORDS for n in nodes.values()))
    free = [color for color in COLORS if color not in used]
    if not free:
        stats['no_free_color'] += 1
        return None
    color = free[index % len(free)]
    verb = 'Are' if entity['question'].lower().startswith('are ') else 'Is'
    nodes['s1'] = {'node_id': 's1', 'node_type': 'attribute', 'question': f'{verb} the {phrase} {color}?', 'entity_phrase': phrase, 'bbox': None,
                   'weight': 1.0, 'threshold': 0.7, 'expected_count': None, 'scope': 'local', 'depends_on': [entity['node_id']]}
    edges['comp_edges'].append([entity['node_id'], 's1'])
    stats['records'] += 1
    stats['nodes'] += len(nodes)
    prompt = f'Change the color of the {phrase} to {color}. Keep everything else in the image exactly as it is.'
    return {'id': f'stress_{index}', 'image': record['image'], 'k': len(nodes), 'kind': 'stress_recolor', 'session_id': record.get('session_id'),
            'source': index, 'parent': entity['node_id'], 'turns': [{'user_prompt': prompt, 'superseded_ids': [], 'graph': {'nodes': list(nodes.values()), **edges}}]}


def audit(record):
    graph = record['turns'][0]['graph']
    nodes = {n['node_id']: n for n in graph['nodes']}
    entity, target = nodes[record['parent']], nodes['s1']
    color = words(target['question']) & COLOR_WORDS
    related = dependents(nodes, graph, record['parent']) - {'s1'}
    issues = []
    if words(entity['question'] + ' ' + entity['entity_phrase']) & (COLOR_WORDS | PERSON_WORDS):
        issues.append('parent mentions a color or a person')
    issues += [f'child mentions a color: {nodes[nid]["question"]}' for nid in related if words(nodes[nid]['question']) & COLOR_WORDS]
    issues += [f'target color reused: {n["question"]}' for nid, n in nodes.items() if nid != 's1' and words(n['question']) & color]
    issues += [f'not a yes/no question: {n["question"]}' for n in nodes.values() if not n['question'].strip().lower().startswith(YES_NO)]
    return entity['question'], target['question'], [nodes[nid]['question'] for nid in sorted(related)], issues


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/final/eval_main.jsonl')
    parser.add_argument('--out', default='data/final/eval_stress.jsonl')
    args = parser.parse_args()
    stats, records = Counter(), []
    for index, line in enumerate(open(args.data, encoding='utf-8')):
        record = recolor_record(index, json.loads(line), stats)
        if record is not None:
            records.append(record)
    with open(args.out, 'w', encoding='utf-8') as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')
    failed = 0
    for record in records:
        parent, target, children, issues = audit(record)
        failed += bool(issues)
        print(f"{record['id']} | {parent} | {target} | {' / '.join(children)} | {'; '.join(issues) or 'ok'}")
    print(dict(stats), f'audit failures {failed}')


if __name__ == '__main__':
    main()
