"""Raw sessions -> validated condition DAGs (GPU step; resumable; shardable across GPUs).

checklist mode (CoReBench / LGBench): every checklist item becomes a node with its original id and
question; the LLM only adds structure (types, entity phrases, scope, dependencies, comp/supp/int
edges) and may add entity nodes e1.. for objects that relations refer to.
instruction mode (MagicBrush): the LLM sees the starting image and every turn's instruction and
returns edit conditions t<turn>_<k> plus up to 24 inventory conditions i<k> (visible facts no
request should change -> protected conditions used to reach K = 10/20/30).

Every answer is validated with the repository's own ConditionDAG (via preserve_previous_conditions);
invalid answers are retried with the error message, then replaced by a heuristic fallback that is
flagged so build_sets.py can exclude it.  --heuristic runs the fallback only (no GPU), which is how
the downstream pipeline is smoke-tested in no-GPU mode.

  python scripts/data/annotate_graphs.py --sessions data/processed/sessions/corebench.jsonl --device cuda:0
  python scripts/data/annotate_graphs.py --sessions ... --shard 1 --num_shards 2 --device cuda:1
"""
import argparse
import json
import re
from pathlib import Path

from _common import PROC, append_jsonl, build_dag, clean_node, done_ids, iter_jsonl, log, sanitize_graph, tighten_scopes

TAG_TYPES = {'instance_pos': 'entity', 'instance_neg': 'prohibition', 'attribute_pos': 'attribute',
             'attribute_neg': 'attribute', 'relation': 'relation', 'content': 'attribute', 'layout': 'relation'}
SYSTEM = ('You convert image requirements into a condition dependency graph used by a verifier-driven local image '
          'editor. Every node is one atomic yes/no question about the final image. Output only one JSON object.')
FIELD_RULES = """Node fields: node_id, node_type, question, entity_phrase, scope, depends_on, expected_count, statement.
- node_type: entity (presence of one object), attribute (property/state/material/text of one object, depends_on its entity),
  relation (spatial or interaction between objects, depends_on every participant), count (number of instances, scope global),
  prohibition (something must be absent, scope global), background (global style, lighting, setting, scope global).
- entity_phrase: short noun phrase an open-vocabulary detector can localize (the owner object for attributes); "" for background.
- scope: local only if the condition is decided inside the boxes of the objects it depends on; uniqueness, absence, counts,
  comparisons with all objects and style are global; use unknown when unsure.
- depends_on: ids of the nodes needed to judge this node.  expected_count: integer for count nodes, otherwise null.
- statement: the requirement as one short declarative sentence.
Edges: comp_edges [entity id, dependent attribute/relation id]; supp_edges [supporting object id, supported object id]
(e.g. table -> cup on the table); int_edges [id, id] for similar or overlapping objects that an edit of one may disturb.
No dependency cycles. Return {"nodes": [...], "comp_edges": [...], "supp_edges": [...], "int_edges": [...]}
Write the JSON compactly on a single line, without indentation or line breaks: large graphs must fit the answer length limit."""
CUT_OFF = ('the answer was cut off before the JSON was complete (answer length limit); write the complete graph again as '
           'compact single-line JSON without indentation or line breaks, with short statements')


def cut_off(text):
    """Pretty-printed graphs with 25+ nodes ran past max_new_tokens; their retries need a different instruction."""
    return not str(text).rstrip().rstrip('`').rstrip().endswith('}')


def checklist_prompt(session):
    fixed = all(c.get('fixed', True) for c in session['checklist'])
    lines = [f"{c['qid']} [{','.join(c['tags']) or c.get('goal_type') or '-'}] {c['question'] if c.get('fixed', True) else c['goal']}"
             for c in session['checklist']]
    keep = ('Checklist (keep every id and question exactly):' if fixed else
            'Goals (keep every id; rewrite each goal as ONE yes/no question that is true exactly when the goal is met):')
    return (f"{'Scene description' if session['kind'] == 't2i' else 'Edit request'}:\n{session['prompt'][:3000]}\n\n{keep}\n"
            + '\n'.join(lines) + '\n\nInclude every item as a node. You may add entity nodes e1, e2, ... (question "Is there '
            '<object> in the image?") for objects that other items depend on but that have no presence item. Tag hints: '
            'instance_pos=entity, instance_neg=prohibition, attribute_*=attribute, relation=relation, add_object=entity, '
            'text=attribute of the object carrying the text, lighting/effect=background unless tied to one object.\n' + FIELD_RULES)


def decompose_prompt(session, limit=40):
    hints = '\n'.join(f'- {h[:400]}' for h in session.get('hints') or [])
    hints = f'\nReference aspects annotated for this request:\n{hints}\n' if hints else ''
    return (f"Request:\n{session['prompt'][:4000]}\n{hints}\nDecompose the request into atomic yes/no conditions about the final "
            f'image, as many as it states and at most {limit} (long requests usually need 25 or more; never invent requirements) '
            '(ids c1, c2, ...; one fact each, phrased as a question such as "Is the mug red?"). Cover every object, '
            'attribute, count, text, spatial relation and global style the request asks for. You may add entity nodes e1, e2, ... '
            'for objects that other conditions depend on.\n' + FIELD_RULES)


def instruction_prompt(session, inventory):
    turns = '\n'.join(f"turn {t['turn']}: {t['instruction']}" for t in session['turns'])
    return (f"The image is the starting canvas of a multi-turn editing session. The user's requests, in order:\n{turns}\n\n"
            'Edit conditions: for each turn t, the atomic conditions that must hold after applying request t, ids t<turn>_<k> '
            '(t1_1, t1_2, ...). A removal becomes a prohibition ("Is the image free of <object>?"); a replacement becomes the new '
            "object's presence plus a prohibition of the old one. If a later request contradicts an earlier condition, list the "
            'earlier id under "superseded": {"<turn>": [ids]}.\n'
            f'Inventory conditions: up to {inventory} more facts clearly visible in the starting image that none of the requests '
            'changes (object presence, colors, materials, counts, spatial relations, scene style), ids i1, i2, ... Skip anything a '
            'request modifies or removes. Edit conditions may depend on inventory nodes.\n'
            + FIELD_RULES.replace('"int_edges": [...]}', '"int_edges": [...], "superseded": {}}'))


def to_graph(value, session, mode):
    """LLM JSON -> (sanitized graph dict, statements, node_turn, superseded, warnings). Raises on invalid output."""
    raw_nodes = value.get('nodes', [])
    if isinstance(raw_nodes, dict):
        raw_nodes = [dict(n, node_id=k) for k, n in raw_nodes.items()]
    nodes, statements, node_turn, warnings = {}, {}, {}, []
    checklist = {c['qid']: c for c in session.get('checklist', [])}
    for raw in raw_nodes:
        if not isinstance(raw, dict) or 'node_id' not in raw:
            continue
        nid = str(raw['node_id']).strip()
        if nid in nodes:
            raise ValueError(f'duplicate node id {nid}')
        if mode == 'checklist':
            if nid in checklist:
                item, asked = checklist[nid], str(raw.get('question') or '').strip()
                usable = not item.get('fixed', True) and asked.endswith('?') and len(asked) >= 8
                raw = dict(raw, question=asked if usable else item['question'])
            elif not re.fullmatch(r'e\d+', nid):
                raise ValueError(f'unexpected node id {nid}; use checklist ids or e<k>')
            node_turn[nid] = 1
        elif mode == 'decompose':
            if not re.fullmatch(r'[ce]\d+', nid):
                raise ValueError(f'unexpected node id {nid}; use c<k> or e<k>')
            node_turn[nid] = 1
        else:
            match = re.fullmatch(r't(\d+)_\d+|i\d+', nid)
            if not match:
                raise ValueError(f'unexpected node id {nid}; use t<turn>_<k> or i<k>')
            node_turn[nid] = int(match.group(1)) if match.group(1) else 0
            if node_turn[nid] > len(session['turns']) or node_turn[nid] < 0:
                raise ValueError(f'node {nid} refers to a turn that does not exist')
        node = clean_node(dict(raw, node_id=nid))
        if not node['question']:
            raise ValueError(f'node {nid} has an empty question')
        if node['node_type'] in ('entity', 'count') and not node['entity_phrase']:
            raise ValueError(f'node {nid} ({node["node_type"]}) needs an entity_phrase')
        nodes[nid] = node
        statements[nid] = re.sub(r'\s+', ' ', str(raw.get('statement') or '')).strip()
    if mode == 'checklist':
        missing = set(checklist) - set(nodes)
        if missing:
            raise ValueError(f'missing checklist ids: {sorted(missing)}')
    elif mode == 'decompose':
        if len(nodes) < 5:
            raise ValueError('too few conditions; decompose the whole request')
    else:
        for turn in session['turns']:
            if not any(t == turn['turn'] for t in node_turn.values()):
                raise ValueError(f"turn {turn['turn']} has no edit condition")
    graph, dropped = sanitize_graph(nodes, value.get('comp_edges'), value.get('supp_edges'), value.get('int_edges'))
    if dropped:
        warnings.append(f'dropped {dropped} invalid references')
    graph = tighten_scopes(graph)
    superseded = {}
    for turn, ids in (value.get('superseded') or {}).items() if isinstance(value.get('superseded'), dict) else ():
        ids = [i for i in ids if node_turn.get(i, 99) < int(turn)] if str(turn).isdigit() else []
        if ids:
            superseded[str(int(turn))] = ids
    build_dag(graph)  # full pool must be a valid ConditionDAG
    return graph, statements, node_turn, superseded, warnings


ENTITY_Q = re.compile(r'^is there (?:an? |the |some |any )?(.+?) (?:in|visible in|present in) the image\??$', re.I)
NEG_Q = re.compile(r'^(?:is there no|are there no|is the image free of|is there not an?) (.+?)(?: in the image)?\??$', re.I)
REL_Q = re.compile(r'^if (?:both )?(?:the )?(.+?) and (?:the )?(.+?) are (?:both )?present', re.I)


def heuristic(session, mode):
    """Rule-based fallback / dry run: valid but conservative graphs (unknown scopes where unsure)."""
    nodes, statements, node_turn, comp = {}, {}, {}, []
    if mode == 'decompose':
        sentences = [s.strip() for s in re.split(r'(?<=[.!?;])\s+', session['prompt']) if len(s.split()) >= 3][:36]
        for i, text in enumerate(sentences, 1):
            nid = f'c{i}'
            nodes[nid] = clean_node({'node_id': nid, 'node_type': 'attribute', 'scope': 'unknown',
                                     'question': f'Does the image satisfy this requirement: {text.rstrip(".!;")}?'})
            statements[nid], node_turn[nid] = text, 1
        return {'nodes': list(nodes.values()), 'comp_edges': [], 'supp_edges': [], 'int_edges': []}, statements, node_turn, {}, []
    if mode == 'instruction':
        for turn in session['turns']:
            nid = f"t{turn['turn']}_1"
            nodes[nid] = clean_node({'node_id': nid, 'node_type': 'attribute', 'scope': 'unknown',
                                     'question': f"Does the image show the result of this edit: \"{turn['instruction']}\"?"})
            statements[nid], node_turn[nid] = turn['instruction'], turn['turn']
        return {'nodes': list(nodes.values()), 'comp_edges': [], 'supp_edges': [], 'int_edges': []}, statements, node_turn, {}, []
    phrase_to_id = {}
    goal_types = {'add_object': 'entity', 'text': 'attribute', 'color': 'attribute', 'lighting': 'background',
                  'effect': 'background', 'composition': 'relation'}
    for item in session['checklist']:
        tags = item['tags'] or ['']
        node_type = TAG_TYPES.get(tags[0]) or goal_types.get(item.get('goal_type'), 'attribute')
        text = item['question']
        phrase, deps, scope = '', [], 'unknown'
        if node_type == 'entity' and ENTITY_Q.match(text):
            phrase, scope = ENTITY_Q.match(text).group(1), 'local'
            phrase_to_id.setdefault(phrase.lower(), item['qid'])
        elif node_type == 'prohibition':
            phrase, scope = (NEG_Q.match(text).group(1) if NEG_Q.match(text) else ''), 'global'
        elif node_type == 'relation' and REL_Q.match(text):
            for name in REL_Q.match(text).groups():
                key = name.strip().lower()
                if key not in phrase_to_id:
                    eid = f'e{len([n for n in nodes if n.startswith("e")]) + 1}'
                    nodes[eid] = clean_node({'node_id': eid, 'node_type': 'entity', 'entity_phrase': name.strip(), 'scope': 'local',
                                             'question': f'Is there {name.strip()} in the image?'})
                    statements[eid], node_turn[eid] = f'There is {name.strip()} in the image.', 1
                    phrase_to_id[key] = eid
                deps.append(phrase_to_id[key])
            scope, phrase = 'local', REL_Q.match(text).group(1)
        nodes[item['qid']] = clean_node({'node_id': item['qid'], 'node_type': node_type, 'question': text,
                                         'entity_phrase': phrase, 'scope': scope, 'depends_on': deps})
        comp += [[d, item['qid']] for d in deps]
        statements[item['qid']], node_turn[item['qid']] = '', 1
    graph, _ = sanitize_graph(nodes, comp, [], [])
    return tighten_scopes(graph), statements, node_turn, {}, []


def load_model(args):
    import torch
    from redag.models.planner import Planner
    planner = Planner.from_pretrained(args.model, args.device, torch.bfloat16)
    tokenizer = getattr(planner.processor, 'tokenizer', planner.processor)
    tokenizer.padding_side = 'left'
    return planner


def generate(planner, conversations, images, max_new_tokens):
    import torch
    texts = [planner.processor.apply_chat_template(c, tokenize=False, add_generation_prompt=True, enable_thinking=False)
             for c in conversations]
    kwargs = {'text': texts, 'padding': True, 'return_tensors': 'pt'}
    if any(images):
        kwargs['images'] = [image for image in images if image is not None]
    inputs = planner.processor(**kwargs)
    device = next(planner.model.parameters()).device
    inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}
    with torch.no_grad():
        out = planner.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return planner.processor.batch_decode(out[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)


def messages(session, mode, inventory, feedback=None, answer=None):
    from PIL import Image
    if mode in ('checklist', 'decompose'):
        content, image = (checklist_prompt if mode == 'checklist' else decompose_prompt)(session), None
    else:
        with Image.open(session['image']) as opened:
            image = opened.convert('RGB').copy()
        image.thumbnail((1024, 1024))
        content = [{'type': 'image', 'image': image}, {'type': 'text', 'text': instruction_prompt(session, inventory)}]
    conversation = [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': content}]
    if feedback:
        conversation += [{'role': 'assistant', 'content': answer[:6000]},
                         {'role': 'user', 'content': f'That graph is invalid: {feedback}. Return the corrected complete JSON only.'}]
    return conversation, image


def record(session, mode, result, status, model, attempts, raw=None):
    graph, statements, node_turn, superseded, warnings = result
    keys = ('id', 'source', 'kind', 'split', 'prompt', 'image', 'turns', 'canvas_origin', 'canvas_prompt')
    row = {k: session.get(k) for k in keys if k in session}
    row.update(mode=mode, graph=graph, statements=statements, node_turn=node_turn, superseded=superseded,
               annotation={'status': status, 'model': model, 'attempts': attempts, 'warnings': warnings})
    if raw is not None and status == 'fallback':
        row['annotation']['last_output'] = raw[-1500:]
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sessions', nargs='+', required=True)
    parser.add_argument('--output_dir', default=str(PROC / 'graphs'))
    parser.add_argument('--model', default='Qwen/Qwen3.8-27B')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--max_new_tokens', type=int, default=4096)
    parser.add_argument('--attempts', type=int, default=3)
    parser.add_argument('--inventory', type=int, default=24)
    parser.add_argument('--limit', type=int, help='first N sessions per file (after sharding)')
    parser.add_argument('--ids', help='optional file with one session id per line to annotate (e.g. selected pools)')
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--num_shards', type=int, default=1)
    parser.add_argument('--heuristic', action='store_true', help='no model: rule-based graphs (CPU smoke test)')
    args = parser.parse_args()
    wanted = set(Path(args.ids).read_text().split()) if args.ids else None
    planner = None if args.heuristic else load_model(args)
    model_name = 'heuristic' if args.heuristic else args.model
    from collections import Counter, deque
    from redag.models.planner import parse_json_object
    from redag.progress import report
    items = []  # (session, mode, output file) still to annotate, over every session file of this shard
    for path in args.sessions:
        name = Path(path).stem + ('' if args.num_shards == 1 else f'.shard{args.shard}of{args.num_shards}')
        output = Path(args.output_dir) / f'{name}.jsonl'
        output.parent.mkdir(parents=True, exist_ok=True)
        finished = done_ids(output)
        todo = [s for i, s in enumerate(iter_jsonl(path)) if i % args.num_shards == args.shard]
        todo = [s for s in todo if s['id'] not in finished and (wanted is None or s['id'] in wanted)]
        todo = todo[:args.limit] if args.limit else todo
        log(f'{path}: {len(todo)} sessions to annotate -> {output} ({len(finished)} already done)')
        items += [(s, 'checklist' if s.get('checklist') else 'instruction' if s.get('turns') else 'decompose', output) for s in todo]
    counts, done, total = Counter(), 0, len(items)
    if args.heuristic:
        for session, mode, output in items:
            append_jsonl(output, record(session, mode, heuristic(session, mode), 'fallback', model_name, 0))
        log(f'{total} heuristic graphs written')
        return
    # A generate() call lasts as long as its longest answer, so batches hold sessions of similar size (largest first),
    # stay full (failed answers are retried inside the next batch instead of in a small batch of their own) and never
    # mix text-only and image sessions.  An out-of-memory batch is split in half and retried.
    size = lambda session: len(json.dumps({k: session.get(k) for k in ('prompt', 'checklist', 'turns', 'hints')}, ensure_ascii=False))
    for mode in ('checklist', 'decompose', 'instruction'):
        group = sorted((it for it in items if it[1] == mode), key=lambda it: -size(it[0]))
        queue = deque(dict(session=s, mode=m, output=o, attempt=1, feedback=None, answer=None) for s, m, o in group)
        limit = args.batch_size if mode != 'instruction' else max(1, args.batch_size // 2)  # image prompts are longer
        full, streak = limit, 0
        while queue:
            batch = [queue.popleft() for _ in range(min(limit, len(queue)))]
            prepared = [messages(it['session'], it['mode'], args.inventory, it['feedback'], it['answer']) for it in batch]
            try:
                outputs = generate(planner, [p[0] for p in prepared], [p[1] for p in prepared], args.max_new_tokens)
            except Exception as error:  # torch.OutOfMemoryError without importing torch here
                if 'out of memory' not in str(error).lower() or limit == 1:
                    raise
                import torch
                torch.cuda.empty_cache()
                limit, streak = max(1, limit // 2), 0
                queue.extendleft(reversed(batch))
                log(f'out of memory: batch size {mode} -> {limit}')
                continue
            streak += 1
            if limit < full and streak >= 3:  # an out-of-memory batch (often long retries) must not halve the rest of the run
                limit, streak = min(full, limit * 2), 0
            retry = []
            for it, text in zip(batch, outputs):
                try:
                    result = to_graph(parse_json_object(text), it['session'], it['mode'])
                except (ValueError, TypeError, KeyError) as error:
                    if it['attempt'] < args.attempts:
                        feedback = CUT_OFF if cut_off(text) else str(error)[:400]
                        retry.append(dict(it, attempt=it['attempt'] + 1, feedback=feedback, answer=text))
                        continue
                    append_jsonl(it['output'], record(it['session'], it['mode'], heuristic(it['session'], it['mode']), 'fallback',
                                                      model_name, args.attempts, text))
                    counts['fallback'] += 1
                    done += 1
                    continue
                status = 'ok' if it['attempt'] == 1 else 'repaired'
                append_jsonl(it['output'], record(it['session'], it['mode'], result, status, model_name, it['attempt']))
                counts[status] += 1
                done += 1
            queue.extendleft(reversed(retry))  # next batch: the retries plus the next sessions of similar size
            log(f'{done}/{total} done {dict(counts)}; {len(retry)} to retry, {len(queue)} queued ({mode}, batch {limit})')
            report(done, total, 'sessions')
    log(f'finished {dict(counts)}')

if __name__ == '__main__':
    main()
