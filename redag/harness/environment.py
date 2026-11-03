from copy import deepcopy
from dataclasses import asdict
import re
import torch
import torch.nn.functional as F
from redag.config import ReDAGConfig
from redag.graph.schema import CanvasState, ConditionDAG, ControllerAction
from redag.graph.dag import (CANVAS, MIN_FOOTPRINT_OVERLAP, WHOLE_CANVAS_SHARE, box_area, compute_active_frontier, compute_affected_subgraph, compute_spatial_subgraph,
                             condition_parents, edit_footprint, footprint_overlap, get_dependency_descendants, get_int_neighbors, semantic_support, topological_nodes)
from redag.edit.compositor import check_local_acceptance, defnorm_box_to_pixel_slice, paste_window_strictly_local
from redag.learn.credit import compute_subgraph_reward
from redag.verify.verifier import expand_bbox_with_margin

BASELINE_MODES = ('flat_full', 'tree_rollback_only')


SRC_INVARIANT_RE = re.compile(r'^(?:i|u)\d+$')


def is_source_invariant_id(node_id):
    return bool(SRC_INVARIANT_RE.match(str(node_id)))


def norm_entity_stem(phrase):
    p = phrase.strip().lower()
    if len(p) > 4 and p.endswith('ies'):
        return p[:-3] + 'y'
    if len(p) > 3 and p.endswith('es') and p[-3] in 'sxzho':
        return p[:-2]
    if len(p) > 2 and p.endswith('s') and not p.endswith('ss'):
        return p[:-1]
    return p


def union_box(boxes):
    return (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes))


def requirement_statement(node):
    text, phrase = node.question.strip().rstrip('?').strip(), node.entity_phrase.strip()
    lower = text.lower()
    for prefix, lead in (('is there ', 'there is '), ('are there ', 'there are ')):
        if lower.startswith(prefix):
            return lead + text[len(prefix):]
    for prefix, verb in (('is the ', 'is'), ('are the ', 'are'), ('does the ', 'has')):
        head = prefix + phrase.lower() + (' have ' if prefix == 'does the ' else ' ')
        if phrase and lower.startswith(head):
            return f'the {phrase} {verb} {text[len(head):]}'
    return f'the answer to "{node.question.strip()}" is yes'


def strong_edges(dag):
    nodes = deepcopy(dag.nodes)
    for node in nodes.values():
        node.depends_on = [p for p in node.depends_on if node.node_type in ('attribute', 'relation', 'count') and dag.nodes[p].node_type == 'entity']
    comp = [(p, c) for p, c in dag.comp_edges if dag.nodes[p].node_type == 'entity' and dag.nodes[c].node_type in ('attribute', 'relation', 'count')]
    return ConditionDAG(nodes, comp_edges=comp)


def preserved_phrases(dag, node_ids, target_phrase, limit=3):
    phrases = []
    for nid in node_ids:
        node = dag.nodes[nid]
        candidates = [node.entity_phrase] + [dag.nodes[p].entity_phrase for p in condition_parents(dag, nid) if dag.nodes[p].node_type == 'entity']
        for phrase in candidates:
            phrase = phrase.strip()
            if phrase and phrase.lower() != target_phrase.strip().lower() and phrase.lower() not in [q.lower() for q in phrases]:
                phrases.append(phrase)
    return phrases[:limit]


def compose_instruction(instruction, statements):
    if not statements:
        return instruction
    return instruction.rstrip() + ' In the same edit, also make sure that ' + '; '.join(statements) + '.'


GLOBAL_ROUTE_TYPES = ('background', 'global', 'relation_wide', 'count', 'prohibition')
WIDE_SUPPORT_SHARE = 0.5


def route_class(dag, node_id):
    node = dag.nodes[node_id]
    if node.node_type in ('background', 'count', 'prohibition'):
        return node.node_type
    if node.scope == 'global':
        return 'entity_global' if node.node_type == 'entity' else 'global'
    if node.node_type == 'relation':
        support = semantic_support(dag, node_id)
        return 'relation_wide' if support is None or box_area(support) >= WIDE_SUPPORT_SHARE else 'relation_local'
    return 'local'


CLAUSE_STOPWORDS = {
    'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'there', 'here',
    'in', 'on', 'at', 'to', 'of', 'for', 'with', 'from', 'by', 'into', 'onto', 'upon',
    'if', 'present', 'visible', 'shown', 'appear', 'appears', 'appearing', 'exists', 'exist',
    'image', 'scene', 'picture', 'photo', 'frame', 'view', 'does', 'do', 'did', 'has', 'have',
    'it', 'its', 'they', 'them', 'their', 'this', 'that', 'these', 'those', 'any', 'no', 'not',
    'text', 'positioned', 'located', 'placed', 'situated', 'contains', 'contain', 'displays', 'display',
}
GENERIC_PHRASES = {
    'text', 'image', 'scene', 'background', 'picture', 'photo', 'page', 'label', 'poster', 'can', 'sign', 'card', 'panel',
}
CHECKLIST_PREFIX_RE = re.compile(
    r'^(?:create an image where|also make sure that|edit the image so(?: that)?)\s*:\s*', re.I
)


def end_sentence(text):
    clean = ' '.join(str(text or '').split()).rstrip(';')
    if not clean:
        return ''
    return clean if clean[-1] in '.!?' else clean + '.'


def split_clauses(text):
    text = ' '.join(str(text or '').split())
    out, buf, in_sq, in_dq = [], [], False, False
    for idx, ch in enumerate(text):
        if ch == "'" and not in_dq:
            prev_alnum = idx > 0 and text[idx - 1].isalnum()
            next_alnum = idx + 1 < len(text) and text[idx + 1].isalnum()
            if not (prev_alnum and next_alnum):
                in_sq = not in_sq
        elif ch == '"' and not in_sq:
            in_dq = not in_dq
        buf.append(ch)
        if ch in '.!?;' and not in_sq and not in_dq:
            nxt = text[idx + 1] if idx + 1 < len(text) else ' '
            if nxt.isspace():
                seg = ''.join(buf).strip()
                if seg:
                    out.append(seg)
                buf = []
    tail = ''.join(buf).strip()
    if tail:
        out.append(tail)
    return out


def is_checklist_prompt(request):
    text = ' '.join(str(request or '').split())
    if not text:
        return False
    if CHECKLIST_PREFIX_RE.match(text):
        return True
    return len(split_clauses(text)) >= 3


def strip_checklist_prefix(request):
    return CHECKLIST_PREFIX_RE.sub('', ' '.join(str(request or '').split())).strip()


def match_node_clauses(cleaned_request, dag, node_id):
    clauses = split_clauses(cleaned_request)
    if not clauses:
        return []
    node = dag.nodes[node_id]
    quotes = [
        (a or b).strip()
        for a, b in re.findall(r"'([^']{2,})'|\"([^\"]{2,})\"", node.question)
        if (a or b).strip() and '...' not in (a or b)
    ]
    if quotes:
        hits = [c for c in clauses if all(qt.lower() in c.lower() for qt in quotes)]
        if not hits:
            hits = [c for c in clauses if any(qt.lower() in c.lower() for qt in quotes)]
        if hits:
            return hits[:2]
    phrase = node.entity_phrase.strip().lower()
    q_words = [w for w in re.findall(r'[a-z0-9]+', node.question.lower()) if len(w) >= 3 and w not in CLAUSE_STOPWORDS]
    distinct = [w for w in q_words if not phrase or w not in phrase.split()]
    scored = []
    for clause in clauses:
        cl = clause.lower()
        c_words = set(re.findall(r'[a-z0-9]+', cl))
        d_hits = sum(
            1 for w in distinct if w in c_words or (w.endswith('s') and w[:-1] in c_words) or (w + 's' in c_words)
        )
        p_hit = bool(phrase and phrase not in GENERIC_PHRASES and phrase in cl)
        if d_hits > 0 and (p_hit or not phrase or phrase in GENERIC_PHRASES or d_hits >= 2):
            scored.append((d_hits * 2 + int(p_hit), clause))
        elif not distinct and p_hit:
            scored.append((1, clause))
    if scored:
        best_score = max(s for s, _ in scored)
        return [c for s, c in scored if s == best_score][:2]
    return []


def prohibition_directive(node, matched_clause=''):
    ql = ' '.join(node.question.strip().split()).rstrip('?').strip()
    patterns = (
        r'^(?:is there no|are there no|is the image free of)\s+(.+?)(?:\s+in the (?:image|scene|room|picture|photo))?$',
        r'^(?:if .+?, )?(?:is|are)\s+.+?\s+free of\s+(.+?)$',
        r'^(?:if .+?, )?does\s+.+?\s+(?:contain|incorporate|show|have)\s+no\s+(.+?)$',
        r'^(?:if .+?, )?does\s+.+?\s+not\s+have\s+(?:any\s+|a\s+|an\s+)?(.+?)$',
        r'^(?:if .+?, )?are there not any\s+(.+?)$',
    )
    for pat in patterns:
        match = re.match(pat, ql, re.I)
        if match:
            return f"Remove any {match.group(1).strip().rstrip('.')} from the image"
    other_match = re.match(r'^are there (?:any )?(.+?\bother than\b.+)$', ql, re.I)
    if other_match:
        return f'Remove any {other_match.group(1).strip()}'
    if matched_clause:
        mc = matched_clause.strip().rstrip('.!?;')
        for no_pat in (
            r'^no\s+(.+?)\s+(?:is|are)\s+(?:present|visible|in the image)$',
            r'^no\s+(.+?)\s+appear(?:s)?(?:\s+in the image)?$',
        ):
            no_match = re.match(no_pat, mc, re.I)
            if no_match:
                return f'Remove any {no_match.group(1).strip()} from the image'
        return mc
    if_not = re.match(r'^if .+?, (?:is|are)\s+(.+?)\s+not\s+(.+?)$', ql, re.I)
    if if_not:
        return f'Make sure {if_not.group(1).strip()} is not {if_not.group(2).strip()}'
    return 'Make sure that ' + requirement_statement(node)


def request_snippet(request, dag, node_id, max_words=40):
    request = ' '.join(str(request or '').split())
    if not request:
        return ''
    parents = condition_parents(dag, node_id)
    phrases = {dag.nodes[node_id].entity_phrase.strip().lower()} | {dag.nodes[p].entity_phrase.strip().lower() for p in parents}
    phrases = {p for p in phrases if p and p not in GENERIC_PHRASES}
    sentences = split_clauses(request)
    chosen = [s for s in sentences if any(p in s.lower() for p in phrases)]
    if chosen:
        return ' '.join(chosen)
    return request if len(request.split()) <= max_words else ''


def with_user_request(instruction, snippet):
    if not snippet:
        return instruction
    clean_snip = snippet.strip().rstrip('.!?;')
    return instruction.rstrip() + ' The user asked: "' + clean_snip + '."'


def global_instruction(request, node):
    lead = end_sentence(request)
    statement = requirement_statement(node)
    return (lead + ' ' if lead else '') + 'Make sure that ' + statement + '.'


def turn_instruction(request):
    lead = end_sentence(request)
    return (lead + ' ' if lead else '') + 'Keep everything else in the image unchanged.'


def retry_instruction(request, dag, undrawn, dropped, limit=3):
    text = turn_instruction(request)
    if undrawn:
        text += ' Make sure that ' + '; '.join(requirement_statement(dag.nodes[n]) for n in undrawn[:limit]) + '.'
    if dropped:
        text += ' Also keep these true: ' + '; '.join(requirement_statement(dag.nodes[n]) for n in dropped[:limit]) + '.'
    return text


def focused_global_instruction(request, dag, node_id):
    node = dag.nodes[node_id]
    stmt = requirement_statement(node)
    keep = ' Keep everything else in the image unchanged.'
    if not is_checklist_prompt(request):
        lead = end_sentence(request)
        if node.node_type == 'prohibition' and not lead:
            lead = end_sentence(prohibition_directive(node))
        return (lead + ' ' if lead else '') + 'Make sure that ' + stmt + '.' + keep
    cleaned = strip_checklist_prefix(request)
    matched = match_node_clauses(cleaned, dag, node_id)
    if node.node_type == 'prohibition':
        directive = end_sentence(prohibition_directive(node, matched[0] if matched else ''))
        return directive + keep
    lead = ' '.join(end_sentence(c) for c in matched).strip()
    return (lead + ' ' if lead else '') + 'Make sure that ' + stmt + '.' + keep


def focused_turn_instruction(request, dag, targets):
    if not is_checklist_prompt(request):
        return turn_instruction(request)
    cleaned = strip_checklist_prefix(request)
    sentences, seen = [], set()
    for nid in targets:
        node = dag.nodes[nid]
        matched = match_node_clauses(cleaned, dag, nid)
        if node.node_type == 'prohibition':
            raw_parts = [prohibition_directive(node, matched[0] if matched else '')]
        elif matched:
            raw_parts = matched
        else:
            raw_parts = ['Make sure that ' + requirement_statement(node)]
        for part in raw_parts:
            for seg in split_clauses(part):
                sent = end_sentence(seg)
                key = sent.rstrip('.!?').lower()
                if key and key not in seen:
                    seen.add(key)
                    sentences.append(sent)
    body = ' '.join(sentences).strip()
    return (body + ' ' if body else '') + 'Keep everything else in the image unchanged.'


def unlocked_satisfied_descendants(dag, scores, node_id):
    if scores.get(node_id, 0.0) >= dag.nodes[node_id].threshold:
        return 0
    trial = dict(scores, **{node_id: max(dag.nodes[node_id].threshold, 1.0)})
    return max(0, len(valid_nodes(dag, trial)) - len(valid_nodes(dag, scores)) - 1)


def logical_parents(dag):
    parents = {nid: set(node.depends_on) for nid, node in dag.nodes.items()}
    for u, v in dag.comp_edges + dag.supp_edges:
        if v in parents:
            parents[v].add(u)
    return {v: {p for p in ps if p in dag.nodes and p != v} for v, ps in parents.items()}


def graph_ancestors(dag, node_id):
    parents = logical_parents(dag)
    seen, stack = set(), list(parents.get(node_id, ()))
    while stack:
        cur = stack.pop()
        if cur in seen or cur == node_id or cur not in dag.nodes:
            continue
        seen.add(cur)
        stack.extend(p for p in parents.get(cur, ()) if p not in seen)
    return seen


def valid_nodes(dag, scores):
    parents, memo = logical_parents(dag), {}

    def valid(v, stack=()):
        if v not in memo:
            memo[v] = scores.get(v, 0.0) >= dag.nodes[v].threshold and all(valid(p, stack + (v,)) for p in parents[v] if p not in stack)
        return memo[v]
    return {v for v in dag.nodes if valid(v)}


def consistent_count(dag, scores, nodes=None):
    valid = valid_nodes(dag, scores)
    return len(valid) if nodes is None else sum(1 for v in nodes if v in valid)


def direct_consistent_count(dag, scores, nodes):
    parents = logical_parents(dag)
    ok = lambda v: scores.get(v, 0.0) >= dag.nodes[v].threshold
    return sum(1 for v in nodes if ok(v) and all(ok(p) for p in parents[v]))


def net_progress_acceptance(dag, targets, prev_scores, new_scores, min_target_gain=0.05):
    before, after = valid_nodes(dag, prev_scores), valid_nodes(dag, new_scores)
    if len(after) != len(before):
        return len(after) > len(before)
    damaged = any(prev_scores.get(v, 0.0) >= dag.nodes[v].threshold > new_scores.get(v, 0.0) for v in dag.nodes)
    return not damaged and any(new_scores[t] - prev_scores[t] >= min_target_gain for t in targets)


def joint_acceptance(dag, targets, prev_scores, new_scores, protected_set, affected, min_target_gain=0.05, gate='joint'):
    if gate == 'net2':
        return net_progress_acceptance(dag, targets, prev_scores, new_scores, min_target_gain)
    target_ok = any(new_scores[t] - prev_scores[t] >= min_target_gain or new_scores[t] >= dag.nodes[t].threshold for t in targets)
    gain = sum(dag.nodes[v].weight * (new_scores[v] - prev_scores[v]) for v in affected)
    if not target_ok or gain <= 0:
        return False
    if gate == 'mean':
        return True
    if gate == 'net':
        return direct_consistent_count(dag, new_scores, affected) > direct_consistent_count(dag, prev_scores, affected)
    return all(new_scores[v] >= dag.nodes[v].threshold for v in protected_set & affected)


class ReDAGHarness:
    def __init__(self, verifier, grounder, feather_px=None, step_cost=None, lam=None, config=None, mode='redag_full'):
        self.config = config or ReDAGConfig()
        self.verifier, self.grounder = verifier, grounder
        self.feather_px = self.config.feather_px if feather_px is None else feather_px
        self.step_cost = self.config.step_cost if step_cost is None else step_cost
        self.lam = self.config.protection_lambda if lam is None else lam
        self.verifier.gate_tau = self.config.gate_tau
        self.mode, self.history, self.last_candidate = mode, [], None
        self.verification_budget = self.generation_budget = self.render_limit = None
        self.cost_budget, self.generation_cost, self.cost_origin, self.budget_per_turn = None, 1.0, None, False
        self.generation_counter, self.last_run, self.unknown_nodes = 0, {}, set()
        self.retry_limit = 2
        self.protect_masks, self.mask_margin, self.container_share, self.min_editable_pixels = False, 16, 0.5, 1024
        self.closure_instructions, self.closure_limit, self.confirm_drops, self.soft_gating = False, 3, False, False
        self.preserve_instructions, self.dag_box, self.edge_policy = False, False, 'full'
        self.counterfactual_credit, self.acceptance_bonus = False, 0.0
        self.scope_routing, self.global_types, self.global_gate = False, GLOBAL_ROUTE_TYPES, 'joint'
        self.user_text, self.turn_pass, self.turn_gate = False, False, 'joint'
        self.turn_request, self.turn_nodes, self.turn_index, self.draft_first_turn = '', set(), 0, False
        self.global_retries, self.global_failures = 0, {}
        self.strategy, self.local_failures, self.turn_count, self.on_turn_start = None, {}, 1, None
        self.turn_retries, self.turn_priority, self.local_gate, self.turn_localize, self.turn_protect = 0, False, 'legacy', False, False
        self.node_turn_request, self.repair_queue, self.repaired_in_turn, self.steps_remaining = {}, [], set(), 0
        self.source_invariants = None
        self.single_stage_verify, self.focused_turn, self.smart_turn_pass = False, False, False
        self.cascade_priority, self.repair_gate, self.monotonic_turn, self.candidate_k = False, False, False, 1

    def budget_left(self):
        if self.render_limit is None or not self.generation_budget:
            return 1.0
        return max(0.0, min(1.0, (self.render_limit - self.generation_counter) / self.generation_budget))

    def route_features(self, state, dag, node_id):
        from redag.learn.strategy import NODE_TYPES, ROUTE_CLASSES
        node, key, cls = dag.nodes[node_id], (self.turn_index, node_id), route_class(dag, node_id)
        return ([1.0] + [float(node.node_type == t) for t in NODE_TYPES] + [float(cls == c) for c in ROUTE_CLASSES]
                + [min(self.global_failures.get(key, 0), 3) / 3, min(self.local_failures.get(key, 0), 3) / 3,
                   float(state.scores.get(node_id, 0.0)), float(self.turn_index >= self.turn_count - 1), self.budget_left()])

    def accept_features(self, dag, targets, parent_state, new_scores, affected, route):
        threshold = lambda v: dag.nodes[v].threshold
        drops = {v for v in dag.nodes if parent_state.scores.get(v, 0.0) >= threshold(v) > new_scores.get(v, 0.0)}
        this_turn = sum(1 for v in drops if v in self.turn_nodes)
        descendants = (get_dependency_descendants(dag, drops) - drops) & parent_state.protected_set if drops else set()
        repairable = {v for v in drops if route_class(dag, v) in ('local', 'relation_local') and edit_footprint(dag, v) is not None}
        hits = sum(1 for t in targets if new_scores[t] >= threshold(t) > parent_state.scores[t])
        gain = sum(dag.nodes[v].weight * (new_scores[v] - parent_state.scores[v]) for v in affected) / max(1, len(affected))
        clip = lambda x: max(-3.0, min(3.0, float(x))) / 3
        single = float(len(drops) == 1 and len(repairable) == 1 and not descendants and self.budget_left() > 0)
        return [1.0, clip(len(drops)), clip(this_turn), clip(len(drops) - this_turn), clip(len(descendants)), clip(len(repairable)), single,
                clip(hits), max(-1.0, min(1.0, 5 * gain)), float(route == 'turn'), float(self.turn_index >= self.turn_count - 1), self.budget_left()]

    def strategy_acceptance(self, dag, targets, parent_state, new_scores, affected, route):
        before, after = valid_nodes(dag, parent_state.scores), valid_nodes(dag, new_scores)
        if len(after) != len(before):
            return len(after) > len(before)
        threshold = lambda v: dag.nodes[v].threshold
        if not any(parent_state.scores.get(v, 0.0) >= threshold(v) > new_scores.get(v, 0.0) for v in dag.nodes):
            return any(new_scores[t] - parent_state.scores[t] >= self.config.min_target_gain for t in targets)
        if not any(new_scores[t] >= threshold(t) > parent_state.scores[t] for t in targets):
            return False
        return self.strategy.decide('accept', self.accept_features(dag, targets, parent_state, new_scores, affected, route))

    def retry_features(self, dag, previous, attempts):
        seen, targets = previous['raw_scores'], previous.get('targets') or [previous['target_node_id']]
        undrawn = sum(1 for n in targets if n in dag.nodes and seen.get(n, 0.0) < dag.nodes[n].threshold) / max(1, len(targets))
        damaged = sum(1 for v in previous['parent_state'].protected_set if v in dag.nodes and seen.get(v, 1.0) < dag.nodes[v].threshold)
        clip = lambda x: max(-3.0, min(3.0, float(x))) / 3
        return [1.0, undrawn, clip(damaged), self.budget_left(), float(self.turn_index >= self.turn_count - 1), clip(attempts)]

    def turn_features(self, state, dag, targets):
        at_risk = min(1.0, sum(1 for n in dag.nodes if n not in targets and state.scores.get(n, 0.0) >= dag.nodes[n].threshold) / 30.0)
        canvas = float(self.turn_box(dag, targets, state.image, state.scores) == CANVAS)
        return [1.0, canvas, at_risk, canvas * at_risk, len(targets) / max(1, len(dag.nodes)), float(self.turn_index >= self.turn_count - 1), self.budget_left()]

    def routes_globally(self, dag, node_id):
        if self.global_retries and self.global_failures.get((self.turn_index, node_id), 0) >= self.global_retries:
            return False
        return self.scope_routing and self.mode == 'redag_full' and route_class(dag, node_id) in self.global_types

    def node_request(self, node_id):
        if node_id in self.node_turn_request:
            return self.node_turn_request[node_id]
        return '' if self.node_turn_request else self.turn_request

    def active_source_invariants(self, dag):
        if self.draft_first_turn:
            return set()
        if self.source_invariants is not None:
            inv = {
                nid for nid in self.source_invariants
                if nid in dag.nodes and all(is_source_invariant_id(a) for a in graph_ancestors(dag, nid))
            }
        else:
            inv = {
                nid for nid in dag.nodes
                if is_source_invariant_id(nid) and all(is_source_invariant_id(a) for a in graph_ancestors(dag, nid))
            }
        return inv if any(n not in inv for n in dag.nodes) else set()

    def turn_targets(self, dag, state):
        if not self.turn_pass or self.mode != 'redag_full' or not self.turn_request.strip():
            return []
        if self.turn_index == 0 and self.draft_first_turn:
            return []
        targets = [nid for nid in dag.nodes if nid in self.turn_nodes and state.scores.get(nid, 0.0) < dag.nodes[nid].threshold]
        if self.smart_turn_pass and self.draft_first_turn and 0 < len(targets) <= 2:
            if all(not self.routes_globally(dag, t) and (fp := edit_footprint(dag, t)) is not None
                   and box_area(fp) < WHOLE_CANVAS_SHARE for t in targets):
                return []
        return targets

    def turn_footprint(self, dag, node_id, scores, located, seen=()):
        node = dag.nodes[node_id]
        if not self.turn_localize:
            return edit_footprint(dag, node_id)
        if node.node_type == 'prohibition':
            return located.get(node_id)
        owners = [p for p in condition_parents(dag, node_id) if dag.nodes[p].node_type == 'entity' and p not in seen]
        if node.node_type == 'entity' and scores.get(node_id, 0.0) < node.threshold:
            placed = [dag.nodes[p].bbox for p in owners if dag.nodes[p].bbox is not None and scores.get(p, 0.0) >= dag.nodes[p].threshold]
            if placed:
                return union_box(placed)
            return () if any(b is not None for b in located.values()) else None
        if node.node_type in ('attribute', 'relation') and owners:
            parts = [self.turn_footprint(dag, p, scores, located, seen + (node_id,)) for p in owners]
            if any(b is None for b in parts):
                return None
            parts = [b for b in parts if b]
            return union_box(parts) if parts else ()
        return edit_footprint(dag, node_id)

    def turn_box(self, dag, targets, image=None, scores=None):
        located = {}
        if self.turn_localize and self.grounder is not None and image is not None:
            for nid in targets:
                node = dag.nodes[nid]
                if node.node_type == 'prohibition' and node.entity_phrase.strip():
                    found = self.grounder.locate(image, node.entity_phrase)
                    located[nid] = union_box(found) if found else None
        boxes = [self.turn_footprint(dag, nid, scores or {}, located) for nid in targets]
        if any(b is None for b in boxes):
            return CANVAS
        if self.focused_turn and self.turn_localize and not is_checklist_prompt(self.turn_request):
            req_low = self.turn_request.lower()
            target_set = set(targets)
            for nid, node in dag.nodes.items():
                phrase = node.entity_phrase.strip().lower()
                pat = rf"(?<![a-z0-9]){re.escape(phrase)}(?:e?s|'s)?(?![a-z0-9])" if phrase else ''
                if (nid not in target_set and phrase and phrase not in GENERIC_PHRASES and re.search(pat, req_low)
                        and node.bbox is not None and (scores or {}).get(nid, 0.0) >= node.threshold):
                    boxes.append(node.bbox)
        boxes = [b for b in boxes if b]
        if not boxes or any(box_area(b) >= WHOLE_CANVAS_SHARE for b in boxes):
            return CANVAS
        union = expand_bbox_with_margin(union_box(boxes), 1.25)
        return CANVAS if box_area(union) >= WHOLE_CANVAS_SHARE else union

    def turn_retry_source(self, records):
        rejected = 0
        for record in reversed(records):
            if record.get('route') != 'turn' or record['accepted']:
                break
            rejected += 1
        if not 0 < rejected <= self.turn_retries:
            return None
        if self.strategy is not None and self.mode == 'redag_full':
            dag = records[-1]['dag']
            if not self.strategy.decide('retry', self.retry_features(dag, records[-1], rejected)):
                return None
        return records[-1]

    def evaluate_best_of_k(self, state, graph, target_node_id, action, editor, seed, mask, route, targets=None, k_count=1):
        targets = list(targets or [target_node_id])
        eff = self.effective_graph(graph)
        base_gf, base_lf = dict(self.global_failures), dict(self.local_failures)
        evaluated = []
        for k_idx in range(k_count):
            if k_idx > 0:
                reserve = 1 if any(c['accepted'] and c.get('repair_drop') for c, _, _ in evaluated) else 0
                if self.render_limit is not None and self.generation_counter + reserve >= self.render_limit:
                    break
                if self.verifier.call_limit is not None and self.verifier.call_counter >= self.verifier.call_limit:
                    break
                if self.cost_budget is not None and self.cost_origin is not None:
                    if self.spent_cost() + reserve * self.generation_cost >= self.cost_budget:
                        break
            render_seed = seed if k_idx == 0 else seed + k_idx * 10007
            self.generation_counter += 1
            crop, editable = self.render(editor, state, action, render_seed, mask)
            self.global_failures, self.local_failures = dict(base_gf), dict(base_lf)
            try:
                self.evaluate_candidate_action(state, graph, target_node_id, action, crop, editable, route=route, targets=targets)
            except RuntimeError as error:
                self.global_failures, self.local_failures = base_gf, base_lf
                if str(error) != 'Verifier call budget exhausted':
                    raise
                if k_idx == 0:
                    return None
                break
            cand = dict(self.last_candidate)
            evaluated.append((cand, dict(self.global_failures), dict(self.local_failures)))
            if cand['accepted'] and not cand.get('repair_drop'):
                break
        best = max(evaluated, key=lambda item: (
            int(item[0]['accepted'] and not item[0].get('repair_drop')),
            int(item[0]['accepted']),
            consistent_count(eff, item[0]['raw_scores']),
            sum(item[0]['raw_scores'][t] for t in targets),
            item[0]['reward'],
        ))
        self.last_candidate, self.global_failures, self.local_failures = best[0], best[1], best[2]
        return best[0]

    def turn_candidates(self, state, dag, editor, targets, seed, previous=None):
        if self.render_limit is not None and self.generation_counter >= self.render_limit:
            return []
        if self.verifier.call_limit is not None and self.verifier.call_counter >= self.verifier.call_limit:
            return []
        if self.cost_budget is not None and self.cost_origin is not None and self.spent_cost() >= self.cost_budget:
            return []
        graph = deepcopy(dag)
        instruction = (
            focused_turn_instruction(self.turn_request, graph, targets)
            if self.focused_turn else turn_instruction(self.turn_request)
        )
        if previous is not None:
            seen = previous['raw_scores']
            undrawn = [n for n in targets if seen.get(n, 0.0) < graph.nodes[n].threshold]
            prev_prot = previous['parent_state'].protected_set
            dropped = [n for n in prev_prot if n in graph.nodes and seen.get(n, 1.0) < graph.nodes[n].threshold]
            use_strip = self.focused_turn and is_checklist_prompt(self.turn_request)
            base_req = strip_checklist_prefix(self.turn_request) if use_strip else self.turn_request
            instruction = retry_instruction(base_req, graph, undrawn, sorted(dropped))
        action = ControllerAction('modify_attr', self.turn_box(graph, targets, state.image, state.scores), instruction)
        mask = self.turn_mask(state, graph, targets, action)
        k_count = self.candidate_k if (self.candidate_k > 1 and self.mode == 'redag_full') else 1
        chosen = self.evaluate_best_of_k(
            state, graph, targets[0], action, editor, seed, mask, route='turn', targets=targets, k_count=k_count
        )
        if chosen is None:
            return []
        return [dict(chosen, sample={'action': action, 'route': 'turn', 'targets': list(targets)})]

    def effective_graph(self, dag):
        if self.mode not in ('redag_full', 'flat_full', 'tree_rollback_only', 'spatial_local', 'spatial_no_gates'):
            raise ValueError('Unknown environment mode')
        if self.mode in ('redag_full', 'spatial_local'):
            return strong_edges(dag) if self.edge_policy == 'strong' else dag
        nodes = deepcopy(dag.nodes)
        for parent, child in dag.comp_edges + dag.supp_edges:
            nodes[child].depends_on = sorted(set(nodes[child].depends_on) | {parent})
        return ConditionDAG(nodes)

    def active_frontier(self, dag, scores):
        graph, gated = self.effective_graph(dag), self.mode in ('redag_full', 'spatial_local')
        ready = compute_active_frontier(graph, scores, self.config.gate_tau, gated=gated)
        if not gated or not self.soft_gating:
            return ready
        return ready + [nid for nid in compute_active_frontier(graph, scores, self.config.gate_tau, gated=False) if nid not in ready
                        and all(scores.get(p, 0.0) >= self.config.gate_tau for p in condition_parents(graph, nid) if graph.nodes[p].node_type == 'entity')]

    def closure_requirements(self, state, dag, target_node_id, action):
        target = dag.nodes[target_node_id]
        if target.node_type == 'entity':
            anchors = {target_node_id}
        elif target.node_type == 'attribute':
            anchors = {p for p in condition_parents(dag, target_node_id) if dag.nodes[p].node_type == 'entity'}
        else:
            return []
        satisfied = {nid for nid, score in state.scores.items() if score >= dag.nodes[nid].threshold}
        chosen = []
        for nid, node in dag.nodes.items():
            if nid == target_node_id or nid in satisfied or node.node_type not in ('attribute', 'relation'):
                continue
            parents = set(condition_parents(dag, nid))
            if not parents & anchors or not parents <= satisfied | anchors | {target_node_id}:
                continue
            support = edit_footprint(dag, nid) if node.node_type == 'attribute' else semantic_support(dag, nid)
            if support is not None and footprint_overlap(support, action.box) < MIN_FOOTPRINT_OVERLAP:
                continue
            chosen.append(nid)
        chosen.sort(key=lambda nid: dag.nodes[nid].weight * (1.0 - state.scores.get(nid, 0.0)), reverse=True)
        return chosen[:self.closure_limit]

    def preserve_requirements(self, state, dag, target_node_id, action):
        graph = self.effective_graph(dag)
        anchors = {target_node_id, *condition_parents(graph, target_node_id)}
        related = anchors | get_dependency_descendants(graph, anchors) | get_int_neighbors(graph, anchors)
        affected = compute_affected_subgraph(graph, action, target_node_id, state.protected_set)
        kept = [nid for nid in graph.nodes if nid != target_node_id and nid in affected & state.protected_set & related]
        rank = {'relation': 0, 'count': 0, 'prohibition': 1, 'attribute': 2, 'entity': 3, 'background': 4}
        kept.sort(key=lambda nid: (rank.get(graph.nodes[nid].node_type, 5), -graph.nodes[nid].weight))
        return kept[:self.closure_limit]

    def snapped_box(self, dag, target_node_id, box):
        node = dag.nodes[target_node_id]
        owners = [dag.nodes[p].bbox for p in condition_parents(dag, target_node_id) if dag.nodes[p].node_type == 'entity' and dag.nodes[p].bbox is not None]
        if node.node_type == 'attribute' and owners:
            support = owners[0]
        elif node.node_type == 'relation' and len(owners) >= 2:
            support = (min(b[0] for b in owners), min(b[1] for b in owners), max(b[2] for b in owners), max(b[3] for b in owners))
        else:
            return box
        snapped = expand_bbox_with_margin(support, 1.1)
        return CANVAS if box_area(snapped) >= WHOLE_CANVAS_SHARE else snapped

    def confirm_protected_drops(self, parent_state, dag, target_node_id, action, candidate, scores, affected):
        spatial, gate, limit = compute_spatial_subgraph(dag, action, target_node_id), self.verifier.gate_tau, self.config.max_protected_drop
        parents = {nid: [p for p, c in dag.comp_edges if c == nid] for nid in dag.nodes}
        gated = {nid for nid in affected if any(scores[p] < gate for p in parents[nid])}
        scores, refuted = dict(scores), []
        height, width = candidate.shape[-2:]
        for nid in topological_nodes(dag):
            if nid not in affected or nid == target_node_id:
                continue
            node = dag.nodes[nid]
            if nid in gated and any(p in refuted for p in parents[nid]) and all(scores[p] >= gate for p in parents[nid]):
                scores[nid] = self.verifier.verify_single_node_two_stage(candidate, node, dag)
            if nid in spatial or nid not in parent_state.protected_set or scores[nid] - parent_state.scores[nid] >= -limit:
                continue
            if any(scores[p] < gate for p in parents[nid]):
                continue
            support = semantic_support(dag, nid)
            if support is None:
                continue
            x1, y1, x2, y2 = defnorm_box_to_pixel_slice(expand_bbox_with_margin(support, 1.2), height, width)
            before, after = parent_state.image[:, y1:y2, x1:x2], candidate[:, y1:y2, x1:x2]
            if not torch.equal(before, after) and self.verifier.score(before, node.question) - self.verifier.score(after, node.question) > limit:
                continue
            scores[nid] = parent_state.scores[nid]
            refuted.append(nid)
        return scores, refuted

    def grounding_state(self):
        return (dict(self.grounder.counts), dict(self.grounder.masks)) if self.grounder else None

    def restore_grounding(self, state):
        if self.grounder and state is not None:
            self.grounder.counts, self.grounder.masks = map(dict, state)

    def verify_current(self, image, dag, initial, changed):
        try:
            scores = self.verifier.verify_subgraph_after_edit(image, self.effective_graph(dag), initial, changed)
        except RuntimeError as error:
            if str(error) != 'Verifier call budget exhausted':
                raise
            self.unknown_nodes.update(changed)
            return {n: 0.0 if n in changed else q for n, q in initial.items()}
        self.unknown_nodes.difference_update(changed)
        return scores

    def initialize(self, image, dag):
        if self.grounder is not None:
            dag = self.grounder.ground(image, dag)
        self.verifier.two_stage = (self.mode not in BASELINE_MODES) and not (self.single_stage_verify and self.mode == 'redag_full')
        self.unknown_nodes = set()
        scores = self.verify_current(image, dag, dict.fromkeys(dag.nodes, 0.0), set(dag.nodes))
        return CanvasState(image, scores, {n for n in dag.nodes if n not in self.unknown_nodes and scores[n] >= dag.nodes[n].threshold}), dag

    def evaluate_candidate_action(self, parent_state, dag, target_node_id, action, edited_crop, editable=None, route='local', targets=None):
        calls, start = self.verifier.call_counter, len(self.verifier.records)
        graph = self.effective_graph(dag)
        self.verifier.two_stage = (self.mode not in BASELINE_MODES) and not (self.single_stage_verify and self.mode == 'redag_full')
        targets = list(targets or [target_node_id])
        whole = box_area(action.box) >= WHOLE_CANVAS_SHARE
        joint = route in ('global', 'turn')
        candidate = paste_window_strictly_local(parent_state.image, edited_crop, action.box, 0 if joint and whole else self.feather_px, editable)
        if joint and whole:
            affected = set(graph.nodes)
        elif route == 'turn':
            affected = set().union(*(compute_affected_subgraph(graph, action, t, parent_state.protected_set) for t in targets))
        elif self.mode == 'redag_full':
            affected = compute_affected_subgraph(graph, action, target_node_id, parent_state.protected_set)
        elif self.mode in ('spatial_local', 'spatial_no_gates'):
            affected = compute_spatial_subgraph(dag, action, target_node_id)
        else:
            affected = set(graph.nodes)
        uncertain = {n for n, node in dag.nodes.items() if node.node_type in ('entity', 'attribute') and edit_footprint(dag, n) is None}
        prescreen_ok = (
            self.single_stage_verify
            and self.mode == 'redag_full'
            and not self.counterfactual_credit
            and self.strategy is None
            and not (route == 'turn' and self.turn_retries > 0)
            and hasattr(self.verifier, 'verify_single_node_two_stage')
        )
        pre_verified, fast_rejected = None, False
        if prescreen_ok:
            comp_parents = {nid: [] for nid in graph.nodes}
            for p_id, c_id in graph.comp_edges:
                if c_id in comp_parents:
                    comp_parents[c_id].append(p_id)
            needed, queue = set(), list(set(targets) & affected)
            while queue:
                cur = queue.pop()
                if cur in needed or cur not in affected:
                    continue
                needed.add(cur)
                queue.extend(p_id for p_id in comp_parents[cur] if p_id in affected and p_id not in needed)
            if len(affected) > len(needed) + 1:
                trial_scores, pre_verified = dict(parent_state.scores), {}
                for v_id in topological_nodes(graph):
                    if v_id not in needed:
                        continue
                    if any(trial_scores.get(p_id, 0.0) < self.verifier.gate_tau for p_id in comp_parents[v_id]):
                        trial_scores[v_id] = 0.0
                    else:
                        val = self.verifier.verify_single_node_two_stage(candidate, graph.nodes[v_id], graph)
                        trial_scores[v_id] = val
                        pre_verified[v_id] = val
                min_gain = self.config.min_target_gain
                target_ok = any(
                    trial_scores[t] >= graph.nodes[t].threshold
                    or trial_scores[t] - parent_state.scores.get(t, 0.0) >= min_gain
                    for t in targets
                )
                if not target_ok:
                    raw_scores = trial_scores
                    fast_rejected = True
        refuted, accepted, repair_drop = [], False, None
        if not fast_rejected:
            if pre_verified:
                raw_scores = self.verifier.verify_subgraph_after_edit(
                    candidate, graph, parent_state.scores, affected, pre_verified=pre_verified
                )
            else:
                raw_scores = self.verifier.verify_subgraph_after_edit(candidate, graph, parent_state.scores, affected)
            target_gain = raw_scores[target_node_id] - parent_state.scores[target_node_id]
            has_gain = target_gain >= self.config.min_target_gain or raw_scores[target_node_id] >= dag.nodes[target_node_id].threshold
            if not joint and self.confirm_drops and self.mode == 'redag_full' and has_gain:
                raw_scores, refuted = self.confirm_protected_drops(
                    parent_state, graph, target_node_id, action, candidate, raw_scores, affected
                )
            if self.strategy is not None and self.mode == 'redag_full':
                accepted = self.strategy_acceptance(graph, targets, parent_state, raw_scores, affected, route)
            elif joint:
                gate_name = self.turn_gate if route == 'turn' else self.global_gate
                accepted = joint_acceptance(
                    graph, targets, parent_state.scores, raw_scores,
                    parent_state.protected_set, affected, self.config.min_target_gain, gate_name
                )
            elif self.local_gate == 'net2':
                accepted = net_progress_acceptance(graph, targets, parent_state.scores, raw_scores, self.config.min_target_gain)
            else:
                accepted = check_local_acceptance(
                    graph, target_node_id, parent_state.scores, raw_scores,
                    parent_state.protected_set, affected, self.config.min_target_gain, self.config.max_protected_drop
                )
            if not accepted and self.repair_gate and self.mode == 'redag_full' and self.steps_remaining > 0:
                has_render = self.render_limit is None or self.generation_counter < self.render_limit
                has_cost = (
                    self.cost_budget is None
                    or self.cost_origin is None
                    or self.spent_cost() + self.generation_cost <= self.cost_budget
                )
                if has_render and has_cost:
                    before_v, after_v = valid_nodes(graph, parent_state.scores), valid_nodes(graph, raw_scores)
                    hit = any(raw_scores[t] >= graph.nodes[t].threshold > parent_state.scores.get(t, 0.0) for t in targets)
                    drops = [
                        v for v in graph.nodes
                        if parent_state.scores.get(v, 0.0) >= graph.nodes[v].threshold > raw_scores.get(v, 0.0)
                    ]
                    if hit and len(after_v) >= len(before_v) and len(drops) == 1:
                        d = drops[0]
                        fp = edit_footprint(graph, d)
                        no_cascade = unlocked_satisfied_descendants(graph, raw_scores, d) == 0
                        prohibited = {
                            norm_entity_stem(n.entity_phrase)
                            for n in graph.nodes.values()
                            if n.node_type == 'prohibition' and n.entity_phrase.strip()
                        }
                        d_stem = norm_entity_stem(graph.nodes[d].entity_phrase)
                        if (d not in targets and d not in self.repaired_in_turn and not self.routes_globally(graph, d)
                                and fp is not None and box_area(fp) < WHOLE_CANVAS_SHARE
                                and no_cascade and (not d_stem or d_stem not in prohibited)):
                            accepted, repair_drop = True, d
        if not accepted and route in ('global', 'local'):
            failures = self.global_failures if route == 'global' else self.local_failures
            key = (self.turn_index, target_node_id)
            failures[key] = failures.get(key, 0) + 1
        delivered_image = candidate if accepted else parent_state.image
        delivered_scores = raw_scores if accepted else dict(parent_state.scores)
        reward_scores = raw_scores if self.counterfactual_credit else delivered_scores
        reward = compute_subgraph_reward(graph, parent_state.scores, reward_scores, parent_state.protected_set, affected, self.step_cost, self.lam)
        if self.counterfactual_credit and accepted and self.acceptance_bonus > 0:
            reward = float(reward + self.acceptance_bonus)
        protected = {nid for nid, score in delivered_scores.items() if score >= dag.nodes[nid].threshold}
        state = CanvasState(delivered_image, delivered_scores, protected, parent_state.step_idx + 1)
        self.last_candidate = dict(dag=deepcopy(dag), parent_state=parent_state,
            candidate_state=CanvasState(candidate, raw_scores, {n for n in raw_scores if raw_scores[n] >= dag.nodes[n].threshold}),
            delivered_state=state, raw_scores=raw_scores, affected=affected, target_node_id=target_node_id, action=action,
            accepted=accepted, reward=reward, coverage_uncertain_nodes=sorted(uncertain), refuted_drops=refuted,
            stage_scores=self.verifier.records[start:], verifier_calls=self.verifier.call_counter - calls, route=route, targets=targets,
            repair_drop=repair_drop)
        return state, reward, accepted, affected

    def editable_mask(self, state, dag, target_node_id, action):
        if not self.protect_masks or self.grounder is None or self.mode != 'redag_full':
            return None
        target = dag.nodes[target_node_id]
        closure = {target_node_id, *condition_parents(dag, target_node_id)}
        free = closure | get_int_neighbors(dag, closure)
        if target.node_type in ('count', 'prohibition'):
            free |= {nid for nid, node in dag.nodes.items() if node.entity_phrase == target.entity_phrase}
        checked = (state.protected_set & compute_affected_subgraph(dag, action, target_node_id, state.protected_set)) - free
        keep = set()
        for nid in checked:
            if dag.nodes[nid].node_type == 'entity':
                keep.add(nid)
            elif dag.nodes[nid].node_type == 'attribute':
                keep.update(p for p in condition_parents(dag, nid) if dag.nodes[p].node_type == 'entity' and p not in free)
        return self.editable_from(state, keep, action.box)

    def turn_mask(self, state, dag, targets, action):
        if not self.turn_protect or self.grounder is None or self.mode != 'redag_full':
            return None
        if any(route_class(dag, t) in ('background', 'global') for t in targets):
            return None
        free = set(targets) | {p for t in targets for p in condition_parents(dag, t)}
        free |= get_int_neighbors(dag, free)
        phrases = {dag.nodes[t].entity_phrase for t in targets if dag.nodes[t].node_type in ('count', 'prohibition')}
        free |= {nid for nid, node in dag.nodes.items() if node.entity_phrase in phrases}
        keep = set()
        for nid in state.protected_set & set(dag.nodes):
            if nid in free:
                continue
            if dag.nodes[nid].node_type == 'entity':
                keep.add(nid)
            elif dag.nodes[nid].node_type == 'attribute':
                keep.update(p for p in condition_parents(dag, nid) if dag.nodes[p].node_type == 'entity' and p not in free)
        height, width = state.image.shape[-2:]
        keep = {nid for nid in keep if nid in self.grounder.masks and float(self.grounder.masks[nid].float().sum()) <= self.container_share * height * width}
        return self.editable_from(state, keep, action.box)

    def editable_from(self, state, keep, box):
        height, width = state.image.shape[-2:]
        x1, y1, x2, y2 = defnorm_box_to_pixel_slice(box, height, width)
        window = (slice(y1, y2), slice(x1, x2))
        layers = []
        for nid in sorted(keep):
            if nid not in self.grounder.masks:
                continue
            layer = self.grounder.masks[nid].to(state.image.device).float()
            inside = float(layer[window].sum())
            if inside > self.container_share * (y2 - y1) * (x2 - x1) and inside < self.container_share * float(layer.sum()):
                continue
            layers.append(layer)
        if not layers:
            return None
        margin = self.mask_margin
        union = F.max_pool2d(torch.stack(layers).amax(0)[None, None], kernel_size=2 * margin + 1, stride=1, padding=margin)[0, 0]
        editable = 1 - union
        if float(editable[window].sum()) < self.min_editable_pixels:
            return None
        return editable

    def spent_cost(self):
        calls, renders = self.cost_origin
        return (self.generation_counter - renders) * self.generation_cost + (self.verifier.call_counter - calls)

    def render(self, editor, state, action, seed, mask):
        if mask is not None:
            try:
                return editor.render(state.image, action, seed=seed, mask=mask), mask
            except ValueError:
                pass
        return editor.render(state.image, action, seed=seed), None

    def sample_group(self, state, dag, controller, editor, target_node_id, group_size=6, seed=42):
        candidates, grounding = [], self.grounding_state()
        try:
            for index in range(group_size):
                if self.render_limit is not None and self.generation_counter >= self.render_limit:
                    break
                if self.verifier.call_limit is not None and self.verifier.call_counter >= self.verifier.call_limit:
                    break
                if self.cost_budget is not None and self.cost_origin is not None and self.spent_cost() >= self.cost_budget:
                    break
                self.restore_grounding(grounding)
                graph = deepcopy(dag)
                route = 'global' if self.routes_globally(graph, target_node_id) else 'local'
                if self.strategy is not None and self.mode == 'redag_full':
                    route = 'global' if self.strategy.decide('route', self.route_features(state, graph, target_node_id)) else 'local'
                req_text = self.node_request(target_node_id)
                if route == 'global':
                    use_focus = self.focused_turn and self.mode == 'redag_full'
                    g_inst = (
                        focused_global_instruction(req_text, graph, target_node_id)
                        if use_focus else global_instruction(req_text, graph.nodes[target_node_id])
                    )
                    sample = {'action': ControllerAction('modify_attr', CANVAS, g_inst)}
                else:
                    det_cnt = self.grounder.counts.get(dag.nodes[target_node_id].entity_phrase, 1) if self.grounder else 1
                    saved_clean = getattr(controller, 'clean_instruction', None)
                    if saved_clean is not None and self.mode != 'redag_full':
                        controller.clean_instruction = False
                    try:
                        sample = controller.sample(
                            state, graph, target_node_id, seed=seed + index, temperature=0.8, detected_count=det_cnt
                        )
                    finally:
                        if saved_clean is not None:
                            controller.clean_instruction = saved_clean
                if self.mode in BASELINE_MODES:
                    sample['action'] = ControllerAction(sample['action'].op, CANVAS, sample['action'].instruction)
                if route == 'local' and self.closure_instructions and self.mode == 'redag_full':
                    sample['closure_nodes'] = self.closure_requirements(state, graph, target_node_id, sample['action'])
                    sample['action'] = ControllerAction(sample['action'].op, sample['action'].box, compose_instruction(
                        sample['action'].instruction, [requirement_statement(graph.nodes[nid]) for nid in sample['closure_nodes']]))
                if route == 'local' and self.dag_box and self.mode == 'redag_full':
                    sample['controller_box'] = sample['action'].box
                    sample['action'] = ControllerAction(sample['action'].op, self.snapped_box(graph, target_node_id, sample['action'].box), sample['action'].instruction)
                if route == 'local' and self.preserve_instructions and self.mode == 'redag_full':
                    sample['preserve_nodes'] = self.preserve_requirements(state, graph, target_node_id, sample['action'])
                    phrases = preserved_phrases(graph, sample['preserve_nodes'], graph.nodes[target_node_id].entity_phrase)
                    if phrases:
                        sample['action'] = ControllerAction(sample['action'].op, sample['action'].box, sample['action'].instruction.rstrip() + ' Keep the ' + ' and the '.join(phrases) + ' unchanged.')
                if route == 'local' and self.user_text and self.mode == 'redag_full':
                    if self.focused_turn and is_checklist_prompt(req_text):
                        snippet = ' '.join(match_node_clauses(strip_checklist_prefix(req_text), graph, target_node_id))
                    else:
                        snippet = request_snippet(req_text, graph, target_node_id, max_words=40)
                    new_inst = with_user_request(sample['action'].instruction, snippet)
                    sample['action'] = ControllerAction(sample['action'].op, sample['action'].box, new_inst)
                sample['route'] = route
                mask = None if route == 'global' else self.editable_mask(state, graph, target_node_id, sample['action'])
                k_count = self.candidate_k if (self.candidate_k > 1 and self.mode == 'redag_full' and group_size == 1) else 1
                chosen = self.evaluate_best_of_k(
                    state, graph, target_node_id, sample['action'], editor, seed, mask, route=route, k_count=k_count
                )
                if chosen is None:
                    break
                candidates.append(dict(chosen, sample=sample))
        finally:
            self.restore_grounding(grounding)
        return candidates

    def run_session(self, state, dag, controller, editor, max_steps=10, seed=42, group_size=1):
        if max_steps < 0 or group_size < 1:
            raise ValueError('Invalid session step or group count')
        records, nodes, current = [], [dict(state=state, dag=deepcopy(dag), parent=None, grounding=self.grounding_state())], 0
        history_offset = len(self.history)
        old_limits = self.verifier.call_limit, self.render_limit
        calls, renders = self.verifier.call_counter, self.generation_counter
        if old_limits[0] is None and self.verification_budget is not None:
            self.verifier.call_limit = calls + self.verification_budget
        if old_limits[1] is None and self.generation_budget is not None:
            self.render_limit = renders + self.generation_budget
        reason, failures = 'step_budget', {}
        self.repair_queue, self.repaired_in_turn = [], set()
        own_origin = self.cost_origin is None
        if own_origin:
            self.cost_origin = (calls, renders)
        try:
            for step in range(max_steps):
                self.steps_remaining = max_steps - 1 - step
                self.repair_queue = [n for n in self.repair_queue if n in dag.nodes and state.scores.get(n, 0.0) < dag.nodes[n].threshold]
                if self.unknown_nodes:
                    reason = 'execution_budget'
                    break
                retry = self.turn_retry_source(records)
                turn_targets = self.turn_targets(dag, state) if step == 0 or retry is not None else []
                if turn_targets and step == 0 and self.strategy is not None and self.mode == 'redag_full' and not self.strategy.decide('turn', self.turn_features(state, dag, turn_targets)):
                    turn_targets = []
                frontier = self.active_frontier(dag, state.scores)
                if self.focused_turn and self.mode == 'redag_full' and not self.draft_first_turn:
                    inv = self.active_source_invariants(dag)
                    if inv:
                        eff_g = self.effective_graph(dag)
                        frontier_set, repair_set = set(frontier), set(self.repair_queue)
                        blocked_edits = [
                            u for u in dag.nodes
                            if u not in inv and state.scores.get(u, 0.0) < dag.nodes[u].threshold and u not in frontier_set
                        ]
                        unblocking = (
                            set().union(*(graph_ancestors(eff_g, u) for u in blocked_edits))
                            if blocked_edits else set()
                        )
                        frontier = [nid for nid in frontier if nid not in inv or nid in repair_set or nid in unblocking]
                if not frontier and not turn_targets:
                    reason = 'satisfied' if all(state.scores[n] >= dag.nodes[n].threshold for n in dag.nodes) else 'blocked_frontier'
                    break
                eff_dag = self.effective_graph(dag)
                repair_rank = {nid: idx for idx, nid in enumerate(self.repair_queue)}
                if self.mode == 'redag_full' and (self.cascade_priority or self.repair_queue):
                    frontier.sort(key=lambda nid: (
                        failures.get(nid, 0) >= self.retry_limit,
                        repair_rank.get(nid, 10000),
                        -(unlocked_satisfied_descendants(eff_dag, state.scores, nid) if self.cascade_priority else 0),
                        self.turn_priority and nid not in self.turn_nodes,
                        -dag.nodes[nid].weight * (1.0 - state.scores.get(nid, 0.0)),
                    ))
                else:
                    frontier.sort(
                        key=lambda nid: (failures.get(nid, 0) >= self.retry_limit, self.turn_priority and nid not in self.turn_nodes)
                    )
                group_calls, group_renders = self.verifier.call_counter, self.generation_counter
                if turn_targets:
                    candidates = self.turn_candidates(state, dag, editor, turn_targets, seed + step * group_size, previous=retry)
                else:
                    candidates = self.sample_group(state, dag, controller, editor, frontier[0], group_size, seed + step * group_size)
                if not candidates:
                    reason = 'execution_budget'
                    break
                chosen = dict(max(candidates, key=lambda c: c['reward']), candidate_group=candidates, branch_parent=current + history_offset)
                chosen['generation_calls'] = self.generation_counter - group_renders
                chosen['verifier_calls'] = self.verifier.call_counter - group_calls
                records.append(chosen)
                state = chosen['delivered_state']
                if chosen['accepted']:
                    failures.clear()
                    if chosen.get('repair_drop'):
                        drop_id = chosen['repair_drop']
                        self.repaired_in_turn.add(drop_id)
                        if drop_id not in self.repair_queue:
                            self.repair_queue.append(drop_id)
                elif not turn_targets:
                    failures[frontier[0]] = failures.get(frontier[0], 0) + 1
                if chosen['accepted'] and self.grounder is not None:
                    dag = self.grounder.ground(state.image, dag, chosen['affected'])
                if self.mode != 'flat_full':
                    nodes.append(dict(state=state, dag=deepcopy(dag), parent=current, grounding=self.grounding_state(), accepted=chosen['accepted']))
                    chosen['branch_id'] = len(nodes) - 1 + history_offset
                    if chosen['accepted']:
                        current = len(nodes) - 1
                    elif self.mode == 'tree_rollback_only' and nodes[current]['parent'] is not None:
                        current = nodes[current]['parent']
                        state, dag = nodes[current]['state'], deepcopy(nodes[current]['dag'])
                        self.restore_grounding(nodes[current]['grounding'])
                        chosen['backtracked_to'] = current + history_offset
                chosen['returned_state'] = state
            if self.mode == 'tree_rollback_only':
                best = max(nodes, key=lambda n: sum(n['dag'].nodes[v].weight * q for v, q in n['state'].scores.items()))
                state, dag = best['state'], deepcopy(best['dag'])
                self.restore_grounding(best['grounding'])
            elif self.monotonic_turn and self.mode == 'redag_full':
                best = max(enumerate(nodes), key=lambda item: (
                    consistent_count(self.effective_graph(item[1]['dag']), item[1]['state'].scores),
                    sum(item[1]['dag'].nodes[v].weight * q for v, q in item[1]['state'].scores.items()),
                    item[0],
                ))[1]
                state, dag = best['state'], deepcopy(best['dag'])
                self.restore_grounding(best['grounding'])
                if records:
                    records[-1]['returned_state'] = state
            if self.mode != 'flat_full':
                for node in nodes:
                    if node['parent'] is not None:
                        node['parent'] += history_offset
                self.history.extend(nodes)
            return state, dag, records
        finally:
            self.last_run = dict(generation_calls=self.generation_counter - renders, verifier_calls=self.verifier.call_counter - calls, stop_reason=reason, unknown_node_ids=sorted(self.unknown_nodes))
            self.verifier.call_limit, self.render_limit = old_limits
            if own_origin:
                self.cost_origin = None

    def run_conversation(self, image, prompts, planner, controller, editor, max_steps=10, seed=42, group_size=1):
        from redag.models.planner import preserve_previous_conditions
        dag, state, records = None, None, []
        self.node_turn_request = {}
        self.source_invariants = set()
        self.turn_count = len(prompts)
        for turn, request in enumerate(prompts):
            request = {'user_prompt': request} if isinstance(request, str) else request
            previous, superseded = dag, request.get('superseded_ids', request.get('supersedes'))
            self.turn_request, self.turn_index = str(request.get('user_prompt') or ''), turn
            if 'graph' in request or 'dag' in request:
                dag = preserve_previous_conditions(request.get('graph', request.get('dag')), previous, superseded)
            else:
                if planner is None:
                    raise ValueError('Text-only conversation turns require a planner')
                kwargs = {'superseded_ids': superseded} if superseded is not None else {}
                dag = planner.plan(request['user_prompt'], previous_dag=previous, **kwargs)
                dag = preserve_previous_conditions(dag, previous, superseded if superseded is not None else getattr(planner, 'last_superseded_ids', []))
            raw_new = set(dag.nodes) - (set(previous.nodes) if previous is not None else set())
            if turn == 0 and not self.draft_first_turn:
                self.source_invariants = {
                    nid for nid in raw_new
                    if is_source_invariant_id(nid) and all(is_source_invariant_id(a) for a in graph_ancestors(dag, nid))
                }
            elif not self.draft_first_turn:
                self.source_invariants = {
                    nid for nid in (self.source_invariants - set(superseded or ()))
                    if nid in dag.nodes and all(is_source_invariant_id(a) for a in graph_ancestors(dag, nid))
                }
            inv_now = self.active_source_invariants(dag) if (self.focused_turn and self.mode == 'redag_full') else set()
            self.turn_nodes = (raw_new - inv_now) if inv_now else raw_new
            for nid in list(self.node_turn_request):
                if nid not in dag.nodes:
                    del self.node_turn_request[nid]
            for nid in raw_new | (set(superseded or ()) & set(dag.nodes)):
                is_src_inv = not self.draft_first_turn and nid in self.source_invariants
                self.node_turn_request[nid] = '' if is_src_inv else self.turn_request
            if state is None:
                state, dag = self.initialize(image, dag)
            else:
                changed = {n for n in dag.nodes if n not in previous.nodes or asdict(dag.nodes[n]) != asdict(previous.nodes[n])}
                changed.update(c for p, c in set(map(tuple, dag.comp_edges + dag.supp_edges)) ^ set(map(tuple, previous.comp_edges + previous.supp_edges)) if c in dag.nodes)
                changed |= get_dependency_descendants(dag, changed)
                self.unknown_nodes.intersection_update(dag.nodes)
                changed |= self.unknown_nodes
                if self.grounder is not None:
                    dag = self.grounder.ground(state.image, dag, changed)
                initial = {n: state.scores.get(n, 0.0) for n in dag.nodes}
                scores = self.verify_current(state.image, dag, initial, changed)
                state = CanvasState(state.image, scores, {n for n in scores if n not in self.unknown_nodes and scores[n] >= dag.nodes[n].threshold}, state.step_idx)
            if self.on_turn_start is not None:
                self.on_turn_start(self, state, dag)
            state, dag, steps = self.run_session(state, dag, controller, editor, max_steps, seed + turn * max_steps * group_size, group_size)
            for step in steps:
                step['turn'] = turn
            records.extend(steps)
        return state, dag, records


def state_from_record(record, harness):
    from redag.models.planner import dag_from_dict
    from redag.verify.grounder import load_image
    dag = dag_from_dict(record.get('graph', record.get('dag')))
    image = record['image'] if torch.is_tensor(record['image']) else load_image(record['image'])
    if 'scores' not in record:
        return harness.initialize(image, dag)
    if set(record['scores']) != set(dag.nodes):
        raise ValueError('Parent scores must cover every graph node exactly')
    if harness.grounder is not None:
        if 'detected_counts' in record:
            harness.grounder.counts = dict(record['detected_counts'])
            harness.grounder.masks = {}
        else:
            dag = harness.grounder.ground(image, dag)
    harness.unknown_nodes = set()
    scores = dict(record['scores'])
    return CanvasState(image, scores, {n for n in scores if scores[n] >= dag.nodes[n].threshold}, record.get('step_idx', 0)), dag


def run_record(record, harness, controller, editor, max_steps=10, seed=42, group_size=1, planner=None):
    from redag.verify.grounder import load_image
    calls, renders = harness.verifier.call_counter, harness.generation_counter
    limits = harness.verifier.call_limit, harness.render_limit
    harness.history, harness.last_run, harness.unknown_nodes = [], {}, set()
    if not harness.budget_per_turn:
        harness.cost_origin = (calls, renders)
        verification_limits = [x for x in (limits[0], calls + harness.verification_budget if harness.verification_budget is not None else None) if x is not None]
        generation_limits = [x for x in (limits[1], renders + harness.generation_budget if harness.generation_budget is not None else None) if x is not None]
        harness.verifier.call_limit = min(verification_limits, default=None)
        harness.render_limit = min(generation_limits, default=None)
    harness.turn_request, harness.turn_nodes, harness.turn_index = '', set(), 0
    harness.node_turn_request = {}
    harness.source_invariants = None
    harness.global_failures, harness.local_failures, harness.turn_count = {}, {}, 1
    harness.draft_first_turn = record.get('kind') == 't2i'
    try:
        if 'turns' in record:
            if not record['turns']:
                raise ValueError('A conversation requires at least one turn')
            image = record['image'] if torch.is_tensor(record['image']) else load_image(record['image'])
            result = harness.run_conversation(image, record['turns'], planner, controller, editor, max_steps, seed, group_size)
        else:
            state, dag = state_from_record(record, harness)
            result = harness.run_session(state, dag, controller, editor, max_steps, seed, group_size)
        return result
    finally:
        harness.last_run.update(generation_calls=harness.generation_counter - renders, verifier_calls=harness.verifier.call_counter - calls)
        harness.verifier.call_limit, harness.render_limit = limits
        harness.cost_origin = None
        harness.node_turn_request = {}
        harness.source_invariants = None


def add_runtime_arguments(parser):
    parser.add_argument('--controller_model', default='Qwen/Qwen3.5-9B')
    parser.add_argument('--editor_model', default='black-forest-labs/FLUX.2-klein-base-9B')
    parser.add_argument('--verifier_model', default='Qwen/Qwen3-VL-8B-Instruct')
    parser.add_argument('--detector_model', default='IDEA-Research/grounding-dino-base')
    parser.add_argument('--sam_model', default='facebook/sam2.1-hiera-large')
    parser.add_argument('--controller_adapter')
    parser.add_argument('--editor_adapter')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--editor_device')
    parser.add_argument('--verifier_device')
    parser.add_argument('--planner_model', default='Qwen/Qwen3.8-27B')
    parser.add_argument('--planner_adapter')
    parser.add_argument('--planner_device')
    parser.add_argument('--num_denoise_steps', type=int, help='editor sampling steps (default: config value, 50)')
    parser.add_argument('--editor_work_side', type=int, default=0)
    parser.add_argument('--editor_context_pad', type=float, default=0.0)
    parser.add_argument('--protect_masks', action='store_true')
    parser.add_argument('--closure_instructions', action='store_true')
    parser.add_argument('--confirm_drops', action='store_true')
    parser.add_argument('--soft_gating', action='store_true')
    parser.add_argument('--preserve_instructions', action='store_true')
    parser.add_argument('--dag_box', action='store_true')
    parser.add_argument('--edge_policy', choices=('full', 'strong'), default='full')
    parser.add_argument('--counterfactual_credit', action='store_true')
    parser.add_argument('--acceptance_bonus', type=float, default=0.0)
    parser.add_argument('--online_planner', action='store_true')
    parser.add_argument('--editor_kind', choices=('flux', 'pipeline'), default='flux')
    parser.add_argument('--editor_offload', action='store_true')
    parser.add_argument('--editor_cfg', type=float)
    parser.add_argument('--scope_routing', action='store_true')
    parser.add_argument('--global_types', default=','.join(GLOBAL_ROUTE_TYPES))
    parser.add_argument('--global_gate', choices=('joint', 'mean', 'net', 'net2'), default='joint')
    parser.add_argument('--local_gate', choices=('legacy', 'net2'), default='legacy')
    parser.add_argument('--global_retries', type=int, default=0)
    parser.add_argument('--user_text', action='store_true')
    parser.add_argument('--turn_pass', action='store_true')
    parser.add_argument('--turn_gate', choices=('joint', 'mean', 'net', 'net2'), default='joint')
    parser.add_argument('--turn_retries', type=int, default=0)
    parser.add_argument('--turn_priority', action='store_true')
    parser.add_argument('--turn_localize', action='store_true')
    parser.add_argument('--turn_protect', action='store_true')
    parser.add_argument('--strategy_policy')
    parser.add_argument('--clean_instruction', action='store_true')
    parser.add_argument('--single_stage_verify', action='store_true')
    parser.add_argument('--focused_turn', action='store_true')
    parser.add_argument('--smart_turn_pass', action='store_true')
    parser.add_argument('--cascade_priority', action='store_true')
    parser.add_argument('--repair_gate', action='store_true')
    parser.add_argument('--monotonic_turn', action='store_true')
    parser.add_argument('--candidate_k', type=int, default=1)
    return parser


def load_runtime(args, trainable_controller=False, trainable_editor=False):
    from redag.models.controller import Controller
    from redag.models.editor import FluxEditor, PipelineEditor
    from redag.verify.grounder import Grounder
    from redag.verify.verifier import TwoStageVerifier, load_vlm_scorer
    dtype = torch.float32 if args.device == 'cpu' else torch.bfloat16
    controller = Controller.from_pretrained(args.controller_model, args.device, dtype, args.controller_adapter, trainable_controller)
    controller.clean_instruction = bool(getattr(args, 'clean_instruction', False))
    edit_device, verify_device = args.editor_device or args.device, args.verifier_device or args.device
    if getattr(args, 'editor_kind', 'flux') == 'pipeline':
        editor = PipelineEditor.from_pretrained(args.editor_model, edit_device, torch.bfloat16, getattr(args, 'editor_offload', False))
    else:
        editor = FluxEditor.from_pretrained(args.editor_model, edit_device, torch.float32 if edit_device == 'cpu' else torch.bfloat16, args.editor_adapter, trainable_editor)
    if getattr(args, 'num_denoise_steps', None):
        editor.num_steps = args.num_denoise_steps
    if getattr(args, 'editor_cfg', None) is not None:
        editor.cfg_scale = args.editor_cfg
    editor.work_side = getattr(args, 'editor_work_side', 0) or None
    editor.context_pad = float(getattr(args, 'editor_context_pad', 0.0) or 0.0)
    verifier = TwoStageVerifier(load_vlm_scorer(args.verifier_model, verify_device, torch.float32 if verify_device == 'cpu' else torch.bfloat16))
    grounder = Grounder.from_pretrained(args.detector_model, args.sam_model, verify_device)
    harness = ReDAGHarness(verifier, grounder)
    harness.protect_masks = getattr(args, 'protect_masks', False)
    for flag in ('closure_instructions', 'confirm_drops', 'soft_gating', 'preserve_instructions', 'dag_box'):
        setattr(harness, flag, getattr(args, flag, False))
    harness.edge_policy = getattr(args, 'edge_policy', 'full')
    harness.counterfactual_credit = getattr(args, 'counterfactual_credit', False)
    harness.acceptance_bonus = float(getattr(args, 'acceptance_bonus', 0.0))
    configure_routing(harness, args)
    return harness, controller, editor


def configure_routing(harness, args):
    harness.scope_routing = bool(getattr(args, 'scope_routing', False))
    types = tuple(t.strip() for t in str(getattr(args, 'global_types', ','.join(GLOBAL_ROUTE_TYPES))).split(',') if t.strip())
    unknown = set(types) - {'background', 'global', 'entity_global', 'relation_wide', 'count', 'prohibition'}
    if unknown:
        raise ValueError(f'Unknown global route types: {sorted(unknown)}')
    harness.global_types = types
    harness.global_gate = getattr(args, 'global_gate', 'joint')
    harness.global_retries = int(getattr(args, 'global_retries', 0) or 0)
    harness.user_text = bool(getattr(args, 'user_text', False))
    harness.turn_pass = bool(getattr(args, 'turn_pass', False))
    harness.turn_gate = getattr(args, 'turn_gate', 'joint')
    harness.turn_retries = int(getattr(args, 'turn_retries', 0) or 0)
    harness.turn_priority = bool(getattr(args, 'turn_priority', False))
    harness.local_gate = getattr(args, 'local_gate', 'legacy') or 'legacy'
    harness.turn_localize = bool(getattr(args, 'turn_localize', False))
    harness.turn_protect = bool(getattr(args, 'turn_protect', False))
    harness.single_stage_verify = bool(getattr(args, 'single_stage_verify', False))
    harness.focused_turn = bool(getattr(args, 'focused_turn', False))
    harness.smart_turn_pass = bool(getattr(args, 'smart_turn_pass', False))
    harness.cascade_priority = bool(getattr(args, 'cascade_priority', False))
    harness.repair_gate = bool(getattr(args, 'repair_gate', False))
    harness.monotonic_turn = bool(getattr(args, 'monotonic_turn', False))
    harness.candidate_k = max(1, int(getattr(args, 'candidate_k', 1) or 1))
    if getattr(args, 'strategy_policy', None):
        from redag.learn.strategy import StrategyPolicy
        harness.strategy = StrategyPolicy.load(args.strategy_policy, sample=False)
    return harness


def validate_dataset(records, experiment, diagnostic, require_graphs=False):
    from collections import defaultdict
    from redag.models.planner import preserve_previous_conditions
    if not records:
        raise ValueError('Evaluation data must not be empty')
    if diagnostic or experiment == 2:
        return
    counts, unplanned = defaultdict(int), 0
    for record in records:
        graph = None
        for turn in record.get('turns', [record]):
            if isinstance(turn, str) or not ('graph' in turn or 'dag' in turn):
                graph = None
                break
            graph = preserve_previous_conditions(turn.get('graph', turn.get('dag')), graph, turn.get('superseded_ids', turn.get('supersedes')))
        if graph is None:
            unplanned += 1
            continue
        counts[record.get('k', len(graph.nodes)) if record.get('revisions') else len(graph.nodes)] += 1
    if unplanned:
        if require_graphs:
            raise ValueError(f'{unplanned} sessions still have turns without a condition graph')
        return
    if set(counts) != {10, 20, 30} or any(n < 100 for n in counts.values()):
        raise ValueError('Main evaluations require at least 100 sessions for each K=10/20/30')
