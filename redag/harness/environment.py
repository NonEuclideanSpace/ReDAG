from copy import deepcopy
from dataclasses import asdict
import torch
from redag.config import ReDAGConfig
from redag.graph.schema import CanvasState, ConditionDAG
from redag.graph.dag import compute_active_frontier, compute_affected_subgraph, compute_spatial_subgraph, get_dependency_descendants, semantic_support
from redag.edit.compositor import check_local_acceptance, paste_window_strictly_local
from redag.learn.credit import compute_subgraph_reward


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
        self.generation_counter, self.last_run, self.unknown_nodes = 0, {}, set()

    def effective_graph(self, dag):
        if self.mode not in ('redag_full', 'flat_full', 'tree_rollback_only', 'spatial_local', 'spatial_no_gates'):
            raise ValueError('Unknown environment mode')
        if self.mode in ('redag_full', 'spatial_local'):
            return dag
        nodes = deepcopy(dag.nodes)
        for parent, child in dag.comp_edges + dag.supp_edges:
            nodes[child].depends_on = sorted(set(nodes[child].depends_on) | {parent})
        return ConditionDAG(nodes)

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
        self.verifier.two_stage = self.mode != 'flat_full'
        self.unknown_nodes = set()
        scores = self.verify_current(image, dag, dict.fromkeys(dag.nodes, 0.0), set(dag.nodes))
        return CanvasState(image, scores, {n for n in dag.nodes if n not in self.unknown_nodes and scores[n] >= dag.nodes[n].threshold}), dag

    def evaluate_candidate_action(self, parent_state, dag, target_node_id, action, edited_crop):
        calls, start = self.verifier.call_counter, len(self.verifier.records)
        graph = self.effective_graph(dag)
        self.verifier.two_stage = self.mode != 'flat_full'
        candidate = paste_window_strictly_local(parent_state.image, edited_crop, action.box, self.feather_px)
        if self.mode == 'redag_full':
            affected = compute_affected_subgraph(dag, action, target_node_id)
        elif self.mode in ('spatial_local', 'spatial_no_gates'):
            affected = compute_spatial_subgraph(dag, action, target_node_id)
        else:
            affected = set(graph.nodes)
        uncertain = {n for n, node in dag.nodes.items() if node.scope != 'global' and node.node_type not in ('background', 'count', 'prohibition') and semantic_support(dag, n) is None}
        raw_scores = self.verifier.verify_subgraph_after_edit(candidate, graph, parent_state.scores, affected)
        accepted = check_local_acceptance(graph, target_node_id, parent_state.scores, raw_scores, parent_state.protected_set, affected,
                                         self.config.min_target_gain, self.config.max_protected_drop)
        delivered_image = candidate if accepted else parent_state.image
        delivered_scores = raw_scores if accepted else dict(parent_state.scores)
        reward = compute_subgraph_reward(graph, parent_state.scores, delivered_scores, parent_state.protected_set, affected, self.step_cost, self.lam)
        protected = {nid for nid, score in delivered_scores.items() if score >= dag.nodes[nid].threshold}
        state = CanvasState(delivered_image, delivered_scores, protected, parent_state.step_idx + 1)
        self.last_candidate = dict(dag=deepcopy(dag), parent_state=parent_state,
            candidate_state=CanvasState(candidate, raw_scores, {n for n in raw_scores if raw_scores[n] >= dag.nodes[n].threshold}),
            delivered_state=state, raw_scores=raw_scores, affected=affected, target_node_id=target_node_id, action=action,
            accepted=accepted, reward=reward, coverage_uncertain_nodes=sorted(uncertain),
            stage_scores=self.verifier.records[start:], verifier_calls=self.verifier.call_counter - calls)
        return state, reward, accepted, affected

    def sample_group(self, state, dag, controller, editor, target_node_id, group_size=6, seed=42):
        candidates, grounding = [], self.grounding_state()
        try:
            for index in range(group_size):
                if self.render_limit is not None and self.generation_counter >= self.render_limit:
                    break
                if self.verifier.call_limit is not None and self.verifier.call_counter >= self.verifier.call_limit:
                    break
                self.restore_grounding(grounding)
                graph = deepcopy(dag)
                sample = controller.sample(state, graph, target_node_id, seed=seed + index, temperature=0.8,
                    detected_count=self.grounder.counts.get(dag.nodes[target_node_id].entity_phrase, 1) if self.grounder else 1)
                self.generation_counter += 1
                crop = editor.render(state.image, sample['action'], seed=42)
                try:
                    self.evaluate_candidate_action(state, graph, target_node_id, sample['action'], crop)
                except RuntimeError as error:
                    if str(error) != 'Verifier call budget exhausted':
                        raise
                    break
                candidates.append(dict(self.last_candidate, sample=sample))
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
        reason = 'step_budget'
        try:
            for step in range(max_steps):
                if self.unknown_nodes:
                    reason = 'execution_budget'
                    break
                frontier = compute_active_frontier(self.effective_graph(dag), state.scores, self.config.gate_tau)
                if not frontier:
                    reason = 'satisfied' if all(state.scores[n] >= dag.nodes[n].threshold for n in dag.nodes) else 'blocked_frontier'
                    break
                group_calls, group_renders = self.verifier.call_counter, self.generation_counter
                candidates = self.sample_group(state, dag, controller, editor, frontier[0], group_size, seed + step * group_size)
                if not candidates:
                    reason = 'execution_budget'
                    break
                chosen = dict(max(candidates, key=lambda c: c['reward']), candidate_group=candidates, branch_parent=current + history_offset)
                chosen['generation_calls'] = self.generation_counter - group_renders
                chosen['verifier_calls'] = self.verifier.call_counter - group_calls
                records.append(chosen)
                state = chosen['delivered_state']
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
            if self.mode != 'flat_full':
                for node in nodes:
                    if node['parent'] is not None:
                        node['parent'] += history_offset
                self.history.extend(nodes)
            return state, dag, records
        finally:
            self.last_run = dict(generation_calls=self.generation_counter - renders, verifier_calls=self.verifier.call_counter - calls, stop_reason=reason, unknown_node_ids=sorted(self.unknown_nodes))
            self.verifier.call_limit, self.render_limit = old_limits

    def run_conversation(self, image, prompts, planner, controller, editor, max_steps=10, seed=42, group_size=1):
        from redag.models.planner import preserve_previous_conditions
        dag, state, records = None, None, []
        for turn, request in enumerate(prompts):
            request = {'user_prompt': request} if isinstance(request, str) else request
            previous, superseded = dag, request.get('superseded_ids', request.get('supersedes'))
            if 'graph' in request or 'dag' in request:
                dag = preserve_previous_conditions(request.get('graph', request.get('dag')), previous, superseded)
            else:
                if planner is None:
                    raise ValueError('Text-only conversation turns require a planner')
                kwargs = {'superseded_ids': superseded} if superseded is not None else {}
                dag = planner.plan(request['user_prompt'], previous_dag=previous, **kwargs)
                dag = preserve_previous_conditions(dag, previous, superseded if superseded is not None else getattr(planner, 'last_superseded_ids', []))
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
    verification_limits = [x for x in (limits[0], calls + harness.verification_budget if harness.verification_budget is not None else None) if x is not None]
    generation_limits = [x for x in (limits[1], renders + harness.generation_budget if harness.generation_budget is not None else None) if x is not None]
    harness.verifier.call_limit = min(verification_limits, default=None)
    harness.render_limit = min(generation_limits, default=None)
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
    return parser


def load_runtime(args, trainable_controller=False, trainable_editor=False):
    from redag.models.controller import Controller
    from redag.models.editor import FluxEditor
    from redag.verify.grounder import Grounder
    from redag.verify.verifier import TwoStageVerifier, load_vlm_scorer
    dtype = torch.float32 if args.device == 'cpu' else torch.bfloat16
    controller = Controller.from_pretrained(args.controller_model, args.device, dtype, args.controller_adapter, trainable_controller)
    edit_device, verify_device = args.editor_device or args.device, args.verifier_device or args.device
    editor = FluxEditor.from_pretrained(args.editor_model, edit_device, torch.float32 if edit_device == 'cpu' else torch.bfloat16, args.editor_adapter, trainable_editor)
    if getattr(args, 'num_denoise_steps', None):
        editor.num_steps = args.num_denoise_steps
    verifier = TwoStageVerifier(load_vlm_scorer(args.verifier_model, verify_device, torch.float32 if verify_device == 'cpu' else torch.bfloat16))
    grounder = Grounder.from_pretrained(args.detector_model, args.sam_model, verify_device)
    return ReDAGHarness(verifier, grounder), controller, editor


def validate_dataset(records, experiment, diagnostic):
    from collections import defaultdict
    from redag.models.planner import preserve_previous_conditions
    if not records:
        raise ValueError('Evaluation data must not be empty')
    if diagnostic or experiment == 2:
        return
    counts = defaultdict(int)
    for record in records:
        graph = None
        for turn in record.get('turns', [record]):
            if isinstance(turn, str) or not ('graph' in turn or 'dag' in turn):
                graph = None
                break
            graph = preserve_previous_conditions(turn.get('graph', turn.get('dag')), graph, turn.get('superseded_ids', turn.get('supersedes')))
        if graph is None:
            return
        counts[len(graph.nodes)] += 1
    if set(counts) != {10, 20, 30} or any(n < 100 for n in counts.values()):
        raise ValueError('Main evaluations require at least 100 sessions for each K=10/20/30')
