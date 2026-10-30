from copy import deepcopy
from dataclasses import asdict
import torch
from redag.config import ReDAGConfig
from redag.graph.schema import CanvasState, ConditionDAG
from redag.graph.dag import compute_active_frontier, compute_affected_subgraph
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
        self.mode = mode
        self.history = []
        self.last_candidate = None
        self.verification_budget = None

    def effective_graph(self, dag):
        if self.mode not in ('redag_full', 'flat_full', 'tree_rollback_only'):
            raise ValueError('Unknown environment mode')
        return dag if self.mode == 'redag_full' else ConditionDAG(deepcopy(dag.nodes))

    def initialize(self, image, dag):
        if self.grounder is not None:
            dag = self.grounder.ground(image, dag)
        self.verifier.two_stage = self.mode != 'flat_full'
        scores = self.verifier.verify_subgraph_after_edit(image, self.effective_graph(dag), dict.fromkeys(dag.nodes, 0.0), set(dag.nodes))
        state = CanvasState(image, scores, {n for n in dag.nodes if scores[n] >= dag.nodes[n].threshold})
        return state, dag

    def evaluate_candidate_action(self, parent_state, dag, target_node_id, action, edited_crop):
        calls = self.verifier.call_counter
        start = len(self.verifier.records)
        graph = self.effective_graph(dag)
        self.verifier.two_stage = self.mode != 'flat_full'
        candidate = paste_window_strictly_local(parent_state.image, edited_crop, action.box, self.feather_px)
        affected = compute_affected_subgraph(graph, action, target_node_id) if self.mode == 'redag_full' else set(graph.nodes)
        raw_scores = self.verifier.verify_subgraph_after_edit(candidate, graph, parent_state.scores, affected)
        accepted = check_local_acceptance(graph, target_node_id, parent_state.scores, raw_scores, parent_state.protected_set, affected,
                                         self.config.min_target_gain, self.config.max_protected_drop)
        delivered_image = candidate if accepted else parent_state.image
        delivered_scores = raw_scores if accepted else dict(parent_state.scores)
        reward = compute_subgraph_reward(graph, parent_state.scores, delivered_scores, parent_state.protected_set, affected,
                                         self.step_cost, self.lam)
        protected = {nid for nid, score in delivered_scores.items() if score >= dag.nodes[nid].threshold}
        state = CanvasState(delivered_image, delivered_scores, protected, parent_state.step_idx + 1)
        self.last_candidate = dict(dag=deepcopy(dag), parent_state=parent_state, candidate_state=CanvasState(candidate, raw_scores, {n for n in raw_scores if raw_scores[n] >= dag.nodes[n].threshold}),
                                   delivered_state=state, raw_scores=raw_scores, affected=affected, target_node_id=target_node_id,
                                   action=action, accepted=accepted, reward=reward, stage_scores=self.verifier.records[start:],
                                   verifier_calls=self.verifier.call_counter - calls)
        return state, reward, accepted, affected

    def sample_group(self, state, dag, controller, editor, target_node_id, group_size=6, seed=42):
        candidates = []
        for index in range(group_size):
            sample = controller.sample(state, dag, target_node_id, seed=seed + index, temperature=0.8,
                                       detected_count=self.grounder.counts.get(dag.nodes[target_node_id].entity_phrase, 1) if self.grounder else 1)
            crop = editor.render(state.image, sample['action'], seed=42)
            self.evaluate_candidate_action(state, dag, target_node_id, sample['action'], crop)
            candidates.append(dict(self.last_candidate, sample=sample))
        return candidates

    def run_session(self, state, dag, controller, editor, max_steps=10, seed=42, group_size=1):
        records = []
        start_calls = self.verifier.call_counter
        for step in range(max_steps):
            graph = self.effective_graph(dag)
            frontier = compute_active_frontier(graph, state.scores, self.config.gate_tau)
            if not frontier:
                break
            worst_calls = 2 * len(dag.nodes) * group_size
            if self.verification_budget is not None and self.verifier.call_counter - start_calls + worst_calls > self.verification_budget:
                break
            candidates = self.sample_group(state, dag, controller, editor, frontier[0], group_size, seed + step * group_size)
            chosen = max(candidates, key=lambda c: c['reward'])
            chosen['candidate_group'] = [dict(c) for c in candidates]
            records.append(chosen)
            if self.mode != "flat_full":
                self.history.append({"state": state, "dag": deepcopy(dag), "parent": len(self.history) - 1})
            state = chosen['delivered_state']
            if chosen['accepted'] and self.grounder is not None:
                dag = self.grounder.ground(state.image, dag, chosen['affected'])
        return state, dag, records

    def run_conversation(self, image, prompts, planner, controller, editor, max_steps=10, seed=42):
        dag, state, records = None, None, []
        for turn, prompt in enumerate(prompts):
            previous = dag
            dag = planner.plan(prompt, previous_dag=dag)
            if state is None:
                state, dag = self.initialize(image, dag)
            else:
                changed = {n for n in dag.nodes if n not in previous.nodes or asdict(dag.nodes[n]) != asdict(previous.nodes[n])}
                changed.update(child for parent, child in set(map(tuple, dag.comp_edges)) ^ set(map(tuple, previous.comp_edges)) if child in dag.nodes)
                from redag.graph.dag import get_comp_descendants
                changed |= get_comp_descendants(dag, changed)
                if self.grounder is not None:
                    dag = self.grounder.ground(state.image, dag, changed)
                initial = {n: state.scores.get(n, 0.0) for n in dag.nodes}
                scores = self.verifier.verify_subgraph_after_edit(state.image, self.effective_graph(dag), initial, changed)
                state = CanvasState(state.image, scores, {n for n in scores if scores[n] >= dag.nodes[n].threshold}, state.step_idx)
            state, dag, steps = self.run_session(state, dag, controller, editor, max_steps, seed + turn * max_steps)
            records.extend(steps)
        return state, dag, records


def state_from_record(record, harness):
    from redag.models.planner import dag_from_dict
    from redag.verify.grounder import load_image
    dag = dag_from_dict(record.get('graph', record.get('dag')))
    image = record['image']
    if not torch.is_tensor(image):
        image = load_image(image)
    if 'scores' not in record:
        return harness.initialize(image, dag)
    if set(record['scores']) != set(dag.nodes):
        raise ValueError('Parent scores must cover every graph node exactly')
    if harness.grounder is not None:
        if 'detected_counts' in record:
            harness.grounder.counts = dict(record['detected_counts'])
        else:
            dag = harness.grounder.ground(image, dag)
    scores = dict(record['scores'])
    state = CanvasState(image, scores, {n for n in scores if scores[n] >= dag.nodes[n].threshold}, record.get('step_idx', 0))
    return state, dag


def add_runtime_arguments(parser):
    parser.add_argument('--controller_model', default='Qwen/Qwen3.5-9B')
    parser.add_argument('--editor_model', default='black-forest-labs/FLUX.2-klein-9B')
    parser.add_argument('--verifier_model', default='Qwen/Qwen3-VL-8B-Instruct')
    parser.add_argument('--detector_model', default='IDEA-Research/grounding-dino-base')
    parser.add_argument('--sam_model', default='facebook/sam2.1-hiera-large')
    parser.add_argument('--controller_adapter')
    parser.add_argument('--editor_adapter')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--editor_device')
    parser.add_argument('--verifier_device')
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
    verifier = TwoStageVerifier(load_vlm_scorer(args.verifier_model, verify_device, torch.float32 if verify_device == 'cpu' else torch.bfloat16))
    grounder = Grounder.from_pretrained(args.detector_model, args.sam_model, verify_device)
    return ReDAGHarness(verifier, grounder), controller, editor
