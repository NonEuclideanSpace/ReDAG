import argparse
import copy
import gc
import json
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from redag.harness.environment import add_runtime_arguments, load_runtime, state_from_record
from redag.learn.credit import compute_pairwise_advantages, compute_subgraph_reward
from redag.learn.sft_trainer import read_records
from redag.verify.verifier import load_vlm_scorer

def ratio(a, b):
    return a / b if b else None

def audit(image, dag, scorer):
    return {nid: float(scorer(image, node.question)) for nid, node in dag.nodes.items()}

def summarize(state, dag, trace, scorer, min_gain=0.05, max_drop=0.08):
    final = audit(state.image, dag, scorer)
    totals = dict(protected=0, regressions=0, good=0, rejected_good=0, stage1_alerts=0, stage2_alerts=0, true1=0, true2=0)
    for step in trace:
        graph = step.get("dag", dag)
        before = audit(step["parent_state"].image, graph, scorer)
        candidate = audit(step["candidate_state"].image, graph, scorer)
        delivered = candidate if step["accepted"] else before
        protected, target = step["parent_state"].protected_set, step["target_node_id"]
        totals["protected"] += len(protected)
        totals["regressions"] += sum(delivered[v] < before[v] - max_drop for v in protected)
        good = (candidate[target] - before[target] >= min_gain or candidate[target] >= graph.nodes[target].threshold) and all(candidate[v] >= before[v] - max_drop for v in protected)
        totals["good"] += good
        totals["rejected_good"] += good and not step["accepted"]
        for record in step["stage_scores"]:
            nid = record["node_id"]
            for number, value in ((1, record["stage1"]), (2, record["stage2"] if record["stage2"] is not None else record["stage1"])):
                alert = value < graph.nodes[nid].threshold
                totals[f"stage{number}_alerts"] += alert
                totals[f"true{number}"] += alert and candidate[nid] < graph.nodes[nid].threshold
    return {"all_satisfied": all(final[nid] >= n.threshold for nid, n in dag.nodes.items()), "steps": len(trace), "verifier_calls": sum(x["verifier_calls"] for x in trace), **totals}

def aggregate(rows):
    counts = {key: sum(row[key] for row in rows) for key in rows[0]}
    return {"sessions": len(rows), "all_satisfied_rate": counts["all_satisfied"] / len(rows), "protected_regression_rate": ratio(counts["regressions"], counts["protected"]), "stage1_vs_stage2_precision": {"stage1": ratio(counts["true1"], counts["stage1_alerts"]), "stage2": ratio(counts["true2"], counts["stage2_alerts"])}, "good_edit_false_reject_rate": ratio(counts["rejected_good"], counts["good"]), "counts": counts}

def credit_experiment(args, records, harness, controller, editor, scorer):
    rng, curves, leakage = random.Random(args.seed), [], []
    methods = ("terminal_broadcast", "step_full_table", "subgraph_credit_ours")
    for index, record in enumerate(records):
        parent, dag = state_from_record(record, harness)
        _, _, trace = harness.run_session(parent, dag, controller, editor, args.max_steps, seed=args.seed + index, group_size=6)
        for step_index, step in enumerate(trace):
            groups = {method: [] for method in methods}
            graph = step.get("dag", dag)
            for candidate in step["candidate_group"]:
                before, delivered = candidate["parent_state"], candidate["delivered_state"]
                full_old = {nid: harness.verifier.verify_single_node_two_stage(before.image, node) for nid, node in graph.nodes.items()}
                full_new = {nid: harness.verifier.verify_single_node_two_stage(delivered.image, node) for nid, node in graph.nodes.items()}
                groups["step_full_table"].append(compute_subgraph_reward(graph, full_old, full_new, before.protected_set, set(graph.nodes), step_cost=harness.step_cost, lam=harness.lam))
                groups["subgraph_credit_ours"].append(candidate["reward"])
                terminal, final_graph, continuation = harness.run_session(delivered, copy.deepcopy(graph), controller, editor, max(0, args.max_steps - step_index - 1), seed=args.seed + index)
                endpoint = sum(n.weight * terminal.scores[nid] for nid, n in final_graph.nodes.items()) / final_graph.total_weight_Z
                groups["terminal_broadcast"].append(endpoint / (step_index + len(continuation) + 1))
                if rng.random() < 0.05:
                    truth_old, truth_new = audit(before.image, graph, scorer), audit(delivered.image, graph, scorer)
                    leak = sum(n.weight * (1 + harness.lam * (nid in before.protected_set)) * abs(truth_new[nid] - truth_old[nid]) for nid, n in graph.nodes.items() if nid not in candidate["affected"]) / graph.total_weight_Z
                    leakage.append({"session": index, "step": step_index, "E_leak": leak})
            for method, rewards in groups.items():
                if len(rewards) != 6:
                    raise ValueError("Experiment 2 requires six candidates from each identical parent state")
                advantages = compute_pairwise_advantages(rewards) if method == "subgraph_credit_ours" else (torch.tensor(rewards) - torch.tensor(rewards).mean()) * 6 / 5
                curves.append({"session": index, "step": step_index, "method": method, "rewards": rewards, "advantages": advantages.tolist(), "reward_variance": float(np.var(rewards)), "advantage_variance": float(advantages.var(unbiased=False))})
    logs = None
    if args.training_logs:
        paths = json.loads(Path(args.training_logs).read_text())
        if set(paths) != set(methods):
            raise ValueError("training_logs must identify each of the three reward methods")
        logs = {name: list(read_records(str((Path(args.training_logs).parent / path).resolve()))) for name, path in paths.items()}
        if any(not rows for rows in logs.values()):
            raise ValueError("Training convergence logs cannot be empty")
    return {"reward_advantage_curves": curves, "full_audits": leakage, "training_convergence": logs, "convergence_available": logs is not None}

def main():
    parser = argparse.ArgumentParser()
    add_runtime_arguments(parser)
    parser.add_argument("--exp_id", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--audit_model", required=True)
    parser.add_argument("--audit_device")
    parser.add_argument("--max_steps", type=int, default=10)
    parser.add_argument("--verification_budget", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stage_checkpoints")
    parser.add_argument("--training_logs")
    parser.add_argument("--allow-small-dataset", action="store_true")
    args = parser.parse_args()
    records = list(read_records(args.data))
    if not records or args.max_steps < 1 or args.verification_budget < 1:
        raise ValueError("Nonempty data and positive execution budgets are required")
    if args.audit_model == args.verifier_model:
        raise ValueError("Audit and training verifier must use different model checkpoints")
    if args.exp_id == 2 and not args.training_logs and not args.allow_small_dataset:
        raise ValueError("Experiment 2 requires --training_logs for measured policy convergence")
    if args.training_logs:
        paths = json.loads(Path(args.training_logs).read_text())
        if set(paths) != {"terminal_broadcast", "step_full_table", "subgraph_credit_ours"} or any(not (Path(args.training_logs).parent / p).is_file() for p in paths.values()):
            raise ValueError("training_logs must identify three existing reward-method JSONL files")
    stages = {"fixed_weights": {"controller": args.controller_adapter, "editor": args.editor_adapter}}
    if args.exp_id == 3:
        if not args.stage_checkpoints:
            raise ValueError("Experiment 3 requires --stage_checkpoints")
        stages = json.loads(Path(args.stage_checkpoints).read_text())
        if set(stages) != {"zero_shot", "sft", "ctrl_rl", "edit_rl", "alternating_rl"} or any(set(v) != {"controller", "editor"} for v in stages.values()):
            raise ValueError("Five named stages must each specify controller and editor adapter paths")
        for stage, checkpoints in stages.items():
            for model, path in checkpoints.items():
                if path:
                    checkpoints[model] = str((Path(args.stage_checkpoints).parent / path).resolve())
                    if not Path(checkpoints[model]).is_dir():
                        raise ValueError("Stage adapter directory does not exist: " + checkpoints[model])
        if len({json.dumps(v, sort_keys=True) for v in stages.values()}) != 5 or any(stages["zero_shot"].values()) or any(not all(stages[s].values()) for s in stages if s != "zero_shot"):
            raise ValueError("Five distinct stages are required: base-only zero shot and paired trained adapters")
        counts = [len(record.get("graph", record.get("dag"))["nodes"]) for record in records]
        if any(k not in (10, 20, 30) for k in counts) or (not args.allow_small_dataset and any(counts.count(k) < 100 for k in (10, 20, 30))):
            raise ValueError("Each K=10/20/30 tier requires at least 100 sessions; use --allow-small-dataset only for diagnostics")
    audit_device = args.audit_device or args.verifier_device or args.device
    scorer = load_vlm_scorer(args.audit_model, audit_device, dtype=torch.float32 if audit_device == "cpu" else torch.bfloat16)
    result = {"exp_id": args.exp_id, "audit_model": args.audit_model, "diagnostic_only": args.allow_small_dataset, "results": {}}
    if args.exp_id == 2:
        harness, controller, editor = load_runtime(args)
        harness.verification_budget = args.verification_budget
        result["results"] = credit_experiment(args, records, harness, controller, editor, scorer)
    else:
        saved, image_rng = {}, random.Random(args.seed)
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        for stage, checkpoints in stages.items():
            args.controller_adapter, args.editor_adapter = checkpoints["controller"], checkpoints["editor"]
            harness, controller, editor = load_runtime(args)
            for mode in (("flat_full", "tree_rollback_only", "redag_full") if args.exp_id == 1 else ("redag_full",)):
                harness.mode, harness.verification_budget = mode, args.verification_budget
                buckets = {k: [] for k in ((10, 20, 30) if args.exp_id == 3 else (0,))}
                for index, record in enumerate(records):
                    for seed in (range(3) if args.exp_id == 3 else (0,)):
                        state, dag = state_from_record(record, harness)
                        k = len(dag.nodes) if args.exp_id == 3 else 0
                        final, graph, trace = harness.run_session(state, dag, controller, editor, args.max_steps, seed=args.seed + seed)
                        buckets[k].append(summarize(final, graph, trace, scorer))
                        if args.exp_id == 3:
                            folder = output.parent / (output.stem + "_images")
                            folder.mkdir(exist_ok=True)
                            name = f"{image_rng.getrandbits(96):024x}.png"
                            Image.fromarray((final.image.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype("uint8")).save(folder / name)
                            saved.setdefault((index, seed), {})[stage] = str(folder / name)
                result["results"][f"{stage}/{mode}"] = {str(k): aggregate(rows) for k, rows in buckets.items() if rows}
            del harness, controller, editor
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if args.exp_id == 3:
            rng, pairs, key = random.Random(args.seed), [], []
            for index, images in rng.sample(list(saved.items()), min(200, len(saved))):
                names = rng.sample(list(stages), 2)
                pair = {"id": len(pairs), "a": images[names[0]], "b": images[names[1]]}
                pairs.append(pair)
                key.append({"id": pair["id"], "session_seed": index, "a": names[0], "b": names[1]})
            result["blind_pairs"] = pairs
            output.with_suffix(".blind_key.json").write_text(json.dumps(key, ensure_ascii=False, indent=2))
            result["human_ratings"] = None
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))

if __name__ == "__main__":
    main()
