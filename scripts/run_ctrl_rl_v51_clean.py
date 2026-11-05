import argparse
import json
import random
import shutil
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from redag.config import ReDAGConfig
from redag.graph.dag import box_area, classify_failure_mode
from redag.harness.environment import add_runtime_arguments, consistent_count, load_runtime
from redag.learn.credit import compute_pairwise_advantages
from redag.learn.rl_trainer import relative_lora_displacement
from redag.learn.sft_trainer import read_records
from redag.models.controller import OPS
from scripts.run_ctrl_rl_v51 import (
    V51_DEFAULTS,
    assert_v51_alignment,
    cache_fingerprint,
    configure_parent_context,
    detach_sample_to_cpu,
    evaluate_dev_v51,
    optimize_ctrl_groups,
    parent_canvas_state,
    prepare_parents,
)


def compute_clean_v51_step_reward(harness, parent_state, dag, cand):
    eff = harness.effective_graph(dag)
    before_con = consistent_count(eff, parent_state.scores)
    after_con = consistent_count(eff, cand["delivered_state"].scores)
    con_gain = after_con - before_con

    target_id = cand["target_node_id"]
    target_gain = float(cand["raw_scores"][target_id] - parent_state.scores[target_id])
    drops = sum(
        1 for v in parent_state.protected_set
        if v in eff.nodes and cand["raw_scores"].get(v, 1.0) < eff.nodes[v].threshold
    )

    if cand["accepted"]:
        base = 0.30 + 0.18 * float(max(0, con_gain)) + 0.10 * max(0.0, target_gain)
    else:
        base = (0.10 * max(0.0, target_gain)) if drops == 0 else 0.0

    sample = cand.get("sample") or {}
    raw_box = sample.get("controller_box", cand["action"].box)
    area = box_area(raw_box)
    det_cnt = (
        harness.grounder.counts.get(dag.nodes[target_id].entity_phrase, 1)
        if getattr(harness, "grounder", None) is not None else 1
    )
    failure_mode = classify_failure_mode(dag, target_id, parent_state.scores, det_cnt)
    expected_op = OPS[failure_mode]
    node_bbox = dag.nodes[target_id].bbox
    ref_area = box_area(node_bbox) if node_bbox is not None else 0.25
    min_area = min(0.12, max(0.04, 0.8 * ref_area)) if failure_mode == "missing" else min(0.06, max(0.02, 0.5 * ref_area))

    box_pen = 0.10 if (area >= 0.92 or area < min_area) else 0.0
    op_pen = 0.06 if cand["action"].op != expected_op else 0.0
    fmt_pen = 0.10 if sample.get("instruction_repaired", False) else 0.0

    reward = base - box_pen - op_pen - fmt_pen
    return float(reward), int(con_gain)


@torch.no_grad()
def rollout_parent_group_clean(harness, controller, editor, parent, group_size, noise_seed, inline_image=None):
    configure_parent_context(harness, parent)
    state = parent_canvas_state(parent, inline_image=inline_image)
    saved_cf, saved_bonus = harness.counterfactual_credit, harness.acceptance_bonus
    harness.counterfactual_credit = True
    harness.acceptance_bonus = 0.0
    try:
        cands = harness.sample_group(
            state, parent.dag, controller, editor, parent.target_node_id, group_size=group_size, seed=noise_seed,
        )
    finally:
        harness.counterfactual_credit = saved_cf
        harness.acceptance_bonus = saved_bonus
        harness.verifier.records.clear()
        harness.last_candidate = None
        harness.global_failures, harness.local_failures = {}, {}
    if len(cands) < 2:
        return None
    items, rewards, con_gains, accepts, areas, ops, repairs = [], [], [], [], [], {}, []
    for cand in cands:
        sample = cand.get("sample") or {}
        if "token_ids" not in sample:
            raise RuntimeError(f"Expected local controller sample with token_ids for {parent.record_id}:{parent.target_node_id}")
        rew, c_gain = compute_clean_v51_step_reward(harness, state, parent.dag, cand)
        raw_box = sample.get("controller_box", cand["action"].box)
        items.append(detach_sample_to_cpu(sample))
        rewards.append(rew)
        con_gains.append(c_gain)
        accepts.append(int(cand["accepted"]))
        areas.append(box_area(raw_box))
        ops[cand["action"].op] = ops.get(cand["action"].op, 0) + 1
        repairs.append(int(bool(sample.get("instruction_repaired", False))))
    return {
        "items": items,
        "rewards": rewards,
        "con_gains": con_gains,
        "accepts": accepts,
        "areas": areas,
        "ops": ops,
        "repairs": repairs,
    }


def train_ctrl_rl_v51_clean(
    harness, controller, editor, train_parents, dev_parents, output,
    max_steps=12, batch_size=6, group_size=4, eval_every=6, final_k=3,
    lr=1.5e-5, ppo_epochs=2, clip_eps=0.2, beta_kl=0.06, target_kl=0.05,
    min_dev_states=64, seed=42, inline_images=None, baseline_json=None,
    top_k_verify=3, cli_args=None,
):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if max_steps < 1:
        raise ValueError(f"max_steps must be >= 1, got {max_steps}")
    if harness.strategy is not None:
        raise ValueError("Controller RL requires harness.strategy to be None during rollout")
    if len(dev_parents) < min_dev_states:
        raise ValueError(f"Hard dev gate requires >= {min_dev_states} local-frontier dev states, got {len(dev_parents)}")
    if len(train_parents) < batch_size:
        raise ValueError(f"Need at least {batch_size} train parents, got {len(train_parents)}")
    if {p.session_id for p in train_parents} & {p.session_id for p in dev_parents}:
        raise ValueError("Train and dev session_ids must be strictly disjoint")

    ctrl_params = controller.lora_parameters()
    if not ctrl_params:
        raise ValueError("Controller requires trainable LoRA adapter parameters")
    for p in ctrl_params:
        p.requires_grad_(True)
    for p in editor.lora_parameters():
        p.requires_grad_(False)
    editor.eval()

    cfg = ReDAGConfig(
        lr_rl_controller=lr,
        ppo_epochs=ppo_epochs,
        clip_eps_ctrl=clip_eps,
        beta_kl_ctrl=beta_kl,
        target_kl_ctrl=target_kl,
    )
    ctrl_ref = controller.snapshot()
    controller.save_adapter(output / "ctrl_sft_init")

    if baseline_json and Path(baseline_json).exists():
        bd = json.loads(Path(baseline_json).read_text(encoding="utf-8"))
        baseline_k1 = bd["k1"]
        baseline_k3 = bd["k_final"]
        if int(baseline_k1["dev_states"]) != len(dev_parents) or int(baseline_k3["candidate_k"]) != int(final_k):
            raise ValueError("Precomputed baseline_dev.json does not match dev_parents or final_k")
        print(f"Reused verified baseline_dev.json from {baseline_json}", flush=True)
    else:
        baseline_k1 = evaluate_dev_v51(dev_parents, harness, controller, editor, ctrl_ref, seed=seed, candidate_k=1, inline_images=inline_images)
        baseline_k3 = (
            evaluate_dev_v51(dev_parents, harness, controller, editor, ctrl_ref, seed=seed, candidate_k=final_k, inline_images=inline_images)
            if final_k > 1 else dict(baseline_k1)
        )
    (output / "baseline_dev.json").write_text(json.dumps({"k1": baseline_k1, "k_final": baseline_k3}, indent=2))

    optimizer = torch.optim.AdamW(ctrl_params, lr=lr, weight_decay=0.0)
    rng = random.Random(seed)
    order = []
    metrics_path = output / "metrics.jsonl"
    metrics_path.write_text("", encoding="utf-8")
    candidates_pool = []
    history = []

    for step in range(1, max_steps + 1):
        t0 = time.perf_counter()
        if len(order) < batch_size:
            extra = list(range(len(train_parents)))
            rng.shuffle(extra)
            order.extend(extra)
        batch_indices, order = order[:batch_size], order[batch_size:]
        old_snap = controller.snapshot()
        groups, rewards_all, adv_all, con_all, acc_all, area_all, ops_all, rep_all = [], [], [], [], [], [], {}, []
        for p_idx, idx in enumerate(batch_indices):
            parent = train_parents[idx]
            noise_seed = seed + 1000 * step + 17 * p_idx
            img = inline_images.get(parent.record_id) if inline_images else None
            rolled = rollout_parent_group_clean(harness, controller, editor, parent, group_size, noise_seed, inline_image=img)
            if rolled is None:
                continue
            adv = compute_pairwise_advantages(rolled["rewards"], cfg.delta_noise, cfg.reward_scale_s0, cfg.adv_clip_max)
            groups.append((rolled["items"], adv))
            rewards_all.extend(rolled["rewards"])
            adv_all.extend(adv.tolist())
            con_all.extend(rolled["con_gains"])
            acc_all.extend(rolled["accepts"])
            area_all.extend(rolled["areas"])
            rep_all.extend(rolled["repairs"])
            for k_op, cnt in rolled["ops"].items():
                ops_all[k_op] = ops_all.get(k_op, 0) + cnt

        opt_metrics = optimize_ctrl_groups(groups, controller, optimizer, old_snap, ctrl_ref, cfg)
        row = {
            "step": step,
            "stage": "ctrl_rl_v51_clean",
            "sampled_parents": len(groups),
            "sampled_candidates": len(rewards_all),
            "mean_reward": sum(rewards_all) / max(len(rewards_all), 1),
            "reward_variance": float(torch.tensor(rewards_all).var(unbiased=False)) if rewards_all else 0.0,
            "advantage_variance": float(torch.tensor(adv_all).var(unbiased=False)) if adv_all else 0.0,
            "train_acceptance_p": sum(acc_all) / max(len(acc_all), 1),
            "train_mean_consistent_gain": sum(con_all) / max(len(con_all), 1),
            "action_box_area_mean": sum(area_all) / max(len(area_all), 1),
            "action_full_box_share": sum(a >= 0.92 for a in area_all) / max(len(area_all), 1),
            "action_tiny_box_share": sum(a < 0.06 for a in area_all) / max(len(area_all), 1),
            "instruction_repair_rate": sum(rep_all) / max(len(rep_all), 1),
            "action_ops": ops_all,
            "ctrl_lora_relative_displacement": relative_lora_displacement(ctrl_params, list(ctrl_ref.values())),
            **opt_metrics,
        }
        if step % eval_every == 0 or step == max_steps:
            dev_k1 = evaluate_dev_v51(dev_parents, harness, controller, editor, ctrl_ref, seed=seed, candidate_k=1, inline_images=inline_images)
            row.update({
                "dev_acceptance_p": dev_k1["dev_acceptance_p"],
                "dev_acceptance_gain": dev_k1["dev_acceptance_p"] - baseline_k1["dev_acceptance_p"],
                "dev_mean_consistent_gain": dev_k1["dev_mean_consistent_gain"],
                "dev_consistent_delta_vs_sft": dev_k1["dev_mean_consistent_gain"] - baseline_k1["dev_mean_consistent_gain"],
                "dev_repair_rate": dev_k1["dev_repair_rate"],
                "dev_full_box_share": dev_k1["dev_full_box_share"],
                "dev_ctrl_kl_sft": dev_k1["ctrl_kl_sft"],
            })
            ckpt_dir = output / f"ckpt_step_{step:03d}"
            controller.save_adapter(ckpt_dir)
            candidates_pool.append((step, dev_k1, controller.snapshot(), str(ckpt_dir)))
        row["step_wall_seconds"] = time.perf_counter() - t0
        with metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        history.append(row)
        print(json.dumps({k: row[k] for k in row if k != "action_ops"}), flush=True)

    candidates_pool.sort(
        key=lambda item: (
            item[1]["dev_mean_consistent_gain"],
            item[1]["dev_acceptance_p"],
            -item[1]["ctrl_kl_sft"],
        ),
        reverse=True,
    )
    top_to_verify = candidates_pool[:max(1, int(top_k_verify))]
    verified_k3 = []
    for step_id, dev_k1, snap, ckpt_dir in top_to_verify:
        controller.load_snapshot(snap)
        dev_k3 = (
            evaluate_dev_v51(dev_parents, harness, controller, editor, ctrl_ref, seed=seed, candidate_k=final_k, inline_images=inline_images)
            if final_k > 1 else dict(dev_k1)
        )
        verified_k3.append((step_id, dev_k1, dev_k3, snap, ckpt_dir))

    verified_k3.sort(
        key=lambda item: (
            item[2]["dev_mean_consistent_gain"],
            item[2]["dev_acceptance_p"],
            item[1]["dev_mean_consistent_gain"],
            -item[2]["ctrl_kl_sft"],
        ),
        reverse=True,
    )
    best_step, best_k1, best_k3, best_snap, _ = verified_k3[0]
    gate_passed = bool(
        best_k3["dev_mean_consistent_gain"] > baseline_k3["dev_mean_consistent_gain"]
        and best_k3["dev_acceptance_p"] >= baseline_k3["dev_acceptance_p"]
        and (
            best_k3["dev_acceptance_p"] > baseline_k3["dev_acceptance_p"]
            or best_k1["dev_acceptance_p"] > baseline_k1["dev_acceptance_p"]
        )
        and best_k3["dev_repair_rate"] <= baseline_k3["dev_repair_rate"] + 0.05
        and best_k3["dev_full_box_share"] <= 0.25
    )
    if gate_passed:
        controller.load_snapshot(best_snap)
        controller.save_adapter(output / "ctrl_best")
    else:
        controller.load_snapshot(ctrl_ref)

    summary = {
        "gate_passed": gate_passed,
        "selected_step": best_step if gate_passed else 0,
        "dev_states": len(dev_parents),
        "train_states": len(train_parents),
        "baseline_k1": baseline_k1,
        "baseline_k_final": baseline_k3,
        "best_k1": best_k1,
        "best_k_final": best_k3,
        "all_verified_k_final": [
            {"step": s, "k1": k1, "k_final": k3, "ckpt_dir": cdir}
            for s, k1, k3, _, cdir in verified_k3
        ],
        "gain_k_final_consistent": best_k3["dev_mean_consistent_gain"] - baseline_k3["dev_mean_consistent_gain"],
        "gain_k_final_acceptance_p": best_k3["dev_acceptance_p"] - baseline_k3["dev_acceptance_p"],
        "gain_k1_consistent": best_k1["dev_mean_consistent_gain"] - baseline_k1["dev_mean_consistent_gain"],
        "gain_k1_acceptance_p": best_k1["dev_acceptance_p"] - baseline_k1["dev_acceptance_p"],
        "cli_args": cli_args or {},
    }
    (output / "rl_v51_summary.json").write_text(json.dumps(summary, indent=2))
    return summary, history


def main():
    parser = add_runtime_arguments(argparse.ArgumentParser())
    parser.set_defaults(**V51_DEFAULTS)
    parser.add_argument("--data", required=True)
    parser.add_argument("--dev_data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--baseline_json", default=None)
    parser.add_argument("--max_rl_steps", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--group_size", type=int, default=4)
    parser.add_argument("--eval_every", type=int, default=6)
    parser.add_argument("--final_k", type=int, default=3)
    parser.add_argument("--top_k_verify", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1.5e-5)
    parser.add_argument("--ppo_epochs", type=int, default=2)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--beta_kl", type=float, default=0.06)
    parser.add_argument("--target_kl", type=float, default=0.05)
    parser.add_argument("--min_dev_states", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out = Path(args.output)
    if (out / "rl_v51_summary.json").exists():
        print("already complete:", out / "rl_v51_summary.json")
        return
    if not args.controller_adapter or not args.editor_adapter:
        parser.error("Both --controller_adapter and --editor_adapter are required")

    started = time.perf_counter()
    torch.manual_seed(args.seed)
    train_records = read_records(args.data)
    dev_records = read_records(args.dev_data)
    harness, controller, editor = load_runtime(args, trainable_controller=True, trainable_editor=False)
    assert_v51_alignment(harness, controller, editor)

    cache_dir = Path(args.cache_dir) if args.cache_dir else (out / "cache")
    dev_hash = cache_fingerprint(args.dev_data, harness, args)
    train_hash = cache_fingerprint(args.data, harness, args)
    dev_parents = prepare_parents(dev_records, harness, cache_path=cache_dir / f"dev_{dev_hash}.jsonl")
    train_parents = prepare_parents(train_records, harness, cache_path=cache_dir / f"train_{train_hash}.jsonl")
    print(f"Prepared {len(dev_parents)}/{len(dev_records)} dev parents and {len(train_parents)}/{len(train_records)} train parents", flush=True)

    summary, _ = train_ctrl_rl_v51_clean(
        harness, controller, editor, train_parents, dev_parents, out,
        max_steps=args.max_rl_steps, batch_size=args.batch_size, group_size=args.group_size,
        eval_every=args.eval_every, final_k=args.final_k, lr=args.lr, ppo_epochs=args.ppo_epochs,
        clip_eps=args.clip_eps, beta_kl=args.beta_kl, target_kl=args.target_kl,
        min_dev_states=args.min_dev_states, seed=args.seed, baseline_json=args.baseline_json,
        top_k_verify=args.top_k_verify, cli_args=vars(args),
    )
    summary["wall_seconds_total"] = time.perf_counter() - started
    (out / "rl_v51_summary.json").write_text(json.dumps(summary, indent=2))
    print("FINAL SUMMARY:", json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
