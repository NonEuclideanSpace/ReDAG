import argparse
import json
import sys
import time
from pathlib import Path

import torch
from peft import set_peft_model_state_dict
from safetensors.torch import load_file

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from redag.harness.environment import add_runtime_arguments, load_runtime
from redag.learn.sft_trainer import read_records
from scripts.run_ctrl_rl_v51 import (
    V51_DEFAULTS,
    assert_v51_alignment,
    cache_fingerprint,
    evaluate_dev_v51,
    prepare_parents,
)


def snapshot_from_ckpt(controller, ckpt_dir):
    state = load_file(str(Path(ckpt_dir) / "adapter_model.safetensors"))
    set_peft_model_state_dict(controller.model, state)
    return controller.snapshot()


def main():
    parser = add_runtime_arguments(argparse.ArgumentParser())
    parser.set_defaults(**V51_DEFAULTS)
    parser.add_argument("--dev_data", default="data/final/rl_dev_v51.jsonl")
    parser.add_argument("--rl_dir", default="runs/rl_v51/ctrl_grpo")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rl_dir = Path(args.rl_dir)
    bd = json.loads((rl_dir / "baseline_dev.json").read_text(encoding="utf-8"))
    b_k1, b_k3 = bd["k1"], bd["k_final"]

    dev_records = read_records(args.dev_data)
    harness, controller, editor = load_runtime(args, trainable_controller=True, trainable_editor=False)
    assert_v51_alignment(harness, controller, editor)

    dev_hash = cache_fingerprint(args.dev_data, harness, args)
    dev_parents = prepare_parents(dev_records, harness, cache_path=rl_dir / "cache" / f"dev_{dev_hash}.jsonl")
    print(f"Loaded {len(dev_parents)} dev parents", flush=True)

    ctrl_ref = controller.snapshot()
    snap_008 = snapshot_from_ckpt(controller, rl_dir / "ckpt_step_008")
    snap_024 = snapshot_from_ckpt(controller, rl_dir / "ckpt_step_024")
    snap_040 = snapshot_from_ckpt(controller, rl_dir / "ckpt_step_040")
    snap_soup_050_024 = {k: (0.5 * ctrl_ref[k] + 0.5 * snap_024[k]) for k in ctrl_ref}

    soup_dir = rl_dir / "ckpt_soup_050_step024"
    controller.load_snapshot(snap_soup_050_024)
    controller.save_adapter(soup_dir)

    metrics_by_step = {}
    for line in (rl_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if "dev_acceptance_p" in row:
                metrics_by_step[int(row["step"])] = row

    tasks = [
        ("ckpt_step_008", snap_008, metrics_by_step.get(8), str(rl_dir / "ckpt_step_008")),
        ("ckpt_step_040", snap_040, metrics_by_step.get(40), str(rl_dir / "ckpt_step_040")),
        ("ckpt_soup_050_step024", snap_soup_050_024, None, str(soup_dir)),
    ]

    out_jsonl = rl_dir / "extra_k3_eval.jsonl"
    out_jsonl.write_text("", encoding="utf-8")

    for name, snap, m_k1, cdir in tasks:
        t0 = time.perf_counter()
        controller.load_snapshot(snap)
        if m_k1 is None:
            k1 = evaluate_dev_v51(dev_parents, harness, controller, editor, ctrl_ref, seed=args.seed, candidate_k=1)
        else:
            k1 = {
                "dev_acceptance_p": m_k1["dev_acceptance_p"],
                "dev_mean_consistent_gain": m_k1["dev_mean_consistent_gain"],
                "dev_repair_rate": m_k1["dev_repair_rate"],
                "dev_full_box_share": m_k1["dev_full_box_share"],
                "ctrl_kl_sft": m_k1["dev_ctrl_kl_sft"],
            }
        k3 = evaluate_dev_v51(dev_parents, harness, controller, editor, ctrl_ref, seed=args.seed, candidate_k=3)
        gate_passed = bool(
            k3["dev_mean_consistent_gain"] > b_k3["dev_mean_consistent_gain"]
            and k3["dev_acceptance_p"] >= b_k3["dev_acceptance_p"]
            and (
                k3["dev_acceptance_p"] > b_k3["dev_acceptance_p"]
                or k1["dev_acceptance_p"] > b_k1["dev_acceptance_p"]
            )
            and k3["dev_repair_rate"] <= b_k3["dev_repair_rate"] + 0.05
            and k3["dev_full_box_share"] <= 0.25
        )
        rec = {
            "name": name,
            "ckpt_dir": cdir,
            "gate_passed": gate_passed,
            "k1": k1,
            "k_final": k3,
            "gain_k_final_consistent": k3["dev_mean_consistent_gain"] - b_k3["dev_mean_consistent_gain"],
            "gain_k_final_acceptance_p": k3["dev_acceptance_p"] - b_k3["dev_acceptance_p"],
            "wall_seconds": time.perf_counter() - t0,
        }
        with out_jsonl.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        print("EVAL DONE:", json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
