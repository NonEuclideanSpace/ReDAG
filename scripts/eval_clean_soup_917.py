import argparse, json, sys, time
from pathlib import Path
from peft import set_peft_model_state_dict
from safetensors.torch import load_file

REPO = Path("/root/autodl-tmp/ReDAG")
sys.path.insert(0, str(REPO))

from redag.harness.environment import add_runtime_arguments, load_runtime
from redag.learn.sft_trainer import read_records
from scripts.run_ctrl_rl_v51 import V51_DEFAULTS, assert_v51_alignment, cache_fingerprint, evaluate_dev_v51, prepare_parents

def snap_from(controller, d):
    st = load_file(str(Path(d) / "adapter_model.safetensors"))
    set_peft_model_state_dict(controller.model, st)
    return controller.snapshot()

def main():
    parser = add_runtime_arguments(argparse.ArgumentParser())
    parser.set_defaults(**V51_DEFAULTS)
    args = parser.parse_args()
    rl_dir = REPO / "runs/rl_v51/ctrl_grpo_clean"
    bd = json.loads((rl_dir / "baseline_dev.json").read_text(encoding="utf-8"))
    b_k1, b_k3 = bd["k1"], bd["k_final"]
    dev_records = read_records(str(REPO / "data/final/rl_dev_v51.jsonl"))
    harness, controller, editor = load_runtime(args, trainable_controller=True, trainable_editor=False)
    assert_v51_alignment(harness, controller, editor)
    dev_hash = cache_fingerprint(str(REPO / "data/final/rl_dev_v51.jsonl"), harness, args)
    dev_parents = prepare_parents(dev_records, harness, cache_path=REPO / "runs/rl_v51/ctrl_grpo_lr25e6/cache" / f"dev_{dev_hash}.jsonl")
    print(f"Loaded {len(dev_parents)} dev parents", flush=True)

    ctrl_ref = controller.snapshot()
    s012 = snap_from(controller, REPO / "runs/rl_v51/ctrl_grpo_clean/ckpt_step_012")
    soup = {k: (0.5 * ctrl_ref[k] + 0.5 * s012[k]) for k in ctrl_ref}
    out_dir = REPO / "runs/rl_v51/ctrl_grpo_clean/ckpt_soup_sft_clean012"
    controller.load_snapshot(soup)
    controller.save_adapter(out_dir)

    t0 = time.perf_counter()
    k1 = evaluate_dev_v51(dev_parents, harness, controller, editor, ctrl_ref, seed=42, candidate_k=1)
    print("K1 DONE:", json.dumps(k1), flush=True)
    k3 = evaluate_dev_v51(dev_parents, harness, controller, editor, ctrl_ref, seed=42, candidate_k=3)
    gate_passed = bool(
        k3["dev_mean_consistent_gain"] > b_k3["dev_mean_consistent_gain"]
        and k3["dev_acceptance_p"] >= b_k3["dev_acceptance_p"]
        and (k3["dev_acceptance_p"] > b_k3["dev_acceptance_p"] or k1["dev_acceptance_p"] > b_k1["dev_acceptance_p"])
        and k3["dev_repair_rate"] <= b_k3["dev_repair_rate"] + 0.05
        and k3["dev_full_box_share"] <= 0.25
    )
    rec = {
        "name": "ckpt_soup_sft_clean012",
        "ckpt_dir": str(out_dir),
        "gate_passed": gate_passed,
        "k1": k1,
        "k_final": k3,
        "gain_k_final_consistent": k3["dev_mean_consistent_gain"] - b_k3["dev_mean_consistent_gain"],
        "gain_k_final_acceptance_p": k3["dev_acceptance_p"] - b_k3["dev_acceptance_p"],
        "wall_seconds": time.perf_counter() - t0,
    }
    (REPO / "runs/rl_v51/ctrl_grpo_clean/soup_sft_clean012_eval.json").write_text(json.dumps(rec, indent=2))
    print("SOUP 917 DONE:", json.dumps(rec), flush=True)

if __name__ == "__main__":
    main()
