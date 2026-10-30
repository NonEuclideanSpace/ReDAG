"""Ground and score failure states once (verifier + grounder only; no controller/editor needed).

state_from_record() re-runs ReDAGHarness.initialize() (full grounding + K verifier calls) every time a
record without cached scores is used: every RL epoch, every stage and every development evaluation.
This writes the grounded graph (boxes), scores and detected_counts back into the file so that work is
done exactly once; states whose active frontier is empty are dropped (nothing left to repair).
  python scripts/data/precompute_states.py --files data/final/rl_train.jsonl data/final/rl_dev.jsonl --device cuda:0
"""
import argparse
import json
import shutil
from pathlib import Path

from _common import log, read_jsonl, write_jsonl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--files', nargs='+', required=True)
    parser.add_argument('--verifier_model', default='Qwen/Qwen3-VL-8B-Instruct')
    parser.add_argument('--detector_model', default='IDEA-Research/grounding-dino-base')
    parser.add_argument('--sam_model', default='facebook/sam2.1-hiera-large')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--keep_satisfied', action='store_true')
    parser.add_argument('--force', action='store_true', help='recompute records that already carry scores')
    args = parser.parse_args()
    import torch
    from redag.graph.dag import compute_active_frontier
    from redag.harness.environment import ReDAGHarness
    from redag.models.planner import dag_from_dict, dag_to_dict
    from redag.verify.grounder import Grounder, load_image
    from redag.verify.verifier import TwoStageVerifier, load_vlm_scorer
    dtype = torch.float32 if args.device == 'cpu' else torch.bfloat16
    harness = ReDAGHarness(TwoStageVerifier(load_vlm_scorer(args.verifier_model, args.device, dtype)),
                           Grounder.from_pretrained(args.detector_model, args.sam_model, args.device))
    for name in args.files:
        path = Path(name)
        backup = path.with_name(path.stem + '.ungrounded.jsonl')
        if not backup.exists():
            shutil.copyfile(path, backup)
        rows, kept, dropped = read_jsonl(path), [], 0
        for index, row in enumerate(rows, 1):
            if 'scores' in row and not args.force:
                kept.append(row)
                continue
            image = load_image((path.parent / row['image']).resolve())
            state, dag = harness.initialize(image, dag_from_dict(row['graph']))
            frontier = compute_active_frontier(dag, state.scores, harness.config.gate_tau)
            if not frontier and not args.keep_satisfied:
                dropped += 1
                continue
            row.update(graph=json.loads(json.dumps(dag_to_dict(dag))), scores=state.scores,
                       detected_counts=dict(harness.grounder.counts), frontier=frontier[:5],
                       protected=len(state.protected_set))
            kept.append(row)
            if index % 20 == 0:
                write_jsonl(path, kept + rows[index:])  # checkpoint progress
                log(f'{path.name}: {index}/{len(rows)} grounded, {dropped} dropped (already satisfied)')
        write_jsonl(path, kept)
        log(f'{path.name}: kept {len(kept)}, dropped {dropped}; original saved as {backup.name}')


if __name__ == '__main__':
    main()
