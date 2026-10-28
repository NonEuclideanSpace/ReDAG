import argparse
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('full_actor_cpu_visual_admission', ROOT / 'src/agentic_image/data/visual_admission.py')
admission = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(admission)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--collection', action='append')
    parser.add_argument('--visual-reviews', default='data/actor-full-sft-visual/visual-reviews-complete.jsonl')
    parser.add_argument('--selection', default='data/training_expansion/training_selection_v1.frozen.jsonl')
    parser.add_argument('--output', default='data/actor-full-sft')
    parser.add_argument('--ready', default='deployment/full-actor-visual-admission.ready.json')
    parser.add_argument('--seed', type=int, default=11)
    parser.add_argument('--updates', type=int, default=200)
    args = parser.parse_args()
    collections = args.collection or ['runs/full-actor-demonstrations/shard-0', 'runs/full-actor-demonstrations/shard-1']
    result = admission.prepare_full_actor_admission(collections, args.output, args.visual_reviews,
        seed=args.seed, selection_path=args.selection, updates=args.updates)
    ready, created = admission.publish_visual_admission_ready(result, args.ready)
    print(json.dumps({'status': ready['status'], 'ready': str(Path(args.ready).resolve()),
                     'created': created, 'identity_sha256': ready['identity_sha256'], 'counts': ready['counts']}, indent=2))


if __name__ == '__main__':
    main()
