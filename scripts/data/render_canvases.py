"""Render the initial canvases queued in data/processed/canvases/requests.jsonl (GPU, resumable, shardable).

Only sessions that build_sets.py actually selected are queued (T2I prompts, and LGBench I2I base scenes from
simple_source_prompt).  Every request has a fixed seed, so all methods start from identical canvases.
  python scripts/data/render_canvases.py --device cuda:1 --size 768 --steps 50
"""
import argparse
from pathlib import Path

from _common import PROC, iter_jsonl, log
from redag.progress import report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--requests', default=str(PROC / 'canvases' / 'requests.jsonl'))
    parser.add_argument('--model', default='black-forest-labs/FLUX.2-klein-base-9B')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--size', type=int, default=768)
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--guidance', type=float, default=4.0)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--num_shards', type=int, default=1)
    args = parser.parse_args()
    rows = [r for i, r in enumerate(iter_jsonl(args.requests)) if i % args.num_shards == args.shard and not Path(r['path']).exists()]
    log(f'{len(rows)} canvases to render on {args.device}')
    if not rows:
        return
    import torch
    from diffusers import Flux2KleinPipeline
    pipe = Flux2KleinPipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16).to(args.device)
    pipe.set_progress_bar_config(disable=True)
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start:start + args.batch_size]
        generators = [torch.Generator(device=args.device).manual_seed(int(r['seed'])) for r in batch]
        images = pipe(prompt=[r['prompt'] for r in batch], height=args.size, width=args.size, num_inference_steps=args.steps,
                      guidance_scale=args.guidance, generator=generators).images
        for row, image in zip(batch, images):
            out = Path(row['path'])
            out.parent.mkdir(parents=True, exist_ok=True)
            tmp = out.with_name(out.stem + '.tmp.png')
            image.save(tmp)
            tmp.replace(out)
        log(f'{start + len(batch)}/{len(rows)} rendered')
        report(start + len(batch), len(rows), 'canvases')


if __name__ == '__main__':
    main()
