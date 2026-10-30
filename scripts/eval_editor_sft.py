"""Held-out check of the editor SFT (GPU, a few minutes).

The SFT training loss of a flow-matching model is dominated by the randomly drawn noise level, so its running mean says
little.  Here the masked flow-matching loss of the SFT objective is measured on editor_sft_dev (MagicBrush dev, never
trained on) at fixed noise levels, with the same noise for the editor with and without the LoRA adapter.  A lower loss
with the adapter means it predicts the MagicBrush edits better than the base editor.

  python scripts/eval_editor_sft.py --data data/final/editor_sft_dev.jsonl --adapter runs/sft/editor --output runs/eval/editor_sft_dev.json
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch  # noqa: E402

from redag.config import ReDAGConfig  # noqa: E402
from redag.learn.sft_trainer import read_records  # noqa: E402
from redag.models.editor import FluxEditor  # noqa: E402
from redag.verify.grounder import load_image  # noqa: E402


def masked_flow_loss(dit_model, z_src, z_0, text_embeds, mask_latent, t, epsilon, alpha_mask):
    """compute_editor_masked_flow_sft_loss at a given noise level t and noise epsilon (instead of random ones)."""
    batch, channels = z_0.shape[:2]
    timesteps = torch.full((batch,), t, device=z_0.device, dtype=z_0.dtype)
    z_t = (1.0 - t) * z_0 + t * epsilon
    v_pred = dit_model(z_t=z_t, z_src=z_src, text_embeds=text_embeds, timesteps=timesteps)
    weight = 1.0 + alpha_mask * mask_latent.float()
    err = (weight * (v_pred.float() - (epsilon.float() - z_0.float())) ** 2).sum(dim=(1, 2, 3))
    return (err / (float(channels) * weight.sum(dim=(1, 2, 3)).clamp(min=1.0))).mean().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--model', default='black-forest-labs/FLUX.2-klein-base-9B')
    parser.add_argument('--adapter', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--limit', type=int, default=200)
    parser.add_argument('--timesteps', type=float, nargs='+', default=[0.25, 0.5, 0.75])
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    from PIL import Image
    import numpy as np
    records = read_records(args.data)[:args.limit]
    started = time.time()
    editor = FluxEditor.from_pretrained(args.model, args.device, torch.float32 if args.device == 'cpu' else torch.bfloat16, args.adapter, False)
    alpha = ReDAGConfig().alpha_mask
    losses = {'base': {t: [] for t in args.timesteps}, 'sft': {t: [] for t in args.timesteps}}
    with torch.no_grad():
        for index, record in enumerate(records):
            source, target = load_image(record['source']), load_image(record['target'])
            with Image.open(record['mask']) as opened:
                mask = torch.from_numpy(np.asarray(opened.convert('L')).copy()).float().gt(127).float()
            prepared = editor.prepare_sft(source, target, record['instruction'], mask)
            for t in args.timesteps:
                generator = torch.Generator(device=prepared['z_0'].device).manual_seed(args.seed + index)
                epsilon = torch.randn(prepared['z_0'].shape, generator=generator, device=prepared['z_0'].device, dtype=prepared['z_0'].dtype)
                for name in ('sft', 'base'):
                    if name == 'base':
                        editor.transformer.disable_lora()
                    losses[name][t].append(masked_flow_loss(editor.dit_model, **prepared, t=t, epsilon=epsilon, alpha_mask=alpha))
                    if name == 'base':
                        editor.transformer.enable_lora()
            if (index + 1) % 50 == 0:
                print(f'{index + 1}/{len(records)} records', flush=True)
    rows = {}
    for t in args.timesteps:
        base, sft = losses['base'][t], losses['sft'][t]
        rows[str(t)] = dict(base=statistics.fmean(base), sft=statistics.fmean(sft),
                            relative_change=statistics.fmean(sft) / statistics.fmean(base) - 1.0,
                            sft_lower_share=sum(s < b for s, b in zip(sft, base)) / len(base))
    base_all = statistics.fmean(x for t in args.timesteps for x in losses['base'][t])
    sft_all = statistics.fmean(x for t in args.timesteps for x in losses['sft'][t])
    summary = dict(records=len(records), base_loss=base_all, sft_loss=sft_all, relative_change=sft_all / base_all - 1.0,
                   sft_lower_share=statistics.fmean(rows[str(t)]['sft_lower_share'] for t in args.timesteps))
    result = dict(summary=summary, by_timestep=rows, adapter=str(Path(args.adapter).resolve()), data=str(Path(args.data).resolve()),
                  seconds=time.time() - started)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=1))
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
