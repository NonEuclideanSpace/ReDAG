import torch
import torch.nn.functional as F

def compute_autoregressive_sft_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[:, :-1, :].float().contiguous()
    shift_labels = labels[:, 1:].contiguous()
    per_token_nll = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-100, reduction='none').view(shift_labels.size(0), shift_labels.size(1))
    valid_mask = (shift_labels != -100).float()
    lengths = torch.clamp(valid_mask.sum(dim=1), min=1.0)
    per_sample_loss = (per_token_nll * valid_mask).sum(dim=1) / lengths
    return per_sample_loss.mean()

def compute_editor_masked_flow_sft_loss(dit_model, z_src: torch.Tensor, z_0: torch.Tensor, text_embeds: torch.Tensor, mask_latent: torch.Tensor, alpha_mask: float=3.0) -> torch.Tensor:
    B, C, h, w = z_0.shape
    t = torch.rand((B,), device=z_0.device, dtype=z_0.dtype)
    t_view = t.view(B, 1, 1, 1)
    epsilon = torch.randn_like(z_0)
    z_t = (1.0 - t_view) * z_0 + t_view * epsilon
    u_target = epsilon.float() - z_0.float()
    v_pred = dit_model(z_t=z_t, z_src=z_src, text_embeds=text_embeds, timesteps=t)
    W = 1.0 + alpha_mask * mask_latent.float()
    sq_err = (v_pred.float() - u_target) ** 2
    weighted_sq_err = (W * sq_err).sum(dim=(1, 2, 3))
    denom = float(C) * W.sum(dim=(1, 2, 3)).clamp(min=1.0)
    return (weighted_sq_err / denom).mean()


def read_records(path):
    import json
    from pathlib import Path
    source = Path(path).resolve()
    records = []
    with source.open(encoding='utf-8') as stream:
        for index, line in enumerate(stream, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f'Invalid JSONL object at line {index}')
            for key in ('image', 'source', 'target', 'mask'):
                if key in record and isinstance(record[key], str):
                    record[key] = str((source.parent / record[key]).resolve())
            records.append(record)
    if not records:
        raise ValueError('Dataset contains no records')
    return records


def train_sft(model, records, output, kind, epochs=1, batch_size=1, learning_rate=None, seed=42):
    import json
    import random
    from pathlib import Path
    from redag.config import ReDAGConfig
    from redag.verify.grounder import load_image
    config = ReDAGConfig()
    rate = learning_rate or getattr(config, 'lr_sft_' + kind)
    parameters = model.lora_parameters()
    if not parameters or not all(p.requires_grad for p in parameters):
        raise ValueError('SFT requires trainable LoRA parameters')
    if epochs < 1 or batch_size < 1:
        raise ValueError('Epoch and batch counts must be positive')
    if kind == 'editor':
        model.train()
    else:
        model.model.train()
    optimizer = torch.optim.AdamW(parameters, lr=rate, weight_decay=0.0)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    torch.manual_seed(seed)
    step = 0
    with (output / 'metrics.jsonl').open('w', encoding='utf-8') as log:
        for epoch in range(epochs):
            order = list(range(len(records)))
            rng.shuffle(order)
            for start in range(0, len(order), batch_size):
                batch = [records[i] for i in order[start:start + batch_size]]
                optimizer.zero_grad(set_to_none=True)
                total = 0.0
                for record in batch:
                    if kind == 'editor':
                        from PIL import Image
                        import numpy as np
                        source, target = load_image(record['source']), load_image(record['target'])
                        with Image.open(record['mask']) as opened:
                            mask = torch.from_numpy(np.asarray(opened.convert('L')).copy()).float().gt(127).float()
                        prepared = model.prepare_sft(source, target, record['instruction'], mask)
                        loss = compute_editor_masked_flow_sft_loss(model.dit_model, **prepared, alpha_mask=config.alpha_mask)
                    else:
                        logits, labels = model.sft_batch([record])
                        loss = compute_autoregressive_sft_loss(logits, labels)
                    if not torch.isfinite(loss):
                        raise FloatingPointError('Non-finite SFT loss')
                    (loss / len(batch)).backward()
                    total += loss.detach().item() / len(batch)
                torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
                optimizer.step()
                log.write(json.dumps(dict(step=step, epoch=epoch, loss=total)) + '\n')
                log.flush()
                step += 1
    model.save_adapter(output)
    return {'steps': step, 'loss': total}


def collect_controller_rs(records, harness, controller, editor, output, seed=42):
    import json
    from dataclasses import asdict
    from pathlib import Path
    from redag.graph.dag import compute_active_frontier
    from redag.harness.environment import state_from_record
    from redag.models.planner import dag_to_dict
    from redag.verify.grounder import save_image
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    images = output.parent / (output.stem + '_images')
    selected, done = 0, set()
    if output.exists():  # resume: rows already written are kept; a line cut off by a crash is dropped
        kept = []
        for line in output.read_text(encoding='utf-8').splitlines():
            try:
                done.add(json.loads(line)['id'])
                kept.append(line)
            except (ValueError, KeyError, TypeError):
                continue
        output.write_text(''.join(line + '\n' for line in kept), encoding='utf-8')
    with output.open('a', encoding='utf-8') as stream:
        for index, record in enumerate(records):
            identifier = str(record.get('id', f'state_{index:06d}'))
            if identifier in done:
                continue
            state, dag = state_from_record(record, harness)
            frontier = compute_active_frontier(dag, state.scores, harness.config.gate_tau)
            if not frontier:
                continue
            candidates = harness.sample_group(state, dag, controller, editor, frontier[0], 3, seed + index * 3)
            if not candidates:  # execution budget exhausted before any candidate was scored
                continue
            best = max(candidates, key=lambda c: c['reward'])
            if not best['accepted'] or best['reward'] <= 0:
                continue
            image_path = images / f'{index:06d}.png'
            save_image(state.image, image_path)
            row = dict(id=identifier, image=str(image_path.relative_to(output.parent)), graph=dag_to_dict(dag), scores=state.scores,
                       protected_set=sorted(state.protected_set), target_node_id=frontier[0], action=asdict(best['action']), reward=best['reward'],
                       detected_count=harness.grounder.counts.get(dag.nodes[frontier[0]].entity_phrase, 1) if harness.grounder else 1)
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
            stream.flush()
            selected += 1
    return {'parents': len(records), 'accepted': selected, 'previously_accepted': len(done)}
