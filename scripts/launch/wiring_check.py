"""CPU wiring check with tiny random models built from the real configs, tokenizers and processors.

Catches API mismatches (transformers / diffusers / peft versions, chat templates, LoRA target names,
VAE latent layout, generation kwargs) before any GPU hour is spent.  Every model keeps its real
architecture and vocabulary but has a few tiny layers, so the whole check fits in ~1.5 GB of RAM:
  python scripts/launch/wiring_check.py                         # build tiny models + run every section
  python scripts/launch/wiring_check.py --only editor controller
Sections: controller planner verifier editor grounder loops (training/RL/eval loops with all tiny parts).
"""
import argparse
import json
import shutil
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
import torch  # noqa: E402

MODELS = Path('/root/autodl-tmp/models')
RESULTS = {}
LIGHT = {'on': False, 'dtype': torch.float32, 'device': 'cpu'}  # --light: bf16 Qwen models, forward-only SFT checks, no reload (fits 1.3 GB RSS)


def copy_files(src, dst, names):
    dst.mkdir(parents=True, exist_ok=True)
    for name in names:
        if (src / name).exists():
            shutil.copy(src / name, dst / name)


def tiny_qwen35(src, dst):
    from transformers import AutoConfig, AutoModelForMultimodalLM
    config = json.loads((src / 'config.json').read_text())
    text, vision = config['text_config'], config['vision_config']
    text.update(hidden_size=64, intermediate_size=128, num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=1,
                head_dim=32, linear_num_key_heads=2, linear_num_value_heads=2, linear_key_head_dim=16, linear_value_head_dim=16,
                layer_types=['linear_attention'] * 3 + ['full_attention'], full_attention_interval=4, mtp_num_hidden_layers=0)
    for key in ('rope_parameters', 'rope_scaling'):
        if isinstance(text.get(key), dict) and 'mrope_section' in text[key]:
            text[key]['mrope_section'] = [2, 1, 1]  # partial_rotary 0.25 * head_dim 32 = 8 -> 4 pairs
    vision.update(depth=1, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=64, deepstack_visual_indexes=[])
    dst.mkdir(parents=True, exist_ok=True)
    (dst / 'config.json').write_text(json.dumps(config))
    model = AutoModelForMultimodalLM.from_config(AutoConfig.from_pretrained(dst))
    model.save_pretrained(dst)
    copy_files(src, dst, ['chat_template.jinja', 'chat_template.json', 'merges.txt', 'preprocessor_config.json', 'tokenizer.json',
                          'tokenizer_config.json', 'video_preprocessor_config.json', 'vocab.json', 'generation_config.json'])


def tiny_qwen3vl(src, dst):
    from transformers import AutoConfig, AutoModelForImageTextToText
    config = json.loads((src / 'config.json').read_text())
    text, vision = config['text_config'], config['vision_config']
    text.update(hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1, head_dim=32)
    for key in ('rope_parameters', 'rope_scaling'):
        if isinstance(text.get(key), dict) and 'mrope_section' in text[key]:
            text[key]['mrope_section'] = [6, 5, 5]
    vision.update(depth=2, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=64, deepstack_visual_indexes=[0])
    dst.mkdir(parents=True, exist_ok=True)
    (dst / 'config.json').write_text(json.dumps(config))
    model = AutoModelForImageTextToText.from_config(AutoConfig.from_pretrained(dst))
    model.save_pretrained(dst)
    copy_files(src, dst, ['chat_template.json', 'chat_template.jinja', 'merges.txt', 'preprocessor_config.json', 'tokenizer.json',
                          'tokenizer_config.json', 'video_preprocessor_config.json', 'vocab.json', 'generation_config.json'])


def tiny_flux(src, dst):
    from diffusers import AutoencoderKLFlux2, Flux2Transformer2DModel
    from transformers import Qwen3Config, Qwen3ForCausalLM
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy(src / 'model_index.json', dst / 'model_index.json')
    for folder in ('scheduler', 'tokenizer'):
        shutil.copytree(src / folder, dst / folder, dirs_exist_ok=True)
    text = json.loads((src / 'text_encoder' / 'config.json').read_text())
    text.update(hidden_size=16, intermediate_size=32, num_hidden_layers=28, num_attention_heads=2, num_key_value_heads=1, head_dim=8)
    if text.get('layer_types'):
        text['layer_types'] = ['full_attention'] * 28
    Qwen3ForCausalLM(Qwen3Config(**{k: v for k, v in text.items() if not k.startswith('_')})).save_pretrained(dst / 'text_encoder')
    cfg = {k: v for k, v in json.loads((src / 'transformer' / 'config.json').read_text()).items() if not k.startswith('_')}
    cfg.update(attention_head_dim=16, axes_dims_rope=[4, 4, 4, 4], num_attention_heads=2, num_layers=1, num_single_layers=1,
               joint_attention_dim=48, timestep_guidance_channels=32, mlp_ratio=2.0)
    Flux2Transformer2DModel(**cfg).save_pretrained(dst / 'transformer')
    vae = {k: v for k, v in json.loads((src / 'vae' / 'config.json').read_text()).items() if not k.startswith('_')}
    vae.update(block_out_channels=[8, 8, 8, 8], layers_per_block=1, norm_num_groups=4)
    AutoencoderKLFlux2(**vae).save_pretrained(dst / 'vae')


def section(name):
    def wrap(func):
        def run(*args):
            started = time.time()
            try:
                detail = func(*args)
                RESULTS[name] = {'ok': True, 'seconds': round(time.time() - started, 1), 'detail': detail}
            except Exception as error:  # report every failure, keep going
                RESULTS[name] = {'ok': False, 'seconds': round(time.time() - started, 1), 'error': repr(error)[:600],
                                 'trace': traceback.format_exc()[-2500:]}
            print(name, json.dumps(RESULTS[name])[:1500], flush=True)
        return run
    return wrap


def canvas(height=96, width=128, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return torch.rand(3, height, width, generator=generator)


def graph():
    from redag.models.planner import dag_from_dict
    return dag_from_dict({'nodes': [
        {'node_id': 'cup', 'node_type': 'entity', 'question': 'Is there a cup?', 'entity_phrase': 'cup', 'scope': 'local'},
        {'node_id': 'red', 'node_type': 'attribute', 'question': 'Is the cup red?', 'entity_phrase': 'cup', 'scope': 'local',
         'depends_on': ['cup'], 'threshold': 0.99},  # random tiny weights never reach 0.99: a repair target always exists
        {'node_id': 'style', 'node_type': 'background', 'question': 'Is it a watercolor painting?', 'entity_phrase': '',
         'scope': 'global'}], 'comp_edges': [['cup', 'red']]})


def fake_grounder():
    from redag.verify.grounder import Grounder
    return Grounder(lambda image, phrase: [[10.0, 12.0, 70.0, 60.0]], None)


def lora_kinds(module):
    names = [n for n, _ in module.named_modules() if n.endswith('lora_A')]
    kinds = {}
    for name in names:
        key = name.split('.')[-2]
        kinds[key] = kinds.get(key, 0) + 1
    return kinds


@section('controller')
def check_controller(path):
    from redag.graph.schema import CanvasState
    from redag.learn.rl_trainer import compute_controller_rl_loss
    from redag.learn.sft_trainer import compute_autoregressive_sft_loss
    from redag.models.controller import Controller
    from redag.models.planner import dag_to_dict
    controller = Controller.from_pretrained(str(path), LIGHT['device'], LIGHT['dtype'], None, True)
    controller.max_new_tokens = 24
    kinds = lora_kinds(controller.model)
    assert kinds and any('proj' in k for k in kinds), kinds
    dag = graph()
    state = CanvasState(canvas(), {'cup': 0.9, 'red': 0.2, 'style': 0.8}, {'cup', 'style'})
    sample = controller.sample(state, dag, 'red', seed=3, temperature=0.8)
    assert sample['action'].op in ('add_entity', 'modify_attr', 'fix_relation', 'remove_extra', 'separate_instances')
    assert 80 <= sample['instruction_tokens'] <= 120
    controller.model.train()
    logits = controller.forward_sample(sample, action_only=True)
    length = int(sample['action_mask'].sum())
    ids = sample['token_ids'][:, -(length + 1):].to(logits.device)
    loss, metrics = compute_controller_rl_loss(logits, logits.detach(), logits.detach(), ids, torch.ones((1, length), device=logits.device),
                                               torch.tensor([0.5], device=logits.device))
    loss.backward()
    grads = sum(float(p.grad.abs().sum()) for p in controller.lora_parameters() if p.grad is not None)
    record = {'graph': json.loads(json.dumps(dag_to_dict(dag))),
              'image': canvas(), 'scores': state.scores, 'target_node_id': 'red',
              'action': {'op': 'modify_attr', 'box': [0.1, 0.1, 0.6, 0.6], 'instruction': 'Make the cup red.'}}
    controller.model.zero_grad(set_to_none=True)
    if LIGHT['on']:  # 150 x 248k-vocab logits do not fit next to the model in 2 GiB; build the SFT prompt only
        from redag.models.controller import bounded_instruction
        text, count, _ = bounded_instruction(record['action']['instruction'], controller.processor.tokenizer)
        inputs = controller.encode(controller.messages(state, dag, 'red'))
        sft, same = torch.tensor(float(count)), None
        assert inputs['input_ids'].shape[1] > 0 and 80 <= count <= 120
    else:
        sft = compute_autoregressive_sft_loss(*controller.sft_batch([record]))
        sft.backward()
        out = path.parent / 'controller_adapter'
        snapshot = controller.snapshot()
        controller.save_adapter(out)
        reloaded = Controller.from_pretrained(str(path), LIGHT['device'], LIGHT['dtype'], str(out), False)
        same = all(torch.equal(snapshot[k], v) for k, v in reloaded.snapshot().items())
    return {'lora_modules': kinds, 'action_tokens': length, 'rl_loss': round(loss.item(), 4), 'grad_sum': grads,
            'sft_loss': round(sft.item(), 4), 'adapter_roundtrip': same, 'ratio': metrics['ctrl_mean_ratio']}


@section('planner')
def check_planner(path):
    from redag.learn.sft_trainer import compute_autoregressive_sft_loss
    from redag.models.planner import Planner, dag_to_dict
    planner = Planner.from_pretrained(str(path), LIGHT['device'], LIGHT['dtype'], None, True)
    response = dag_to_dict(graph())
    saucer = {'nodes': [{'node_id': 'saucer', 'node_type': 'entity', 'question': 'Is there a saucer?', 'entity_phrase': 'saucer',
                         'scope': 'local'}]}
    if LIGHT['on']:
        with torch.no_grad():
            loss = compute_autoregressive_sft_loss(*planner.sft_batch([{'user_prompt': 'A saucer.', 'response': saucer}]))
    else:
        logits, labels = planner.sft_batch([{'user_prompt': 'A red cup in a watercolor painting.', 'response': response},
                                            {'user_prompt': 'Also add a saucer.', 'previous_dag': response, 'response': saucer}])
        loss = compute_autoregressive_sft_loss(logits, labels)
        loss.backward()
    inputs = planner.encode(planner.messages('A red cup.'))
    generated = planner.model.generate(**inputs, max_new_tokens=8, do_sample=False)
    return {'sft_loss': round(loss.item(), 4), 'prompt_tokens': int(inputs['input_ids'].shape[1]),
            'generated': int(generated.shape[1] - inputs['input_ids'].shape[1]), 'lora_modules': lora_kinds(planner.model)}


@section('verifier')
def check_verifier(path):
    from redag.verify.verifier import TwoStageVerifier, load_vlm_scorer
    scorer = load_vlm_scorer(str(path), LIGHT['device'], LIGHT['dtype'])
    value = scorer(canvas(), 'Is there a cup?')
    small = scorer(canvas(20, 9), 'Is the cup red?')  # tiny crops must still be accepted by the processor
    verifier = TwoStageVerifier(scorer)
    scores = verifier.verify_subgraph_after_edit(canvas(), graph(), {'cup': 0.0, 'red': 0.0, 'style': 0.0}, {'cup', 'red', 'style'})
    assert 0 <= value <= 1 and 0 <= small <= 1 and set(scores) == {'cup', 'red', 'style'}
    return {'score': value, 'tiny_crop_score': small, 'calls': verifier.call_counter}


@section('editor')
def check_editor(path):
    from redag.graph.schema import ControllerAction
    from redag.learn.rl_trainer import compute_editor_rl_loss
    from redag.learn.sft_trainer import compute_editor_masked_flow_sft_loss
    from redag.models.editor import FluxEditor
    editor = FluxEditor.from_pretrained(str(path), LIGHT['device'], torch.float32, None, True)
    editor.num_steps = 6
    kinds = lora_kinds(editor.transformer)
    assert kinds, 'no LoRA modules were created in the transformer'
    image = canvas()
    action = ControllerAction('modify_attr', (0.11, 0.23, 0.57, 0.81), 'Make the cup red.')  # 43 x 58 px: not a multiple of 16
    conditioning = editor.prepare(image, action)
    trajectory = editor.sample(image, action, seed=5, conditioning=conditioning)
    crop = trajectory['edited_crop']
    expected = editor._crop(image, action).shape
    assert tuple(crop.shape) == tuple(expected), (crop.shape, expected)
    rendered = editor.render(image, action, seed=5)
    assert torch.allclose(rendered.cpu().float(), crop.cpu().float(), atol=1e-3)  # render() returns on the image's device
    editor.train()
    loss, metrics = compute_editor_rl_loss(editor.dit_model, [trajectory], **conditioning, advantages=torch.tensor([0.7], device=editor.device),
                                           lora_params_current=editor.lora_parameters(),
                                           lora_params_sft_snapshot=[p.detach().clone() for p in editor.lora_parameters()],
                                           dt=1 / editor.num_steps)
    loss.backward()
    source, target = canvas(50, 70, 1), canvas(50, 70, 2)
    mask = torch.zeros(50, 70)
    mask[10:30, 20:50] = 1
    prepared = editor.prepare_sft(source, target, 'Make the cup red.', mask)
    sft = compute_editor_masked_flow_sft_loss(editor.dit_model, **prepared, alpha_mask=3.0)
    sft.backward()
    grads = sum(float(p.grad.abs().sum()) for p in editor.lora_parameters() if p.grad is not None)
    if not LIGHT['on']:  # the render_canvases.py call; needs more than the 2 GiB of no-GPU mode
        generators = [torch.Generator(device=LIGHT['device']).manual_seed(i) for i in range(2)]
        images = editor.pipeline(prompt=['a red cup', 'a blue ball'], height=64, width=64, num_inference_steps=2,
                                 guidance_scale=4.0, generator=generators).images
        assert len(images) == 2 and images[0].size == (64, 64)
    out = path.parent / 'editor_adapter'
    editor.save_adapter(out)
    reloaded = FluxEditor.from_pretrained(str(path), LIGHT['device'], torch.float32, str(out), False)
    same = all(torch.equal(a.float(), b.float()) for a, b in zip(editor.lora_parameters(), reloaded.lora_parameters()))
    return {'lora_modules': kinds, 'latent_shape': list(conditioning['z_src'].shape), 'crop': list(crop.shape),
            'rl_loss': round(loss.item(), 4), 'sft_loss': round(sft.item(), 4), 'grad_sum': grads, 'adapter_roundtrip': same,
            'batched_cfg': editor.batched_cfg, 'text_embeds': list(conditioning['text_embeds'].shape)}


@section('grounder')
def check_grounder(_):
    """Processor contracts of GroundingDINO and SAM2 with the real processor files (no weights loaded)."""
    import numpy as np
    from PIL import Image
    from transformers import AutoProcessor, Sam2Processor
    rgb = Image.fromarray((np.random.rand(60, 90, 3) * 255).astype('uint8'))
    processor = AutoProcessor.from_pretrained(MODELS / 'IDEA-Research/grounding-dino-base')
    inputs = processor(images=rgb, text='red cup.', do_pad=False, return_tensors='pt')
    queries = 7
    outputs = type('Out', (), {})()
    outputs.logits = torch.randn(1, queries, inputs['input_ids'].shape[1]) * 4
    outputs.pred_boxes = torch.rand(1, queries, 4) * 0.5 + 0.25
    result = processor.post_process_grounded_object_detection(outputs, inputs.input_ids, threshold=0.30, text_threshold=0.25,
                                                              target_sizes=[(60, 90)])[0]
    sam_processor = Sam2Processor.from_pretrained(MODELS / 'facebook/sam2.1-hiera-large')
    sam_inputs = sam_processor(images=rgb.crop((0, 0, 60, 60)), input_boxes=[[[5.0, 5.0, 40.0, 50.0], [1.0, 2.0, 20.0, 30.0]]],
                               return_tensors='pt')
    masks = sam_processor.post_process_masks(torch.randn(1, 2, 1, 256, 256), sam_inputs['original_sizes'])[0]
    masks = masks.reshape(2, 60, 60).bool()
    return {'gdino_inputs': sorted(inputs.keys()), 'boxes': list(result['boxes'].shape), 'sam_inputs': sorted(sam_inputs.keys()),
            'mask_shape': list(masks.shape)}


@section('loops')
def check_loops(paths):
    """SFT, rejection sampling, alternating RL and an evaluation session with every tiny component together."""
    from types import SimpleNamespace
    from redag.harness.environment import ReDAGHarness, run_record
    from redag.learn.rl_trainer import train_alternating
    from redag.learn.sft_trainer import collect_controller_rs, read_records, train_sft
    from redag.models.controller import Controller
    from redag.models.editor import FluxEditor
    from redag.models.planner import dag_to_dict
    from redag.config import ReDAGConfig
    from redag.verify.grounder import save_image
    from redag.verify.verifier import TwoStageVerifier, load_vlm_scorer
    from run_eval_benchmarks import export_case
    work = paths['work'] / 'loops'
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    graph_dict = json.loads(json.dumps(dag_to_dict(graph())))
    rows = []
    for index in range(6):
        save_image(canvas(seed=index), work / f'img{index}.png')
        rows.append({'id': f's{index}', 'image': f'img{index}.png', 'graph': graph_dict})
    (work / 'states.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows[:4]))
    (work / 'dev.jsonl').write_text(json.dumps(rows[4]) + '\n')
    for name in ('src', 'tgt'):
        save_image(canvas(40, 52, 1 if name == 'src' else 2), work / f'e_{name}.png')
    from PIL import Image
    mask = Image.new('L', (52, 40))
    mask.paste(255, (10, 8, 40, 30))
    mask.save(work / 'e_mask.png')
    (work / 'editor.jsonl').write_text(json.dumps({'source': 'e_src.png', 'target': 'e_tgt.png', 'mask': 'e_mask.png',
                                                   'instruction': 'Make the cup red.'}) + '\n')
    device = LIGHT['device']
    editor = FluxEditor.from_pretrained(str(paths['flux']), device, torch.float32, None, True)
    sft_editor = train_sft(editor, read_records(work / 'editor.jsonl'), work / 'sft_editor', 'editor')
    controller = Controller.from_pretrained(str(paths['qwen35']), device, torch.float32, None, True)
    controller.max_new_tokens = 24
    harness = ReDAGHarness(TwoStageVerifier(load_vlm_scorer(str(paths['qwen3vl']), device, torch.float32)), fake_grounder())
    editor.num_steps = 6
    rs = collect_controller_rs(read_records(work / 'states.jsonl'), harness, controller, editor, work / 'rs.jsonl')
    sft_rows = read_records(work / 'rs.jsonl') if (work / 'rs.jsonl').stat().st_size else []
    if not sft_rows:  # random weights rarely produce an accepted edit; exercise the controller trainer anyway
        sft_rows = [dict(rows[0], image=str(work / 'img0.png'), scores={'cup': 0.9, 'red': 0.1, 'style': 0.9}, target_node_id='red',
                         action={'op': 'modify_attr', 'box': [0.1, 0.1, 0.7, 0.7], 'instruction': 'Make the cup red.'})]
    sft_controller = train_sft(controller, sft_rows, work / 'sft_controller', 'controller')
    history = train_alternating(harness, controller, editor, read_records(work / 'states.jsonl'), read_records(work / 'dev.jsonl'),
                                work / 'rl', config=ReDAGConfig(num_denoise_steps=6), rounds=2, batch_size=4, rollout_steps=2)
    checkpoints = json.loads((work / 'rl' / 'stage_checkpoints.json').read_text())
    assert set(checkpoints) == {'zero_shot', 'sft', 'ctrl_rl', 'edit_rl', 'alternating_rl'}, checkpoints
    turns = [{'user_prompt': 'A red cup.', 'graph': graph_dict},
             {'user_prompt': 'Add a saucer.', 'graph': {'nodes': [{'node_id': 'saucer', 'node_type': 'entity', 'question': 'Is there a saucer?',
                                                                    'entity_phrase': 'saucer', 'scope': 'local'}]}}]
    harness.verification_budget, harness.generation_budget = 200, 4
    state, dag, trace = run_record({'image': str(work / 'img5.png'), 'turns': turns}, harness, controller, editor, max_steps=2, seed=7)
    case = export_case(state, dag, trace, work / 'eval_images', 'case_0', 'test/redag_full/4', harness.last_run, harness.config)
    return {'editor_sft_steps': sft_editor['steps'], 'rs': rs, 'controller_sft_steps': sft_controller['steps'],
            'rl_steps': len(history), 'rl_stages': sorted({h['stage'] for h in history}), 'eval_steps': len(trace),
            'eval_nodes': len(dag.nodes), 'stop_reason': harness.last_run.get('stop_reason'), 'case_views': len(case['views'])}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--work', default=str(ROOT / 'runs' / 'wiring'))
    parser.add_argument('--only', nargs='+', choices=('controller', 'planner', 'verifier', 'editor', 'grounder', 'loops'))
    parser.add_argument('--rebuild', action='store_true')
    parser.add_argument('--light', action='store_true', help='AutoDL no-GPU mode (2 GiB): bf16, forward-only SFT, no loops')
    parser.add_argument('--device', default='cpu', help='cuda:0 also exercises the GPU kernels (flash-linear-attention, SDPA)')
    args = parser.parse_args()
    LIGHT.update(on=args.light, dtype=torch.bfloat16 if args.light else torch.float32, device=args.device)
    torch.manual_seed(0)
    work = Path(args.work)
    paths = {'work': work, 'qwen35': work / 'tiny_qwen35', 'qwen3vl': work / 'tiny_qwen3vl', 'flux': work / 'tiny_flux'}
    builders = {'qwen35': (tiny_qwen35, MODELS / 'Qwen/Qwen3.5-9B'), 'qwen3vl': (tiny_qwen3vl, MODELS / 'Qwen/Qwen3-VL-8B-Instruct'),
                'flux': (tiny_flux, MODELS / 'black-forest-labs/FLUX.2-klein-base-9B')}
    for key, (build, source) in builders.items():
        if args.rebuild or not (paths[key] / ('model_index.json' if key == 'flux' else 'config.json')).exists() or \
                not any(paths[key].rglob('*.safetensors')):
            shutil.rmtree(paths[key], ignore_errors=True)
            build(source, paths[key])
            print('built', key, flush=True)
    wanted = args.only or ['controller', 'planner', 'verifier', 'editor', 'grounder'] + ([] if args.light else ['loops'])
    for name in wanted:
        {'controller': lambda: check_controller(paths['qwen35']), 'planner': lambda: check_planner(paths['qwen35']),
         'verifier': lambda: check_verifier(paths['qwen3vl']), 'editor': lambda: check_editor(paths['flux']),
         'grounder': lambda: check_grounder(None), 'loops': lambda: check_loops(paths)}[name]()
    (work / 'wiring_report.json').write_text(json.dumps(RESULTS, indent=2))
    failed = [k for k, v in RESULTS.items() if not v['ok']]
    print('FAILED: ' + ', '.join(failed) if failed else 'ALL WIRING CHECKS PASSED', flush=True)
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
