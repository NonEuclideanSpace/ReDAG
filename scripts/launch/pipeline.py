"""The whole ReDAG pipeline as one resumable job graph.  Normally used through scripts/launch/go.sh.

  python scripts/launch/pipeline.py run [--retry-failed] [--only REGEX] [--allow-cpu-phase-on-gpu]
  python scripts/launch/pipeline.py status [--brief] [--json]
  python scripts/launch/pipeline.py stop [--drain]
  python scripts/launch/pipeline.py plan [--gpus N ...] [--from-scratch]
  python scripts/launch/pipeline.py list | logs JOB [-n N] [-f] | reset JOB... [--dependents]
  python scripts/launch/pipeline.py bundle export {data,results} [--out DIR] [--with-adapters] [--split 3900]
  python scripts/launch/pipeline.py bundle import FILE...

Settings come from the environment (scripts/launch/env.sh).  Settings that change results are recorded in
runs/state/config.json on the first run; a later run with different values is refused (--accept-config-change
overrides), so a restarted or handed-over run cannot silently mix settings.
"""
import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import orchestrator as orch  # noqa: E402
from orchestrator import HOUR, Job, Notifier, Scheduler, lock_held, query_gpus, read_json, simulate  # noqa: E402

ROOT = Path(os.environ.get('REDAG_ROOT', HERE.parents[1]))
STAGES = ('zero_shot', 'sft', 'ctrl_rl', 'edit_rl', 'alternating_rl')
# where each experiment-3 stage comes from: (stage file, share of rl_main after which it exists)
RL_STAGE_SHARE = {'ctrl_rl': 0.25, 'edit_rl': 0.5, 'alternating_rl': 1.0}


def env(name, default):
    value = os.environ.get(name)
    return default if value in (None, '') else value


def settings():
    models = Path(env('MODELS', '/root/autodl-tmp/models'))
    runs = Path(env('RUNS', ROOT / 'runs'))
    s = dict(
        PY=env('PY', sys.executable), ROOT=str(ROOT), RUNS=str(runs), DATA=str(ROOT / 'data' / 'final'),
        PLANNER=env('PLANNER', str(models / 'Qwen/Qwen3.8-27B')), CONTROLLER=env('CONTROLLER', str(models / 'Qwen/Qwen3.5-9B')),
        EDITOR=env('EDITOR', str(models / 'black-forest-labs/FLUX.2-klein-base-9B')),
        VERIFIER=env('VERIFIER', str(models / 'Qwen/Qwen3-VL-8B-Instruct')),
        DETECTOR=env('DETECTOR', str(models / 'IDEA-Research/grounding-dino-base')),
        SAM=env('SAM', str(models / 'facebook/sam2.1-hiera-large')),
        CANVAS_DIR=env('CANVAS_DIR', str(ROOT / 'data/raw/corebench_images/FLUX.2-klein-9B')),
    )
    ints = dict(STEPS_RL=16, STEPS_EVAL=28, MAX_STEPS=4, GEN_BUDGET=8, VER_BUDGET=400, ROLLOUT_STEPS=3, PER_TIER=100,
                N_RS_STATES=1000, N_RL_TRAIN=128, N_RL_DEV=16, N_EXP2=64, EXP2_MAX_STEPS=3, CANVAS_SIZE=768, CANVAS_STEPS=50,
                EDITOR_EPOCHS=1, PLANNER_EPOCHS=1, CONTROLLER_EPOCHS=2, ANNOTATE_SHARDS=8, ANNOTATE_BATCH=8, CANVAS_SHARDS=4,
                RS_SHARDS=4, EXP1_SHARDS=12, EXP3_SHARDS=12, EXP1_EXTRA=0, RUN_EXP2=1, FETCH_WORKERS=6, WORKERS=1, SKIP_PREFLIGHT=0,
                DOWNLOAD_WORKERS=16)
    for key, default in ints.items():
        s[key] = int(env(key, default))
    s.update(CKPT_MINUTES=float(env('CKPT_MINUTES', 20)), RS_LAYOUT=env('RS_LAYOUT', 'shard'), EVAL_LAYOUT=env('EVAL_LAYOUT', 'shard'),
             RL_LAYOUT=env('RL_LAYOUT', 'auto'),
             EXPERIMENTS=env('EXPERIMENTS', '1 2 3').split(), MB_SPLITS=env('MB_SPLITS', 'dev train').split())
    return s


# settings that change what is computed; layouts, shard placement and the experiment selection do not
RESULT_KEYS = ('PLANNER', 'CONTROLLER', 'EDITOR', 'VERIFIER', 'DETECTOR', 'SAM', 'STEPS_RL', 'STEPS_EVAL', 'MAX_STEPS', 'GEN_BUDGET',
               'VER_BUDGET', 'ROLLOUT_STEPS', 'PER_TIER', 'N_RS_STATES', 'N_RL_TRAIN', 'N_RL_DEV', 'N_EXP2', 'EXP2_MAX_STEPS',
               'CANVAS_SIZE', 'CANVAS_STEPS', 'EDITOR_EPOCHS', 'PLANNER_EPOCHS', 'CONTROLLER_EPOCHS', 'ANNOTATE_SHARDS', 'RS_SHARDS',
               'EXP1_SHARDS', 'EXP3_SHARDS', 'EXP1_EXTRA')


def code_version():
    """Hash of the code that jobs run (not docs or tests): a change re-runs the GPU preflight and retries failed jobs."""
    digest = hashlib.sha256()
    for path in sorted([*ROOT.glob('redag/**/*.py'), *ROOT.glob('scripts/**/*.py'), *ROOT.glob('scripts/**/*.sh')]):
        if '__pycache__' in path.parts:
            continue
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()[:12]


def rl_on_one_gpu(s, gpu_info):
    """RL keeps controller, verifier, grounder and editor together on one GPU when every GPU has >= 90 GiB (estimated
    peak ~80 GiB: 68.4 GiB of weights plus one trainable LoRA at a time); 80 GB cards use two GPUs per run."""
    if s['RL_LAYOUT'] in ('single', 'split'):
        return s['RL_LAYOUT'] == 'single'
    return bool(gpu_info) and min(g['total'] for g in gpu_info) >= 90 * 1024


def environment_version():
    """Versions of the packages the GPU preflight validates (read from package metadata, no imports)."""
    from importlib import metadata
    versions = []
    for name in ('torch', 'transformers', 'diffusers', 'peft', 'accelerate', 'bitsandbytes', 'flash-linear-attention'):
        try:
            versions.append(f'{name}={metadata.version(name)}')
        except metadata.PackageNotFoundError:
            versions.append(f'{name}=-')
    return hashlib.sha256(';'.join(versions).encode()).hexdigest()[:12]


def build_jobs(s, gpu_signature='', gpu_info=None):
    py, data, runs = s['PY'], Path(s['DATA']), Path(s['RUNS'])
    sft, out, D = runs / 'sft', runs / 'eval', 'scripts/data'
    runtime = ['--controller_model', s['CONTROLLER'], '--editor_model', s['EDITOR'], '--verifier_model', s['VERIFIER'],
               '--detector_model', s['DETECTOR'], '--sam_model', s['SAM'], '--planner_model', s['PLANNER']]
    budget = ['--max_steps', s['MAX_STEPS'], '--generation_budget', s['GEN_BUDGET'], '--verification_budget', s['VER_BUDGET'],
              '--num_denoise_steps', s['STEPS_EVAL']]
    me = [py, str(HERE / 'pipeline.py')]
    jobs = []

    def add(name, cmd, gpus=0, deps=(), hours=0.05, group='', **extra):
        if gpus and not s['SKIP_PREFLIGHT']:
            deps = ('preflight', *deps)  # every GPU job waits for the preflight of the current code
        jobs.append(Job(name, [str(c) for c in cmd] if isinstance(cmd, list) else cmd, gpus=gpus, deps=tuple(deps),
                        hours=hours, group=group, **extra))

    # ---- phase 1: CPU only (AutoDL no-GPU mode); same job names as 01_prepare_data.sh
    g = '1 data (CPU)'
    add('download_models', [py, 'scripts/launch/download_models.py', '--workers', s['DOWNLOAD_WORKERS']], hours=1.0, group=g)
    add('fetch_corebench', [py, f'{D}/fetch_sources.py', '--what', 'corebench'], hours=0.1, group=g)
    add('fetch_benchmarks', [py, f'{D}/fetch_sources.py', '--what', 'lgbench', 'longbench'], hours=0.2, group=g)
    add('fetch_corebench_images', [py, f'{D}/fetch_sources.py', '--what', 'corebench_images', '--archives', 'FLUX.2-klein-9B'],
        hours=0.3, group=g)
    for split in s['MB_SPLITS']:
        add(f'fetch_magicbrush_{split}', [py, f'{D}/fetch_sources.py', '--what', 'magicbrush', '--splits', split,
                                          '--workers', s['FETCH_WORKERS']], hours=1.5 if split == 'train' else 0.3, group=g)
        add(f'prep_magicbrush_{split}', [py, f'{D}/prep_magicbrush.py', '--splits', split, '--size', 512, '--workers', s['WORKERS']],
            deps=(f'fetch_magicbrush_{split}',), hours=2.0 if split == 'train' else 0.3, group=g)
    canvas = shlex.quote(s['CANVAS_DIR'])
    # sessions without canvases first (enough for graph annotation, which only reads text), then again with the
    # CoReBench canvases once that archive is in; ids are identical and build_sets takes canvases from the latter
    add('prep_checklists_text', [py, f'{D}/prep_checklists.py', '--what', 'corebench', 'lgbench', 'longbench'],
        deps=('fetch_corebench', 'fetch_benchmarks'), hours=0.05, group=g)
    add('prep_checklists', f'{shlex.quote(py)} {D}/prep_checklists.py --what corebench lgbench longbench '
        f'$([ -d {canvas} ] && echo --canvas_dir {canvas})', deps=('prep_checklists_text', 'fetch_corebench_images'),
        hours=0.1, group=g)
    add('plan_annotation', [py, f'{D}/plan_annotation.py'],
        deps=tuple(f'prep_magicbrush_{x}' for x in s['MB_SPLITS']) + ('prep_checklists',), group=g)
    # plan_annotation picks each (pool, source) independently, so the checklist benchmarks' ids are final before
    # MagicBrush is ready: their annotation can start while MagicBrush still downloads
    sessions = 'data/processed/sessions'
    checklists = f'$(ls {sessions}/corebench.jsonl {sessions}/lgbench.jsonl {sessions}/longbench.jsonl 2>/dev/null)'
    add('plan_annotation_checklists', f'{shlex.quote(py)} {D}/plan_annotation.py --sessions {checklists} '
        f'--output {sessions}/annotate_ids_checklists.txt', deps=('prep_checklists_text',), group=g)
    cpu_phase = [j.name for j in jobs]

    # ---- GPU preflight: once per machine and software environment.  Code changes do not re-run it (that would stall every
    # GPU job for minutes after each fix); they are covered by the unit tests and re-checked with `go.sh preflight`.
    if not s['SKIP_PREFLIGHT']:
        jobs.append(Job('preflight', ['bash', 'scripts/launch/00_check_env.sh'], gpus=1, hours=0.15, group='2 preflight', deps=('download_models',),
                        stamp=f'{environment_version()}|{gpu_signature}', env={'MIN_GPUS': '1'}, attempts=1))

    # ---- phase 2: graph annotation, sets, canvases, state grounding
    g = '3 data (GPU)'
    shards = s['ANNOTATE_SHARDS']

    def annotate(files, ids, k):
        return (f'{shlex.quote(py)} {D}/annotate_graphs.py --sessions {files} --ids {ids} --model {shlex.quote(s["PLANNER"])} '
                f'--device cuda:0 --shard {k} --num_shards {shards} --batch_size {s["ANNOTATE_BATCH"]}')
    for k in range(shards):  # same shard k in both passes writes the same files, so the second pass skips the first's ids
        add(f'annotate_checklists_shard{k}', annotate(checklists, f'{sessions}/annotate_ids_checklists.txt', k),
            gpus=1, deps=('plan_annotation_checklists',), hours=3.6 / shards, group=g, family='annotate')
        add(f'annotate_shard{k}', annotate(f'{sessions}/*.jsonl', f'{sessions}/annotate_ids.txt', k),
            gpus=1, deps=('plan_annotation', f'annotate_checklists_shard{k}'), hours=2.4 / shards, group=g, family='annotate')
    add('build_sets', [py, f'{D}/build_sets.py', '--per_tier', s['PER_TIER'], '--n_rs_states', s['N_RS_STATES'],
                       '--n_rl_train', s['N_RL_TRAIN'], '--n_rl_dev', s['N_RL_DEV']],
        deps=tuple(f'annotate_shard{k}' for k in range(shards)) + ('prep_checklists',), group=g)
    canvases = tuple(f'canvases_shard{k}' for k in range(s['CANVAS_SHARDS']))
    for k, name in enumerate(canvases):
        add(name, [py, f'{D}/render_canvases.py', '--model', s['EDITOR'], '--device', 'cuda:0', '--shard', k,
                   '--num_shards', s['CANVAS_SHARDS'], '--size', s['CANVAS_SIZE'], '--steps', s['CANVAS_STEPS']],
            gpus=1, deps=('build_sets',), hours=2.0 / len(canvases), group=g, family='canvases')
    add('precompute_rl_states', [py, f'{D}/precompute_states.py', '--files', data / 'rl_train.jsonl', data / 'rl_dev.jsonl',
                                 '--verifier_model', s['VERIFIER'], '--detector_model', s['DETECTOR'], '--sam_model', s['SAM'],
                                 '--device', 'cuda:0'], gpus=1, deps=canvases, hours=0.5, group=g)
    add('check_data_phase2', [py, f'{D}/check_data.py'], deps=('precompute_rl_states',), group=g)

    # ---- phase 3: supervised stages and rejection sampling
    g = '4 SFT'
    add('sft_editor', [py, 'scripts/run_sft_editor.py', '--data', data / 'editor_sft_train.jsonl', '--output', sft / 'editor',
                       '--model', s['EDITOR'], '--device', 'cuda:0', '--epochs', s['EDITOR_EPOCHS'], '--checkpoint_minutes', s['CKPT_MINUTES']],
        gpus=1, deps=('prep_magicbrush_train',) if 'train' in s['MB_SPLITS'] else (), hours=6.0, group=g)
    add('sft_planner', [py, 'scripts/run_sft_planner.py', '--data', data / 'planner_sft.jsonl', '--output', sft / 'planner',
                        '--model', s['PLANNER'], '--device', 'cuda:0', '--epochs', s['PLANNER_EPOCHS'], '--checkpoint_minutes', s['CKPT_MINUTES']],
        gpus=1, deps=('build_sets',), hours=4.0, group=g)
    add('eval_planner_zero_shot', [py, 'scripts/run_eval_planner.py', '--data', data / 'planner_sft_dev.jsonl', '--model', s['PLANNER'],
                                   '--device', 'cuda:0', '--output', out / 'planner_zero_shot.json'], gpus=1, deps=('build_sets',), hours=0.5, group=g)
    add('eval_planner_sft', [py, 'scripts/run_eval_planner.py', '--data', data / 'planner_sft_dev.jsonl', '--model', s['PLANNER'],
                             '--adapter', sft / 'planner', '--device', 'cuda:0', '--output', out / 'planner_sft.json'],
        gpus=1, deps=('sft_planner',), hours=0.5, group=g)
    rs_common = [py, 'scripts/collect_controller_rs.py', *runtime, '--editor_adapter', sft / 'editor', '--verifier_device', 'cuda:0',
                 '--num_denoise_steps', s['STEPS_EVAL']]
    if s['RS_LAYOUT'] == 'split':
        add('rs_single', rs_common + ['--device', 'cuda:0', '--editor_device', 'cuda:1', '--data', data / 'controller_rs_states.jsonl',
                                      '--output', data / 'controller_sft.jsonl'], gpus=2, deps=('sft_editor',) + canvases, hours=2.0, group=g)
        rs_done = 'rs_single'
    else:
        parts = s['RS_SHARDS']
        add('rs_split', me + ['_split', data / 'controller_rs_states.jsonl', parts], deps=('build_sets',), group=g)
        for k in range(parts):
            add(f'rs_part{k}', rs_common + ['--device', 'cuda:0', '--editor_device', 'cuda:0',
                                            '--data', data / f'controller_rs_states.part{k}.jsonl', '--output', data / f'controller_sft.part{k}.jsonl',
                                            '--seed', 42 + k * 100000],
                gpus=1, deps=('sft_editor', 'rs_split') + canvases, hours=4.0 / parts, group=g, family='rs')
        add('rs_merge', me + ['_concat', data / 'controller_sft.jsonl'] + [data / f'controller_sft.part{k}.jsonl' for k in range(parts)],
            deps=tuple(f'rs_part{k}' for k in range(parts)), group=g)
        rs_done = 'rs_merge'
    add('check_controller_sft', [py, f'{D}/check_data.py', '--only', 'controller_sft'], deps=(rs_done,), group=g)
    add('sft_controller', [py, 'scripts/run_sft_controller.py', '--data', data / 'controller_sft.jsonl', '--output', sft / 'controller',
                           '--model', s['CONTROLLER'], '--device', 'cuda:0', '--epochs', s['CONTROLLER_EPOCHS'],
                           '--checkpoint_minutes', s['CKPT_MINUTES']], gpus=1, deps=('check_controller_sft',), hours=2.0, group=g)

    # ---- phase 4: alternating RL (controller + verifier + grounder on one GPU, editor on the other)
    g = '5 RL'
    single = rl_on_one_gpu(s, gpu_info)
    rl_gpus = 1 if single else 2
    rl = [py, 'scripts/run_rl_alternating.py', *runtime, '--controller_adapter', sft / 'controller', '--editor_adapter', sft / 'editor',
          '--device', 'cuda:0', '--verifier_device', 'cuda:0', '--editor_device', 'cuda:0' if single else 'cuda:1',
          '--num_denoise_steps', s['STEPS_RL'], '--rollout_steps', s['ROLLOUT_STEPS'], '--dev_data', data / 'rl_dev.jsonl']
    trained = ('sft_controller', 'sft_editor', 'check_data_phase2')
    add('rl_main', rl + ['--data', data / 'rl_train.jsonl', '--reward_mode', 'subgraph_credit_ours', '--output', runs / 'rl' / 'main'],
        gpus=rl_gpus, deps=trained, hours=12.0, group=g)
    exp2_modes = ('subgraph_credit_ours', 'step_full_table', 'terminal_broadcast')
    want_exp2 = '2' in s['EXPERIMENTS'] and s['RUN_EXP2']
    if want_exp2:
        add('exp2_subset', me + ['_head', data / 'rl_train.jsonl', data / 'rl_train_exp2.jsonl', s['N_EXP2']], deps=('check_data_phase2',), group=g)
        for mode in exp2_modes:
            add(f'rl_exp2_{mode}', rl + ['--data', data / 'rl_train_exp2.jsonl', '--reward_mode', mode, '--output', runs / 'rl' / 'exp2' / mode],
                gpus=rl_gpus, deps=trained + ('exp2_subset',), hours=6.0, group=g, family='rl_exp2')
        add('exp2_logs', me + ['_exp2_logs', runs / 'rl' / 'exp2'], deps=tuple(f'rl_exp2_{m}' for m in exp2_modes), group=g)

    # ---- phase 5: experiments 1-3 under one fixed per-session budget
    g = '6 eval'
    split = s['EVAL_LAYOUT'] == 'split'
    eval_gpus = 2 if split else 1
    devices = ['--device', 'cuda:0', '--verifier_device', 'cuda:0', '--editor_device', 'cuda:1' if split else 'cuda:0']
    evaluate = [py, 'scripts/run_eval_benchmarks.py', *runtime]
    if '1' in s['EXPERIMENTS']:
        modes = ['flat_full', 'tree_rollback_only', 'redag_full'] + (['spatial_local', 'spatial_no_gates'] if s['EXP1_EXTRA'] else [])
        n = s['EXP1_SHARDS']
        for k in range(n):
            add(f'exp1_s{k}', evaluate + budget + ['--exp_id', 1, '--data', data / 'eval_main.jsonl', '--exp1_modes', *modes,
                                                   '--controller_adapter', sft / 'controller', '--editor_adapter', sft / 'editor', *devices,
                                                   '--shard', k, '--num_shards', n, '--output', out / f'exp1.s{k}.json'],
                gpus=eval_gpus, deps=trained, hours=23.0 * len(modes) / 3 / n * eval_gpus, group=g, family='exp1')
        add('merge_exp1', [py, 'scripts/merge_eval_results.py', '--inputs', *[out / f'exp1.s{k}.json' for k in range(n)],
                           '--output', out / 'exp1.json'], deps=tuple(f'exp1_s{k}' for k in range(n)), group=g)
    if want_exp2:
        add('exp2_eval', evaluate + ['--exp_id', 2, '--data', data / 'eval_credit.jsonl', '--training_logs', runs / 'rl/exp2/training_logs.json',
                                     '--controller_adapter', sft / 'controller', '--editor_adapter', sft / 'editor',
                                     '--max_steps', s['EXP2_MAX_STEPS'], '--generation_budget', s['GEN_BUDGET'],
                                     '--verification_budget', s['VER_BUDGET'], '--num_denoise_steps', s['STEPS_EVAL'], *devices,
                                     '--output', out / 'exp2.json'], gpus=eval_gpus, deps=trained + ('exp2_logs',), hours=3.0 * eval_gpus, group=g)
    if '3' in s['EXPERIMENTS']:
        n, parts = s['EXP3_SHARDS'], []
        pre_rl = out / 'exp3_pre_rl_stages.json'
        main_stages = runs / 'rl' / 'main' / 'stage_checkpoints.json'
        add('exp3_stage_file', me + ['_stage_file', pre_rl], group=g)
        for stage in STAGES:
            extra = {}
            if stage in RL_STAGE_SHARE:
                source, deps = main_stages, ('check_data_phase2',)
                extra = dict(ready=lambda stage=stage: stage_ready(main_stages, stage), waiting_for=f'rl_main checkpoint "{stage}"',
                             soft=(('rl_main', RL_STAGE_SHARE[stage]),))
            else:
                source, deps = pre_rl, ('exp3_stage_file', 'check_data_phase2') + (('sft_controller', 'sft_editor') if stage == 'sft' else ())
            for k in range(n):
                name = f'exp3_{stage}_s{k}'
                parts.append(name)
                add(name, evaluate + budget + ['--exp_id', 3, '--data', data / 'eval_main.jsonl', '--stage_checkpoints', source,
                                               '--stages', stage, *devices, '--shard', k, '--num_shards', n,
                                               '--output', out / f'exp3_{stage}.s{k}.json'],
                    gpus=eval_gpus, deps=deps, hours=23.0 / n * eval_gpus, group=g, family='exp3', **extra)
        add('merge_exp3', [py, 'scripts/merge_eval_results.py', '--inputs',
                           *[out / f'exp3_{stage}.s{k}.json' for stage in STAGES for k in range(n)], '--output', out / 'exp3.json'],
            deps=tuple(parts), group=g)
    finals = [j.name for j in jobs if j.name in ('merge_exp1', 'exp2_eval', 'merge_exp3', 'eval_planner_zero_shot', 'eval_planner_sft', 'rl_main')]
    add('report', me + ['_report'], deps=tuple(finals), group='7 report')
    return jobs, cpu_phase


MILESTONES = {
    'plan_annotation': 'CPU data phase finished: switch the instance to GPU mode and run go.sh',
    'check_data_phase2': 'all data built and checked (graphs, sets, canvases, RL states)',
    'sft_controller': 'all supervised stages finished (editor, planner, rejection sampling, controller)',
    'rl_main': 'main alternating RL finished; experiment-3 RL stages can all run now',
    'merge_exp1': 'experiment 1 merged: runs/eval/exp1.json', 'exp2_eval': 'experiment 2 finished: runs/eval/exp2.json',
    'merge_exp3': 'experiment 3 merged: runs/eval/exp3.json (+ blind pairs for human rating)',
    'report': 'pipeline finished: runs/RESULTS.md',
}


def stage_ready(path, stage):
    mapping = read_json(path, {})
    entry = mapping.get(stage) if isinstance(mapping, dict) else None
    return bool(entry) and all((Path(path).parent / p).is_dir() for p in entry.values() if p)


# --------------------------------------------------------------------------- helpers run as small CPU jobs
def helper(argv):
    name, args = argv[0], argv[1:]
    if name == '_split':  # round-robin parts of a JSONL file (same rule as the old 03_sft.sh)
        src, parts = Path(args[0]), int(args[1])
        lines = [line for line in src.read_text(encoding='utf-8').splitlines() if line.strip()]
        for k in range(parts):
            dst = src.with_name(f'{src.stem}.part{k}.jsonl')
            orch.atomic_write(dst, ''.join(line + '\n' for i, line in enumerate(lines) if i % parts == k))
        print(f'{len(lines)} rows -> {parts} parts')
    elif name == '_concat':
        dst, parts = Path(args[0]), [Path(p) for p in args[1:]]
        rows = [line for p in parts for line in p.read_text(encoding='utf-8').splitlines() if line.strip()]
        orch.atomic_write(dst, ''.join(line + '\n' for line in rows))
        print(f'accepted rejection-sampling rows: {len(rows)}')
    elif name == '_head':
        src, dst, n = Path(args[0]), Path(args[1]), int(args[2])
        rows = [line for line in src.read_text(encoding='utf-8').splitlines() if line.strip()][:n]
        orch.atomic_write(dst, ''.join(line + '\n' for line in rows))
        print(f'{len(rows)} parents in {dst.name}')
    elif name == '_exp2_logs':
        root = Path(args[0])
        modes = ('terminal_broadcast', 'step_full_table', 'subgraph_credit_ours')
        orch.atomic_write(root / 'training_logs.json', json.dumps({m: f'{m}/metrics.jsonl' for m in modes}, indent=2))
        print('wrote', root / 'training_logs.json')
    elif name == '_stage_file':  # zero-shot and SFT stages of experiment 3, usable before RL starts
        dst = Path(args[0])
        dst.parent.mkdir(parents=True, exist_ok=True)
        orch.atomic_write(dst, json.dumps({'zero_shot': {'controller': None, 'editor': None},
                                           'sft': {'controller': '../sft/controller', 'editor': '../sft/editor'}}, indent=2))
        print('wrote', dst)
    elif name == '_report':
        write_report(settings())
    else:
        raise SystemExit(f'unknown helper {name}')


def write_report(s):
    runs = Path(s['RUNS'])
    lines = [f'# ReDAG results ({time.strftime("%F %T")})', '']
    for name in ('exp1', 'exp2', 'exp3', 'planner_zero_shot', 'planner_sft'):
        path = runs / 'eval' / f'{name}.json'
        if not path.exists():
            lines.append(f'- {name}: missing')
            continue
        result = read_json(path, {})
        detail = {k: result[k] for k in ('tier_counts', 'summary', 'metrics') if k in result}
        lines.append(f'- {name}: `{path.relative_to(ROOT)}` ' + json.dumps(detail, ensure_ascii=False)[:1500])
    for metrics in sorted(runs.glob('rl/**/metrics.jsonl')):
        rows = orch.read_jsonl(metrics)
        if rows:
            last = rows[-1]
            lines.append(f'- RL `{metrics.parent.relative_to(runs)}`: {len(rows)} steps, final dev acceptance '
                         f'{last.get("dev_acceptance_p")}, gain {last.get("dev_acceptance_gain")}')
    spent = sum(r.get('seconds', 0) * max(len(str(r.get('gpus', '')).split(',')) if r.get('gpus') else 0, 0)
                for r in orch.read_jsonl(runs / 'logs' / 'timings.jsonl')) / HOUR
    lines += ['', f'GPU hours used (all attempts): {spent:.1f}', '', 'Human evaluation: see scripts/launch/README.md (blind pairs in runs/eval).']
    (runs / 'RESULTS.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))


# --------------------------------------------------------------------------- guards
def legacy_scripts():
    """Old phase scripts (00-05, smoke) still running: they would compete for the same outputs."""
    found = []
    for proc in Path('/proc').glob('[0-9]*'):
        try:
            cmd = (proc / 'cmdline').read_bytes().replace(b'\0', b' ').decode('utf-8', 'replace')
        except OSError:
            continue
        if re.search(r'scripts/launch/(0[0-5]_\w+|smoke_test)\.sh', cmd) and 'pipeline.py' not in cmd:
            found.append(f'{proc.name}: {cmd.strip()[:120]}')
    return found


def check_config(s, accept):
    path = Path(s['RUNS']) / 'state' / 'config.json'
    current = {k: s[k] for k in RESULT_KEYS}
    saved = read_json(path)
    if saved is None:
        path.parent.mkdir(parents=True, exist_ok=True)
        orch.atomic_write(path, json.dumps(dict(settings=current, history=[], created=time.strftime('%F %T')), indent=2))
        return []
    diff = [f'{k}: {saved["settings"].get(k)} -> {v}' for k, v in current.items() if saved['settings'].get(k) != v]
    if diff and accept:
        saved['history'].append(dict(time=time.strftime('%F %T'), old=saved['settings']))
        saved['settings'] = current
        orch.atomic_write(path, json.dumps(saved, indent=2))
        return []
    return diff


def scheduler_running(runs):
    return lock_held(Path(runs) / 'state' / 'scheduler.lock')


def start_checks(args, s, jobs, cpu_phase, gpus):
    problems = []
    legacy = legacy_scripts()
    if legacy:
        problems.append('old phase scripts are still running; let them finish (or stop them) first:\n  ' + '\n  '.join(legacy))
    diff = check_config(s, args.accept_config_change)
    if diff:
        problems.append('settings differ from the ones this run started with (runs/state/config.json):\n  ' + '\n  '.join(diff)
                        + '\n  restore them, or pass --accept-config-change if you really want to mix settings')
    probe = Scheduler(jobs, s['RUNS'], ROOT, gpus=[], log=lambda text: None)
    missing = [n for n in cpu_phase if not probe.is_done(n)]
    if gpus and missing:
        message = ('the CPU-only data phase is not finished (' + ', '.join(missing) + '); it runs alongside the GPU jobs that do '
                   'not need it, but the GPU may idle until the first data is ready. (Cheaper next time: run go.sh once in AutoDL '
                   'no-GPU mode first.)')
        if env('REQUIRE_CPU_PHASE', '0') == '1' and not args.allow_cpu_phase_on_gpu:
            problems.append(message)
        else:
            print('note: ' + message)
    return problems


# --------------------------------------------------------------------------- commands
def make_scheduler(args, s, quiet=False, gpus=None):
    info = query_gpus()
    signature = f'{len(info)}x' + ','.join(sorted({g["name"] for g in info}))
    jobs, cpu_phase = build_jobs(s, signature, info)
    if getattr(args, 'only', None):  # the matching jobs and everything they need; the run is complete when those are
        pattern = re.compile(args.only)
        by_name = {j.name: j for j in jobs}
        keep, todo = set(), [j.name for j in jobs if pattern.search(j.name)]
        if not todo:
            raise SystemExit(f'--only {args.only!r} matches no job (see `go.sh list`)')
        while todo:
            name = todo.pop()
            if name not in keep:
                keep.add(name)
                todo += list(by_name[name].deps)
        jobs = [j for j in jobs if j.name in keep]
        for job in jobs:
            job.soft = tuple((d, f) for d, f in job.soft if d in keep)
        cpu_phase = [n for n in cpu_phase if n in keep]
    # Several machines, one pipeline (HANDOFF.md section 4): the same regex is EXTERNAL_JOBS on the main machine and
    # LOCAL_JOBS on a helper.  External jobs are never started here; their dependents wait until `pipeline.py sync`
    # brings their outputs and .done marker.  The CPU data phase and the preflight always run on every machine.
    external, local_only = env('EXTERNAL_JOBS', ''), env('LOCAL_JOBS', '')
    for job in jobs:
        shared = job.name not in cpu_phase and job.name not in ('preflight', 'download_models')
        if shared and ((external and re.search(external, job.name)) or (local_only and not re.search(local_only, job.name))):
            job.ready, job.waiting_for = (lambda: False), 'runs on another machine'
    orch.MILESTONES.update(MILESTONES)
    scheduler = Scheduler(jobs, s['RUNS'], ROOT, gpus=gpus, cpu_slots=int(env('CPU_SLOTS', 0)) or (min(8, max(4, (os.cpu_count() or 8) // 2)) if info else 1),
                          notifier=Notifier(os.environ.get('NOTIFY_URL'), log=print), gpu_free_mib=float(env('GPU_FREE_MIB', 2000)),
                          code_version=code_version(), retry_failed=getattr(args, 'retry_failed', False),
                          max_hours=float(env('MAX_HOURS', 0)) or None, log=None if not quiet else (lambda text: None))
    return scheduler, jobs, cpu_phase, info


def cmd_run(args, s):
    scheduler, jobs, cpu_phase, info = make_scheduler(args, s)
    if scheduler_running(s['RUNS']):
        print('scheduler already running; showing status instead')
        return cmd_status(argparse.Namespace(brief=True, json=False), s)
    problems = start_checks(args, s, jobs, cpu_phase, info)
    if problems:
        print('NOT STARTED:\n- ' + '\n- '.join(problems))
        return 1
    if args.check_only:
        print('ok to start')
        return 0
    outcome = scheduler.run(on_finish=lambda result: maybe_shutdown(result, s, bool(info), scheduler.notifier))
    return 0 if outcome in ('complete', 'stopped', 'drained', 'already-running') else 2


def maybe_shutdown(outcome, s, have_gpus, notifier):
    mode = env('AUTO_SHUTDOWN', 'off')
    if not have_gpus or not ((mode == 'done' and outcome == 'complete') or (mode == 'idle' and outcome in ('complete', 'stalled'))):
        return
    grace = float(env('SHUTDOWN_GRACE_MIN', 10))
    cancel = Path(s['RUNS']) / 'state' / 'cancel_shutdown'
    cancel.unlink(missing_ok=True)
    notifier.send(f'shutting down in {grace:.0f} min ({outcome})', f'touch {cancel} to cancel')
    print(f'AUTO_SHUTDOWN={mode}: shutting the instance down in {grace:.0f} min (touch {cancel} to cancel)', flush=True)
    deadline = time.time() + grace * 60
    while time.time() < deadline:
        if cancel.exists():
            print('shutdown cancelled', flush=True)
            return
        time.sleep(10)
    command = ['/usr/bin/shutdown'] if Path('/usr/bin/shutdown').exists() else ['shutdown', '-h', 'now']
    subprocess.run(command)


def fmt_time(t):
    return time.strftime('%m-%d %H:%M', time.localtime(t)) if t else '-'


def fmt_hours(h):
    return f'{h:.1f} h' if h < 48 else f'{h / 24:.1f} d'


def cmd_status(args, s):
    scheduler, jobs, cpu_phase, info = make_scheduler(args, s, quiet=True)
    running = scheduler_running(s['RUNS'])
    saved = read_json(Path(s['RUNS']) / 'state' / 'status.json')
    if running and saved and time.time() - saved.get('updated', 0) < 120:
        status = saved  # the live scheduler's view (it knows progress rates)
    else:
        scheduler.refresh_gpus()
        scheduler.adopt()
        status = scheduler.status()
    if args.json:
        print(json.dumps(status, indent=1, ensure_ascii=False))
        return 0
    rows = status['jobs']
    gpu_line = ', '.join(f"{g.get('name', 'GPU')}" for g in status.get('gpus', [])) or 'none (no-GPU mode)'
    print(f"ReDAG pipeline  {time.strftime('%F %T')}  scheduler: {'running' if running else 'NOT running'}  GPUs: {gpu_line}")
    active = [(n, r) for n, r in rows.items() if r['state'] == 'running']
    for name, r in sorted(active, key=lambda x: x[1].get('gpu_ids') or [99]):
        prog = r.get('progress') or {}
        bar = f"{prog.get('done')}/{prog.get('total')} {prog.get('unit', '')}" if prog.get('total') else ''
        gpus = ','.join(map(str, r.get('gpu_ids') or [])) or 'cpu'
        print(f"  [{gpus:>4}] {name:<28} {max(r.get('elapsed_hours', 0), 0):5.1f} h done, ~{fmt_hours(r.get('left_hours', 0))} left "
              f"(ETA {fmt_time(r.get('eta'))})  {bar}")
    groups = {}
    for name, r in rows.items():
        groups.setdefault(r['group'], {}).setdefault(r['state'], []).append(name)
    print(f"  {'phase':<16}{'done':>8}{'running':>9}{'waiting':>9}{'failed':>8}")
    for group in sorted(groups):
        c = groups[group]
        total = sum(len(v) for v in c.values())
        waiting = sum(len(c.get(k, [])) for k in ('waiting', 'ready', 'held'))
        failed = len(c.get('failed', [])) + len(c.get('blocked', []))
        print(f"  {group:<16}{len(c.get('done', [])):>4}/{total:<3}{len(c.get('running', [])):>9}{waiting:>9}{failed:>8}")
    failed = [(n, r) for n, r in rows.items() if r['state'] == 'failed']
    for name, r in failed:
        print(f"  FAILED {name}: {(r.get('failed') or {}).get('reason')}  -> fix, then `go.sh retry`; log: runs/logs/{name}.log")
    held = [(n, r) for n, r in rows.items() if r['state'] == 'held' and r.get('waiting_for')]
    if held and not args.brief:
        reasons = {}
        for name, r in held:
            reasons.setdefault(r['waiting_for'], []).append(name)
        for reason, names in reasons.items():
            print(f"  waiting ({reason}): {', '.join(names[:6])}{' ...' if len(names) > 6 else ''}")
    if status.get('foreign_busy_gpus'):
        print(f"  GPUs used by other processes (not scheduled): {status['foreign_busy_gpus']}")
    finish = status.get('projected_finish')
    if finish:
        print(f"  projected finish: {fmt_time(finish)} (in {fmt_hours((finish - time.time()) / HOUR)}), "
              f"~{status.get('gpu_hours_left')} GPU-hours of work left")
    if not running and any(r['state'] not in ('done',) for r in rows.values()):
        print('  start/resume with: bash scripts/launch/go.sh')
    return 0


def cmd_plan(args, s):
    scheduler, jobs, cpu_phase, info = make_scheduler(args, s, quiet=True)
    scheduler.refresh_gpus()
    scheduler.adopt()
    now = time.time()
    states = {j.name: 'waiting' for j in jobs} if args.from_scratch else scheduler.classify(now)
    est = scheduler.estimates(now)
    counts = args.gpus or [len(info) or 2]
    print(f"remaining work: {sum(est[n] * scheduler.jobs[n].gpus for n, st in states.items() if st not in ('done',)) / HOUR:.0f} GPU-hours "
          f"(A800 estimates, refined from finished jobs)")
    for n in counts:
        done = [] if args.from_scratch else [k for k, v in states.items() if v == 'done']
        running = [] if args.from_scratch else [(k, scheduler.remaining(k, now, est[k]), scheduler.jobs[k].gpus) for k in scheduler.running]
        timeline = simulate(jobs, est, n, done=done, running=running, now=0.0)
        if not timeline:
            print(f'{n} GPU(s): nothing left to run')
            continue
        end = max(e for _, e in timeline.values())
        skipped = [j.name for j in jobs if j.name not in timeline and j.name not in done]
        busy = sum((e - b) * scheduler.jobs[k].gpus for k, (b, e) in timeline.items()) / max(end * n, 1)
        print(f"{n} GPU(s): finish in {fmt_hours(end / HOUR)} (utilisation {busy:.0%})" + (f"; cannot run: {', '.join(skipped[:5])}" if skipped else ''))
        if args.verbose or len(counts) == 1:
            ends = {}
            for k, (b, e) in timeline.items():
                g = scheduler.jobs[k].group
                ends[g] = max(ends.get(g, 0), e)
            for g in sorted(ends):
                print(f'    {g:<16} done after {fmt_hours(ends[g] / HOUR)}')
    return 0


def cmd_stop(args, s):
    state = Path(s['RUNS']) / 'state'
    state.mkdir(parents=True, exist_ok=True)
    if args.scheduler_only:  # e.g. to load new job definitions: jobs keep running and the next scheduler adopts them
        pid = int((state / 'scheduler.pid').read_text().strip()) if (state / 'scheduler.pid').exists() else 0
        if pid and scheduler_running(s['RUNS']):
            os.kill(pid, 15)
            for _ in range(50):
                if not scheduler_running(s['RUNS']):
                    break
                time.sleep(0.2)
            print(f'scheduler {pid} stopped; running jobs continue')
        else:
            print('no scheduler running')
        return 0
    if scheduler_running(s['RUNS']):
        orch.atomic_write(state / 'control', 'drain\n' if args.drain else 'stop\n')
        print('drain requested: running jobs finish, nothing new starts' if args.drain else
              'stop requested: running jobs are terminated and resume from their checkpoints next time')
        return 0
    # no scheduler: stop orphaned jobs that still hold their locks
    stopped = 0
    for spec_path in (state / 'jobs').glob('*.json'):
        name = spec_path.stem
        if lock_held(state / 'jobs' / f'{name}.lock') and not args.drain:
            pgid = (read_json(spec_path) or {}).get('pgid')
            if pgid:
                try:
                    os.killpg(pgid, 15)
                    stopped += 1
                    print('terminated', name)
                except ProcessLookupError:
                    pass
    print('no scheduler running' + (f'; terminated {stopped} job(s)' if stopped else ''))
    return 0


def cmd_list(args, s):
    jobs, _ = build_jobs(s)
    for job in jobs:
        print(f'{job.group:<14} {job.name:<30} gpus={job.gpus} est={job.hours:5.2f} h  deps={",".join(job.deps)}')
    return 0


def cmd_logs(args, s):
    path = Path(s['RUNS']) / 'logs' / f'{args.job}.log'
    if not path.exists():
        print('no log yet:', path)
        return 1
    if args.follow:
        os.execvp('tail', ['tail', '-n', str(args.n), '-f', str(path)])
    print(orch.tail_lines(path, args.n))
    return 0


def cmd_reset(args, s):
    jobs, _ = build_jobs(s)
    names = {j.name for j in jobs}
    unknown = [n for n in args.jobs if n not in names]
    if unknown:
        print('unknown jobs:', unknown)
        return 1
    if scheduler_running(s['RUNS']):
        print('stop the scheduler first (go.sh stop)')
        return 1
    targets = set(args.jobs)
    if args.dependents:
        changed = True
        while changed:
            changed = False
            for job in jobs:
                if job.name not in targets and (set(job.deps) & targets or {d for d, _ in job.soft} & targets):
                    targets.add(job.name)
                    changed = True
    runs = Path(s['RUNS'])
    for name in sorted(targets):
        for path in (runs / 'logs' / f'{name}.done', runs / 'state' / 'jobs' / f'{name}.failed'):
            path.unlink(missing_ok=True)
        print('reset', name)
    print('Markers removed; outputs are kept, so resumable jobs skip work already done. Delete a job\'s output folder '
          'as well for a clean rerun.')
    return 0


# --------------------------------------------------------------------------- bundles (hand-over / backup)
def referenced_files(final):
    """Files outside data/final that the final JSONL files point to (images of sessions, states, SFT pairs)."""
    found = set()

    def walk(value, base):
        if isinstance(value, str) and len(value) < 400 and value.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
            path = (base / value).resolve() if not os.path.isabs(value) else Path(value)
            if path.is_file():
                found.add(path)
        elif isinstance(value, dict):
            for item in value.values():
                walk(item, base)
        elif isinstance(value, list):
            for item in value:
                walk(item, base)
    for path in Path(final).glob('*.jsonl'):
        for row in orch.read_jsonl(path):
            walk(row, path.parent)
    return found


def bundle_members(what, s, with_adapters=False):
    runs, data = Path(s['RUNS']), ROOT / 'data'
    files = set()

    def tree(folder, pattern='**/*'):
        folder = Path(folder)
        if folder.exists():
            files.update(p for p in folder.glob(pattern) if p.is_file() and '.tmp' not in p.name)
    if what == 'data':
        tree(data / 'final')
        tree(data / 'processed' / 'graphs')
        tree(data / 'processed' / 'sessions')
        tree(data / 'processed' / 'canvases')
        tree(data / 'raw' / 'benchmarks')
        files.update(referenced_files(data / 'final'))
        jobs, cpu_phase = build_jobs(s)
        data_jobs = set(cpu_phase) | {j.name for j in jobs if j.group == '3 data (GPU)'}
        files.update(p for p in (runs / 'logs').glob('*.done') if p.stem in data_jobs)
    elif what == 'results':
        tree(runs / 'eval', '*.json')
        tree(runs / 'eval', '*_images/blind_*')
        for pattern in ('**/metrics.jsonl', '**/training_config.json', '**/runtime.json', '**/stage_checkpoints.json',
                        '**/training_audits.jsonl', '**/sft_complete.json', '**/rl_complete.json'):
            files.update(p for p in runs.glob(pattern) if p.is_file() and 'smoke' not in p.parts and 'wiring' not in p.parts)
        for name in ('logs/timings.jsonl', 'state/config.json', 'state/status.json', 'RESULTS.md'):
            if (runs / name).exists():
                files.add(runs / name)
        if with_adapters:
            for adapter in list(runs.glob('sft/*/adapter_*')) + list(runs.glob('sft/*/pytorch_lora_weights.safetensors')) + \
                    list(runs.glob('rl/**/controller/*')) + list(runs.glob('rl/**/editor/*')):
                if adapter.is_file() and 'resume' not in adapter.parts:
                    files.add(adapter)
            files.update(p for p in runs.glob('sft/*/*.json') if p.is_file())
    else:
        raise SystemExit('bundle kinds: data, results')
    return sorted(files)


def cmd_bundle(args, s):
    import tarfile
    if args.action == 'export':
        members = bundle_members(args.what, s, args.with_adapters)
        out_dir = Path(args.out or (ROOT.parent / 'bundles'))
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = f"redag_{args.what}_{time.strftime('%Y%m%d_%H%M')}"
        total = sum(p.stat().st_size for p in members)
        print(f'{len(members)} files, {total / 2**30:.2f} GiB -> {out_dir}/{stem}.tar' + (f' in {args.split} MiB parts' if args.split else ''))
        writer = PartWriter(out_dir / f'{stem}.tar', args.split * 2**20 if args.split else None)
        with tarfile.open(fileobj=writer, mode='w|', format=tarfile.PAX_FORMAT) as tar:
            for path in members:
                tar.add(str(path), arcname=str(path.relative_to(ROOT)), recursive=False)
        writer.close()
        manifest = dict(kind=args.what, created=time.strftime('%F %T'), files=len(members), bytes=total, code=code_version(),
                        parts=[dict(name=p.name, bytes=p.stat().st_size, sha256=h) for p, h in writer.parts],
                        settings={k: s[k] for k in RESULT_KEYS})
        (out_dir / f'{stem}.manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
        print(f'wrote {len(writer.parts)} part(s) and {stem}.manifest.json; import on the other machine with:\n'
              f'  bash scripts/launch/go.sh bundle import {out_dir}/{stem}.manifest.json')
        return 0
    manifest_path = Path(args.files[0])
    manifest = json.loads(manifest_path.read_text())
    parts = [manifest_path.parent / p['name'] for p in manifest['parts']]
    for part, info in zip(parts, manifest['parts']):
        if not part.exists() or part.stat().st_size != info['bytes']:
            raise SystemExit(f'missing or incomplete part: {part}')
        if not args.no_verify and sha256_file(part) != info['sha256']:
            raise SystemExit(f'checksum mismatch: {part}')
    reader = PartReader(parts)
    with tarfile.open(fileobj=reader, mode='r|') as tar:
        for member in tar:
            target = (ROOT / member.name).resolve()
            if not str(target).startswith(str(ROOT.resolve()) + os.sep) or member.issym() or member.islnk():
                raise SystemExit(f'refusing unsafe path in bundle: {member.name}')
            tar.extract(member, path=str(ROOT), filter='data') if hasattr(tarfile, 'data_filter') else tar.extract(member, path=str(ROOT))
    print(f"imported {manifest['files']} files ({manifest['bytes'] / 2**30:.2f} GiB) of a {manifest['kind']} bundle")
    if manifest['kind'] == 'data':
        print('checking data:')
        subprocess.run([s['PY'], 'scripts/data/check_data.py'], cwd=ROOT)
    return 0


SYNC_INCLUDE = ('data/final/***', 'data/processed/graphs/***', 'data/processed/canvases/***',
                'runs/sft/***', 'runs/rl/***', 'runs/eval/***')
SYNC_EXCLUDE = ('resume/', 'resume.tmp/', 'resume.old/', '*.tmp', '*.incomplete', '__pycache__/')


def marker_names(s):
    """GPU-phase .done markers are shared; CPU-phase ones and the preflight stay per machine (each machine runs its own)."""
    jobs, cpu_phase = build_jobs(s)
    return sorted(j.name for j in jobs if j.name not in cpu_phase and j.name != 'preflight')


def rsync(src, dst, ssh, filters, dry=False):
    cmd = ['rsync', '-a', '--update', '--partial', '--timeout=120', '-e', ssh]
    cmd += [f'--exclude={x}' for x in SYNC_EXCLUDE] + filters + [src, dst]
    if dry:
        cmd.insert(1, '-n')
    return subprocess.run(cmd, capture_output=True, text=True)


def cmd_sync(args, s):
    """Exchange outputs with another machine that runs other jobs of the same pipeline (see EXTERNAL_JOBS).

    Both directions, newest file wins (--update).  Outputs go first and the .done markers / stage files last, so a
    machine never sees a finished marker before the files behind it."""
    host, _, port = args.peer.partition(':')
    ssh = f'ssh -p {port or 22} -i {args.key} -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 -o ServerAliveInterval=30'
    remote = f'{host}:{args.remote_root.rstrip("/")}/'
    local = str(ROOT) + '/'
    markers = [f'--include=runs/logs/{n}.done' for n in marker_names(s)] + [f'--include=runs/logs/{n}.log' for n in marker_names(s)]
    dirs = ['--include=*/', '--prune-empty-dirs']
    outputs = [*dirs, '--exclude=runs/logs/**', '--exclude=runs/rl/**/stage_checkpoints.json',
               *[f'--include={p}' for p in SYNC_INCLUDE], '--exclude=*']
    finals = [*dirs, *markers, '--include=runs/rl/**/stage_checkpoints.json', '--exclude=*']
    while True:
        started = time.time()
        for label, src, dst in (('pull', remote, local), ('push', local, remote)):
            for kind, filters in (('outputs', outputs), ('markers', finals)):
                result = rsync(src, dst, ssh, filters, args.dry_run)
                if result.returncode not in (0, 24):  # 24: files vanished during transfer (normal while jobs write)
                    print(f'{time.strftime("%T")} {label} {kind} failed ({result.returncode}): {result.stderr.strip()[-300:]}', flush=True)
        print(f'{time.strftime("%T")} sync with {args.peer} done in {time.time() - started:.0f}s', flush=True)
        if not args.loop:
            return 0
        time.sleep(args.loop)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(16 * 2**20), b''):
            digest.update(block)
    return digest.hexdigest()


class PartWriter:
    """File-like sink that splits a stream into numbered parts (netdisk size limits) and hashes each part."""

    def __init__(self, path, limit=None):
        self.path, self.limit, self.parts, self.stream, self.index = Path(path), limit, [], None, 0
        self.open_next()

    def open_next(self):
        if self.stream:
            self.finish()
        name = self.path if self.limit is None else self.path.with_name(f'{self.path.name}.part{self.index:03d}')
        self.stream, self.size, self.hash, self.name = open(name, 'wb'), 0, hashlib.sha256(), name
        self.index += 1

    def finish(self):
        self.stream.close()
        self.parts.append((self.name, self.hash.hexdigest()))
        self.stream = None

    def write(self, data):
        view = memoryview(data)
        while view:
            room = len(view) if self.limit is None else self.limit - self.size
            if room <= 0:
                self.open_next()
                continue
            chunk = view[:room]
            self.stream.write(chunk)
            self.hash.update(chunk)
            self.size += len(chunk)
            view = view[room:]
        return len(data)

    def close(self):
        if self.stream:
            self.finish()

    def flush(self):
        pass


class PartReader:
    def __init__(self, parts):
        self.parts, self.stream = list(parts), None

    def read(self, size=-1):
        out = b''
        while size < 0 or len(out) < size:
            if self.stream is None:
                if not self.parts:
                    break
                self.stream = open(self.parts.pop(0), 'rb')
            chunk = self.stream.read(-1 if size < 0 else size - len(out))
            if not chunk:
                self.stream.close()
                self.stream = None
                continue
            out += chunk
        return out


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0].startswith('_'):
        return helper(argv)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    run = sub.add_parser('run', help='start or resume the scheduler in the foreground (go.sh runs it in the background)')
    run.add_argument('--retry-failed', action='store_true', help='retry jobs that failed with the current code')
    run.add_argument('--only', help='regular expression: schedule only matching jobs')
    run.add_argument('--allow-cpu-phase-on-gpu', action='store_true')
    run.add_argument('--accept-config-change', action='store_true')
    run.add_argument('--check-only', action='store_true', help='only run the start checks')
    status = sub.add_parser('status')
    status.add_argument('--brief', action='store_true')
    status.add_argument('--json', action='store_true')
    stop = sub.add_parser('stop')
    stop.add_argument('--drain', action='store_true', help='let running jobs finish, start nothing new')
    stop.add_argument('--scheduler-only', action='store_true', help='stop only the scheduler; jobs keep running')
    plan = sub.add_parser('plan', help='projected timeline (A800 estimates refined by finished jobs)')
    plan.add_argument('--gpus', type=int, nargs='+')
    plan.add_argument('--from-scratch', action='store_true')
    plan.add_argument('--verbose', action='store_true')
    sub.add_parser('list')
    logs = sub.add_parser('logs')
    logs.add_argument('job')
    logs.add_argument('-n', type=int, default=40)
    logs.add_argument('-f', '--follow', action='store_true')
    reset = sub.add_parser('reset', help='forget that jobs finished (outputs are kept)')
    reset.add_argument('jobs', nargs='+')
    reset.add_argument('--dependents', action='store_true')
    bundle = sub.add_parser('bundle', help='export / import data or results for hand-over and backup')
    bundle.add_argument('action', choices=('export', 'import'))
    bundle.add_argument('what', nargs='?', choices=('data', 'results'))
    bundle.add_argument('files', nargs='*')
    bundle.add_argument('--out')
    bundle.add_argument('--with-adapters', action='store_true')
    bundle.add_argument('--split', type=int, default=0, help='part size in MiB (e.g. 3900 for netdisk limits)')
    bundle.add_argument('--no-verify', action='store_true')
    sync = sub.add_parser('sync', help='exchange outputs with another machine running other jobs (EXTERNAL_JOBS)')
    sync.add_argument('--peer', required=True, help='root@host:port of the other machine (key-based ssh)')
    sync.add_argument('--key', default=os.path.expanduser('~/.ssh/redag_sync'))
    sync.add_argument('--remote-root', default=str(ROOT))
    sync.add_argument('--loop', type=int, default=0, help='repeat every N seconds')
    sync.add_argument('--dry-run', action='store_true')
    sub.add_parser('is-running', help='exit status 0 if a scheduler is running')
    notify = sub.add_parser('notify-test', help='send a test message to NOTIFY_URL')
    notify.add_argument('text', nargs='?', default='test message')
    args = parser.parse_args(argv)
    s = settings()
    if args.command == 'bundle':
        if args.action == 'export' and not args.what:
            parser.error('bundle export needs data or results')
        if args.action == 'import':
            args.files = ([args.what] if args.what else []) + args.files
            if not args.files:
                parser.error('bundle import needs the manifest file')
        return cmd_bundle(args, s)
    if args.command == 'is-running':
        return 0 if scheduler_running(s['RUNS']) else 1
    if args.command == 'notify-test':
        ok = Notifier(os.environ.get('NOTIFY_URL'), log=print).send('notification test', args.text)
        print('sent' if ok else 'not sent (is NOTIFY_URL set?)')
        return 0 if ok else 1
    return {'run': cmd_run, 'status': cmd_status, 'stop': cmd_stop, 'plan': cmd_plan, 'list': cmd_list,
            'logs': cmd_logs, 'reset': cmd_reset, 'sync': cmd_sync}[args.command](args, s)


if __name__ == '__main__':
    sys.exit(main() or 0)
