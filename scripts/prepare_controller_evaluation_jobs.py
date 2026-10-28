import argparse
import json
from pathlib import Path

from agentic_image.runtime.controller_evaluation import DEFAULT_SELECTION, METHODS
from agentic_image.runtime.fixed_evaluation import SEEDS


def jobs(python, output, priority=12):
    configs = {'base': 'configs/pilot-small.yaml', 'sft': 'data/actor-full-sft/rl_unfrozen.yaml',
        'agent_rl': 'runs/full-actor-rl/rl_config.yaml', 'joint_rl': 'runs/full-actor-rl/rl_config.yaml'}
    after = {'sft': 'full-actor-sft200', 'agent_rl': 'full-agent-rl32', 'joint_rl': 'full-joint-rl32'}
    result = []
    for method in METHODS:
        for start in range(0, 64, 8):
            for seed_index, seed in enumerate(SEEDS):
                device = (start // 8 + seed_index) % 2
                name = f'development64-controller-{method}-{start:03d}-seed{seed}'
                proof = 'data/actor-full-sft/training_completed.json' if method == 'sft' else f'runs/full-actor-rl/rl_{method.removesuffix("_rl")}/training_completed.json'
                required = [configs[method], DEFAULT_SELECTION, 'data/evaluation/development64/public_tasks.jsonl',
                    'data/private_eval_v1/development64/private_tasks.jsonl', 'data/calibration/thresholds.json',
                    'data/calibration/thresholds.json.model.json']
                gates = [{'path': DEFAULT_SELECTION, 'key': 'accepted', 'equals': True,
                    'reason': 'The fallback sampler must actually pass and be locked before official episodes'}]
                if method != 'base':
                    required.extend(['data/actor-full-sft/training_completed.json', proof])
                    gates.append({'path': proof, 'key': 'status', 'equals': 'completed'})
                if method in ('agent_rl', 'joint_rl'):
                    required.append(str(Path(proof).parent / 'metrics.jsonl'))
                    gates.append({'path': proof, 'key': 'effective_target_met', 'equals': True})
                row = {'id': name, 'title': f'{method}: 8 fixed development families, seed {seed}, actual controller episodes',
                    'gpus': [device], 'priority': priority, 'minimum_free_gib': 40,
                    'exclusive_group': f'renderer-evaluation-gpu{device}',
                    'requires': sorted(set(required)), 'gates': gates,
                    'pid_file': f'deployment/{name}.pid', 'exit_file': f'deployment/{name}.exit.json',
                    'started_file': f'deployment/{name}.started.json', 'log_file': f'deployment/{name}.log',
                    'command': [python, 'scripts/evaluate_development_controller.py', 'run', '--method', method,
                        '--config', configs[method], '--sampler-selection', DEFAULT_SELECTION, '--output', output,
                        '--start', str(start), '--count', '8', '--seed', str(seed), '--device', f'cuda:{device}'],
                    'env': {'PYTHONPATH': 'src', 'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '4'}}
                if method != 'base':
                    row['after'] = [after[method]]
                result.append(row)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--python', default='/usr/local/miniconda3/envs/py312/bin/python')
    parser.add_argument('--output', default='runs/development64-controller-episodes')
    parser.add_argument('--fragment', default='deployment/development64-controller-jobs.json')
    parser.add_argument('--priority', type=int, default=12)
    args = parser.parse_args()
    rows = jobs(args.python, args.output, args.priority)
    aggregate = {'id': 'development64-controller-aggregate', 'title': 'Aggregate all 1280 actual controller episodes',
        'gpus': [], 'priority': args.priority, 'after': [row['id'] for row in rows],
        'requires': [str(Path(args.output) / 'common_protocol.json')],
        'pid_file': 'deployment/development64-controller-aggregate.pid',
        'exit_file': 'deployment/development64-controller-aggregate.exit.json',
        'started_file': 'deployment/development64-controller-aggregate.started.json',
        'log_file': 'deployment/development64-controller-aggregate.log',
        'command': [args.python, 'scripts/evaluate_development_controller.py', 'aggregate', '--output', args.output],
        'env': {'PYTHONPATH': 'src'}}
    path = Path(args.fragment)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({'scope': 'four_configuration_actual_controller_episodes', 'jobs': rows + [aggregate],
        'episodes_per_job': 8, 'required_episodes': 1280, 'deployment_performed': False}, indent=2) + '\n')
    print(json.dumps({'fragment': str(path), 'episode_jobs': len(rows), 'aggregate_jobs': 1, 'episodes': 1280, 'deployed': False}))


if __name__ == '__main__':
    main()
