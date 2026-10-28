import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

from agentic_image.runtime.fixed_evaluation import SEEDS, atomic_json, read_json, registered_tasks
from agentic_image.runtime.controller_evaluation import (
    DEFAULT_SELECTION, METHODS, evaluated_policy, freeze_json, load_evaluated_policy,
    policy_identity, sampler_protocol,
)


def aggregate(tasks, output):
    report={'scope':'actual_controller_episodes','fixed_families':64,'image_seeds':list(SEEDS),'required_episodes_per_configuration':320,'configurations':{},'cross_configuration_complete':False}
    for method in METHODS:
        results=[];missing=[]
        for index,task in enumerate(tasks):
            for seed in SEEDS:
                path=Path(output)/method/'episodes'/f'{index:03d}'/f'seed{seed}'/'episode.json'
                if not path.is_file():missing.append({'index':index,'seed':seed});continue
                row=read_json(path)
                if row['task_id']!=task.task_id or row['family_id']!=task.family_id or row['seed']!=seed or row['method']!=method:
                    raise ValueError('Episode identity differs from the frozen development set')
                results.append(row)
        report['configurations'][method]={'observed_episodes':len(results),'required_episodes':320,'missing':missing,
            'success_count':sum(r['terminal']['success'] for r in results),
            'success_rate':sum(r['terminal']['success'] for r in results)/320 if len(results)==320 else None,
            'protocol_failures':sum(r['protocol_failure'] for r in results),
            'no_delivered_image':sum(not r['delivered_image_id'] for r in results),
            'episodes_with_unknown_atoms':sum(any(not a['known'] for a in r['terminal']['atoms']) for r in results),
            'logical_actions':sum(r['budget']['actions'] for r in results),'logical_renders':sum(r['budget']['renders'] for r in results),
            'cost_ledger_complete_episodes':sum(r['costs']['historical_measurements_complete'] for r in results),
            'observed_gpu_seconds_lower_bound':sum(r['costs']['observed_gpu_seconds'] for r in results)}
    common=Path(output)/'common_protocol.json'
    report['common_protocol']=read_json(common) if common.is_file() else None
    report['evaluated_policies']={method:read_json(Path(output)/method/'registration.json')['evaluated_policy']
        for method in METHODS if (Path(output)/method/'registration.json').is_file()}
    report['cross_configuration_complete']=common.is_file() and len(report['evaluated_policies'])==4 and all(v['observed_episodes']==320 for v in report['configurations'].values())
    report['status']='completed' if report['cross_configuration_complete'] else 'incomplete'
    atomic_json(Path(output)/'aggregate.json',report)
    return report


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['run','aggregate'])
    parser.add_argument('--method',choices=METHODS)
    parser.add_argument('--config')
    parser.add_argument('--sampler-selection',default=DEFAULT_SELECTION)
    parser.add_argument('--sft-completion',default='data/actor-full-sft/training_completed.json')
    parser.add_argument('--checkpoint')
    parser.add_argument('--tasks',default='data/evaluation/development64/public_tasks.jsonl')
    parser.add_argument('--reservation',default='data/evaluation/development64/reserved.frozen.jsonl')
    parser.add_argument('--private-tasks',default='data/private_eval_v1/development64/private_tasks.jsonl')
    parser.add_argument('--output',default='runs/development64-controller-episodes')
    parser.add_argument('--start',type=int,default=0)
    parser.add_argument('--count',type=int,default=8)
    parser.add_argument('--seed',type=int,choices=SEEDS)
    parser.add_argument('--device',choices=['cuda:0','cuda:1'],default='cuda:0')
    parser.add_argument('--actor-device',choices=['cuda:0','cuda:1'])
    parser.add_argument('--renderer-device',choices=['cuda:0','cuda:1'])
    args=parser.parse_args()
    tasks=registered_tasks(args.tasks,args.reservation)
    if args.stage=='aggregate':print(json.dumps(aggregate(tasks,args.output)),flush=True);return
    if not args.config or not args.method or not args.sampler_selection:parser.error('Run requires explicit configuration, method and accepted sampler proof')
    if args.count!=8 or args.start not in range(0,64,8):raise ValueError('Use an original eight-task shard')
    from agentic_image.config import load_config
    config=load_config(args.config)
    sampler=sampler_protocol(args.sampler_selection,tasks)
    config['renderer'].update(sampler['parameters'])
    policy=evaluated_policy(config,args.method,sampler,args.sft_completion,args.checkpoint)
    calibration_path=config['judge']['calibration']
    if not calibration_path or not Path(calibration_path).is_file():raise ValueError('Actual frozen judge calibration is required')
    common={'scope':'actual_controller_episodes','models':{k:v for k,v in config['models'].items() if not k.endswith(('_policy','_reference'))},
        'actor_sampling':config['actor'],'sampler_protocol':sampler,'renderer':config['renderer'],'canvas':config['canvas'],'episode':config['episode'],
        'tasks':[t.model_dump(mode='json') for t in tasks],'private_evaluations':[json.loads(l) for l in Path(args.private_tasks).read_text().splitlines() if l.strip()],
        'calibration':read_json(calibration_path),'calibration_identity':read_json(calibration_path+'.model.json'),'image_seeds':list(SEEDS),
        'public_check_extractor_versions':sorted({t.check_spec.extractor_version for t in tasks}),
        'public_check_limitation':'Frozen public tasks use original whole-request fallback checks; not a validated decomposition of all public atoms.',
        'private_target_used_for_actor_actions':False,'training_updates_during_evaluation':0,'human_reviewed':False}
    freeze_json(Path(args.output)/'common_protocol.json',common)
    registration={'scope':'actual_controller_episodes','method':args.method,'models':config['models'],
        'qualified_on':sampler['qualified_on'],'evaluated_policy':policy,'common_protocol':common}
    root=Path(args.output)/args.method
    freeze_json(root/'registration.json',registration)
    seeds=(args.seed,) if args.seed is not None else SEEDS
    pending=[(i,s) for i in range(args.start,args.start+args.count) for s in seeds if not (root/'episodes'/f'{i:03d}'/f'seed{s}'/'episode.json').is_file()]
    if not pending:print(json.dumps({'status':'completed','models_loaded':False,'reused_episodes':args.count*len(seeds)}),flush=True);return
    import torch
    from agentic_image.runtime.components import Components
    from agentic_image.runtime.rollout import RolloutCollector
    from agentic_image.runtime.benchmarks import BenchmarkCosts
    from agentic_image.learning.sampling import stream_seed
    worker=root/'workers'/f'{args.start:03d}_seed{args.seed or "all"}'
    config['run_dir']=str(worker)
    config['data']['private_tasks']=args.private_tasks
    config['models']['actor_reference']=None;config['models']['renderer_reference']=None
    for service in config['devices']:config['devices'][service]=args.device
    if args.actor_device:config['devices']['actor']=args.actor_device
    if args.renderer_device:config['devices']['renderer']=args.renderer_device
    components=Components(config);components.load_all(private=True)
    loaded=load_evaluated_policy(components,policy)
    atomic_json(worker/'policy_load.json',{
        'evaluated_policy':policy,'qualified_on':sampler['qualified_on'],'loaded_state_verification':loaded,
        'devices':config['devices']})
    components.judge.require_calibration=True;components.evaluator.unconfirmed_route_policy='unknown'
    behavior=f'development64-actual:{args.method}:{root.resolve()}'
    components.render_bridge.behavior_version=behavior
    collector=RolloutCollector(components)
    for index,seed in pending:
        task=tasks[index];folder=root/'episodes'/f'{index:03d}'/f'seed{seed}';folder.mkdir(parents=True,exist_ok=True)
        collector.admit(task)
        environment=collector.restore_progress(folder,behavior)
        if environment is None:environment=collector.environment(task,f'development64:{args.method}:{index}:{seed}')
        if environment.task!=task:raise ValueError('Resumed controller task changed')
        while not environment.state.done:
            action_index=environment.state.budget.actions;cost_path=folder/f'cost_{action_index:02d}.json'
            previous=read_json(cost_path) if cost_path.is_file() else {'attempts':[],'historical_cost_gap':(folder/f'action_{action_index:02d}.pt').is_file()}
            measured=None;success=False
            try:
                with BenchmarkCosts(components) as measured:
                    collector.advance(environment,folder,stream_seed(seed,action_index),behavior)
                    success=True
            finally:
                if measured is not None:
                    previous['attempts'].append({'action_index':action_index,'completed_step':success,**measured.values})
                    atomic_json(cost_path,previous)
        action_paths=sorted(folder.glob('action_*.pt'))
        scoring_started=time.monotonic()
        episode=collector.finish(environment,folder,0,0,[str(p.resolve()) for p in action_paths])
        ledgers=[read_json(p) for p in sorted(folder.glob('cost_*.json'))]
        attempts=[a for ledger in ledgers for a in ledger['attempts']]
        costs={'historical_measurements_complete':len(ledgers)==environment.state.budget.actions and not any(l['historical_cost_gap'] for l in ledgers),
            'observed_gpu_seconds':sum(a.get('gpu_seconds',0.) for a in attempts),
            'observed_wall_seconds':sum(a.get('wall_seconds',0.) for a in attempts),
            'sampled_actions_observed':sum(a.get('sampled_actions',0) for a in attempts),
            'output_tokens_in_saved_actions':sum(torch.load(p,map_location='cpu',weights_only=True)['token_ids'].numel() for p in action_paths),
            'actual_dit_forwards_observed':sum(a.get('actual_dit_forwards',0) for a in attempts),
            'external_tool_server_gpu_time_included':False,'measurement_scope':'Per-action local services; private terminal scoring separately timed',
            'cost_ledger_paths':[str(p.resolve()) for p in sorted(folder.glob('cost_*.json'))]}
        row={'scope':'actual_controller_episode','method':args.method,'index':index,'task_id':task.task_id,'family_id':task.family_id,'seed':seed,
            'budget':environment.state.budget.model_dump(),'protocol_failure':environment.state.protocol_failure,
            'delivered_image_id':environment.state.delivered_image_id,'terminal':episode.terminal,'episode_record':asdict(episode),
            'costs':costs,'private_scoring_seconds':time.monotonic()-scoring_started,'public_check_extractor_version':task.check_spec.extractor_version,
            'private_target_used_for_actions':False,'training_updates':0,
            'policy_registration':str((root/'registration.json').resolve()),'sampler_parameters':sampler['parameters'],
            'loaded_state_verification':loaded}
        atomic_json(folder/'episode.json',row)
        print(json.dumps({'method':args.method,'index':index,'seed':seed,'success':episode.terminal['success'],'actions':environment.state.budget.actions,'renders':environment.state.budget.renders}),flush=True)
    atomic_json(root/'shards'/f'{args.start:03d}_{args.start+args.count-1:03d}_seed{args.seed or "all"}.completed.json',{'status':'completed','episodes':args.count*len(seeds),'scope':'actual_controller_episodes'})


if __name__=='__main__':main()
