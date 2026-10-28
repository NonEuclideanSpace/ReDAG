import argparse
import copy
import json
from pathlib import Path

from agentic_image.config import load_config
from agentic_image.core.artifacts import ArtifactStore
from agentic_image.runtime.fixed_evaluation import (
    SEEDS, SCOPE, aggregate_fixed_scores, atomic_json, fixed_inputs, read_json,
    registered_tasks, score_fixed_images,
)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['inspect','score','aggregate'])
    parser.add_argument('--config',default='configs/sampler-expanded-200.yaml')
    parser.add_argument('--configuration',choices=['baseline','sft200'],default='baseline')
    parser.add_argument('--tasks',default='data/evaluation/development64/public_tasks.jsonl')
    parser.add_argument('--reservation',default='data/evaluation/development64/reserved.frozen.jsonl')
    parser.add_argument('--private-tasks',default='data/private_eval_v1/development64/private_tasks.jsonl')
    parser.add_argument('--render-root')
    parser.add_argument('--baseline-root',default='runs/development64-baseline')
    parser.add_argument('--output',default='runs/development64-fixed-private')
    parser.add_argument('--start',type=int,default=0)
    parser.add_argument('--count',type=int,default=8)
    parser.add_argument('--device',choices=['cuda:0','cuda:1'],default='cuda:0')
    parser.add_argument('--calibration')
    parser.add_argument('--bootstrap-iterations',type=int,default=10000)
    args=parser.parse_args()
    tasks=registered_tasks(args.tasks,args.reservation)
    if args.stage=='aggregate':
        report=aggregate_fixed_scores(tasks,args.output,args.bootstrap_iterations)
        print(json.dumps(report),flush=True)
        return
    config=copy.deepcopy(load_config(args.config))
    render_root=args.render_root or f'runs/development64-{args.configuration}'
    store=ArtifactStore(config['data']['artifacts'])
    inputs,missing=fixed_inputs(tasks,render_root,args.start,args.count,args.configuration,store,
        config['models']['renderer_revision'],args.baseline_root)
    report={'scope':SCOPE,'configuration':args.configuration,'start':args.start,'count':args.count,
        'required_observations':args.count*len(SEEDS),'available_bound_images':len(inputs),'missing':missing,
        'status':'waiting_for_existing_renders' if missing else 'inputs_ready','rerenders_performed':0,
        'controller_episode_evaluation':False}
    name=f'{args.configuration}_{args.start:03d}_{args.start+args.count-1:03d}'
    atomic_json(Path(args.output)/'shards'/f'{name}.inputs.json',report)
    if args.stage=='inspect' or missing:
        print(json.dumps(report),flush=True)
        if args.stage=='score' and missing:raise SystemExit(75)
        return
    config['data']['private_tasks']=args.private_tasks
    config['judge']['calibration']=args.calibration or config['judge']['calibration']
    calibration_path=Path(config['judge']['calibration'])
    identity_path=Path(str(calibration_path)+'.model.json')
    if not calibration_path.is_file() or not identity_path.is_file():
        raise ValueError('Existing checkpoint-specific calibration files are required')
    calibration={'identity':read_json(identity_path),'routes':read_json(calibration_path)}
    expected_identity={'model_id':config['models']['judge'],'revision':config['models']['judge_revision'],'precision':'nf4-double-bf16'}
    if calibration['identity']!=expected_identity:raise ValueError('Judge identity differs from the calibration')
    evaluations={r['task_id']:r['evaluation'] for r in (json.loads(l) for l in Path(args.private_tasks).read_text().splitlines() if l.strip())}
    cached=[]
    for record in inputs:
        path=Path(args.output)/'scores'/args.configuration/f'{record["index"]:03d}_{record["eval_seed"]}.json'
        if not path.is_file():continue
        previous=read_json(path)
        fixed={'input':record,'private_evaluation':evaluations[record['task_id']],'calibration':calibration}
        if previous.get('binding')!=fixed:raise ValueError('A completed score changed its frozen inputs')
        if previous.get('status')=='scored':cached.append(previous)
    if len(cached)==len(inputs):
        results=cached;models_loaded=False
    else:
        from agentic_image.runtime.components import Components
        config['run_dir']=str(Path(args.output)/'workers'/name)
        for service in config['devices']:config['devices'][service]=args.device
        config['models']['actor_reference']=None
        config['models']['renderer_reference']=None
        components=Components(config)
        components.load_judge(private=True)
        components.record_memory('fixed_existing_image_private_judge_loaded')
        results=score_fixed_images(components,tasks,inputs,args.output)
        components.record_memory('fixed_existing_image_private_scoring_finished')
        models_loaded=True
    failures=[{'index':r['index'],'seed':r['eval_seed'],'error':r.get('error')} for r in results if r['status']!='scored']
    report.update(status='blocked' if failures else 'completed',scored=len(results)-len(failures),failures=failures,models_loaded=models_loaded)
    atomic_json(Path(args.output)/'shards'/f'{name}.completed.json',report)
    aggregate=aggregate_fixed_scores(tasks,args.output,args.bootstrap_iterations)
    report['aggregate_status']=aggregate['status']
    print(json.dumps(report),flush=True)
    if failures:raise SystemExit(2)


if __name__=='__main__':main()
