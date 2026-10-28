import json
import math
from pathlib import Path
import time

from agentic_image.core.types import PlanSpec, PublicTask


SEEDS = (11, 23, 47, 73, 101)
CONFIGURATIONS = {'baseline': 0, 'sft200': 200}
SCOPE = 'renderer_only_fixed_original_request_conditions'


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def registered_tasks(tasks_path, reservation_path):
    tasks = [PublicTask.model_validate(json.loads(l)) for l in Path(tasks_path).read_text().splitlines() if l.strip()]
    reserved = [json.loads(l) for l in Path(reservation_path).read_text().splitlines() if l.strip()]
    if len(tasks) != 64 or len(reserved) != 64 or len({task.family_id for task in tasks}) != 64:
        raise ValueError('The original 64 distinct development families are required')
    for index, (task, row) in enumerate(zip(tasks, reserved)):
        if (task.split_id != 'development' or row['split_id'] != 'development' or row['evaluation_index'] != index
                or task.task_id != 'development64:' + row['sample_id'] or task.family_id != row['family_id']
                or task.user_request.text != row['instruction']):
            raise ValueError(f'Development registration changed at {index}')
    return tasks


def fixed_inputs(tasks, render_root, start, count, configuration, artifacts, renderer_revision, baseline_root):
    if count != 8 or start not in range(0, 64, 8) or configuration not in CONFIGURATIONS:
        raise ValueError('Use one registered eight-task shard and baseline or sft200')
    rows, missing = [], []
    name = f'{start:03d}-{start+count-1:03d}'
    for seed in SEEDS:
        folder = Path(render_root)/name/f'seed{seed}'
        paths = [folder/'metadata.json', folder/'results.json', folder/'summary.json']
        if not all(path.is_file() for path in paths):
            missing.extend({'index':i,'seed':seed,'configuration':configuration,'reason':'render_metadata_pending'} for i in range(start,start+count))
            continue
        metadata, records, summary = map(read_json, paths)
        expected = {'steps':50,'eta':.3,'cfg_scale':4.,'seed':seed,'optimizer_updates':CONFIGURATIONS[configuration],
                    'renderer_revision':renderer_revision,'tasks':[t.task_id for t in tasks[start:start+count]]}
        if any(metadata.get(k) != v for k,v in expected.items()):
            raise ValueError(f'Render protocol changed: {folder}')
        cohort = metadata.get('development_cohort', {})
        if (cohort.get('fixed_families') != 64 or cohort.get('indices') != list(range(start,start+count))
                or cohort.get('registered_seeds') != list(SEEDS)):
            raise ValueError('Render metadata lost its original development family and seed registration')
        if summary.get('status') != 'completed' or summary.get('completed') != count or len(records) != count:
            missing.extend({'index':i,'seed':seed,'configuration':configuration,'reason':'render_shard_pending'} for i in range(start,start+count))
            continue
        baseline = None
        if configuration == 'sft200':
            prior = Path(baseline_root)/name/f'seed{seed}'
            if not (prior/'results.json').is_file() or not (folder/'policy_load.json').is_file():
                missing.extend({'index':i,'seed':seed,'configuration':configuration,'reason':'baseline_or_policy_receipt_pending'} for i in range(start,start+count))
                continue
            baseline = read_json(prior/'results.json')
            receipt = read_json(folder/'policy_load.json')
            if (len(baseline) != count or receipt.get('exact_tensor_equality') is not True
                    or receipt.get('training_updates') != 200 or receipt.get('nonzero_lora_b_elements',0) <= 0
                    or receipt.get('path') != metadata.get('policy_path')
                    or not metadata.get('fixed_condition_bundles') or metadata.get('evaluation_optimizer_updates') != 0):
                raise ValueError('Trained image lacks its actual 200-update policy and fixed-condition evidence')
        for local_index, record in enumerate(records):
            index=start+local_index;task=tasks[index]; result=record['result']
            plan=PlanSpec(global_text=task.user_request.text).model_dump(mode='json')
            if (record['index'] != local_index or record['task_id'] != task.task_id or record['family_id'] != task.family_id
                    or record['request'] != task.user_request.text or record['source_image_id'] != task.user_request.source_image_id
                    or record['plan'] != plan or result['metadata']['seed'] != seed
                    or result['metadata'].get('behavior_version') != ('initial' if configuration=='baseline' else 'renderer-global-ai:step200')):
                raise ValueError('A rendered image differs from the fixed public task or seed')
            recipe_path=folder/'bundles'/(result['bundle_id'].replace(':','_')+'.json')
            if not recipe_path.is_file() or not Path(record['image']).is_file():
                missing.append({'index':index,'seed':seed,'configuration':configuration,'reason':'render_pixels_or_recipe_missing'})
                continue
            recipe=read_json(recipe_path)
            if (recipe['bundle_id'] != result['bundle_id'] or recipe['plan'] != plan
                    or recipe['source_image_id'] != task.user_request.source_image_id
                    or recipe.get('model_versions',{}).get('renderer') != renderer_revision):
                raise ValueError('Rendered condition recipe changed')
            if baseline is not None:
                before=baseline[local_index]
                if (before['task_id'] != task.task_id or before['source_image_id'] != record['source_image_id']
                        or before['plan'] != plan or before['result']['bundle_id'] != result['bundle_id']
                        or before['result']['metadata']['seed'] != seed or record.get('source_condition_reused_exactly') is not True
                        or record.get('before_bundle_id') != result['bundle_id']):
                    raise ValueError('Trained and baseline images do not reuse exactly the same conditions')
            ref=artifacts.get_ref(result['image_id'])
            if (ref.metadata.get('bundle_id') != result['bundle_id'] or ref.metadata.get('seed') != seed
                    or tuple(ref.shape[:2]) != (512,512)):
                raise ValueError('Rendered artifact identity differs from its bundle, seed or canvas')
            rows.append({'scope':SCOPE,'method':configuration,'index':index,'training_seed':0,'checkpoint_replicates':1,
                'task_id':task.task_id,'family_id':task.family_id,'eval_seed':seed,
                'source_image_id':record['source_image_id'],'source_value_id':artifacts.get_ref(record['source_image_id']).value_id,
                'image_id':ref.artifact_id,'image_value_id':ref.value_id,'image':record['image'],
                'bundle_id':result['bundle_id'],'condition_recipe':recipe,'plan':plan,'render_metadata':metadata,
                'render_record':record,'actual_renderer_seconds':record['seconds'],
                'controller_episode_evaluated':False,'actual_controller_actions':0,'actual_renders':1})
    return rows, missing


def score_fixed_images(components, tasks, inputs, output):
    components.judge.require_calibration=True
    components.evaluator.unconfirmed_route_policy='unknown'
    calibration={'identity':components.judge.config.calibration_identity,'routes':components.judge.thresholds}
    if not calibration['routes'] or not any(v.get('confirmed') for v in calibration['routes'].values()):
        raise ValueError('Checkpoint-specific calibration is required')
    results=[]
    for record in inputs:
        task=tasks[record['index']];evaluation=components.evaluator.evaluations[task.task_id]
        path=Path(output)/'scores'/record['method']/f'{record["index"]:03d}_{record["eval_seed"]}.json'
        fixed={'input':record,'private_evaluation':evaluation.model_dump(mode='json'),'calibration':calibration}
        if path.is_file():
            previous=read_json(path)
            if previous.get('binding') != fixed:
                raise ValueError('A cached score changed its image, evaluation or calibration binding')
            if previous.get('status') == 'scored':
                results.append(previous)
                continue
        started=time.monotonic()
        try:
            reward=components.evaluator.evaluate(task,record['image_id'],renders=1,inspects=0,actions=0,invalids=0)
            if reward.preservation_status not in {'measured','not_applicable'}:
                raise ValueError('Source protection remains unavailable')
            routes={a.eval_id:a.expected_value.get('kind',a.checker_route) if isinstance(a.expected_value,dict) else a.checker_route for a in evaluation.eval_atoms}
            missing=sorted({r for r in routes.values() if not calibration['routes'].get(r,{}).get('confirmed',False)})
            if any(a.known and routes[a.atom_id] in missing for a in reward.atoms):
                raise ValueError('An unconfirmed route produced a known score')
            row={**record,'status':'scored','binding':fixed,'success':float(reward.success),'dense':reward.dense,
                'protection':reward.preservation,'preservation_status':reward.preservation_status,'private_reward':reward.to_dict(),
                'unknown_atom_ids':[a.atom_id for a in reward.atoms if not a.known],'unconfirmed_routes':missing,
                'evaluation_complete':True,'known_scores_calibrated':True,'scoring_seconds':time.monotonic()-started,
                'cost_scope':'One actual fixed render, zero actor actions; not controller episode performance',
                'severe_defect':None,'severe_defect_status':'not_reviewed','visual_success_human_verified':False}
        except (ValueError,KeyError,RuntimeError) as error:
            if 'CUDA' in str(error) or 'out of memory' in str(error).lower():raise
            row={**record,'binding':fixed,'status':'score_blocked','error':{'type':type(error).__name__,'message':str(error)},'success':None}
        atomic_json(path,row)
        results.append(row)
        print(json.dumps({'stage':'fixed_image_score','configuration':record['method'],'index':record['index'],'seed':record['eval_seed'],'status':row['status']}),flush=True)
    return results


def aggregate_fixed_scores(tasks, output, iterations=10000):
    from agentic_image.evaluation.statistics import paired_bootstrap
    output=Path(output);rows=[];missing=[];summaries={}
    for method in CONFIGURATIONS:
        observed=[]
        for index,task in enumerate(tasks):
            for seed in SEEDS:
                path=output/'scores'/method/f'{index:03d}_{seed}.json'
                if not path.is_file():
                    missing.append({'method':method,'index':index,'seed':seed,'reason':'score_pending'})
                    continue
                row=read_json(path)
                if row['method'] != method or row['index'] != index or row['task_id'] != task.task_id or row['family_id'] != task.family_id or row['eval_seed'] != seed or row['source_image_id'] != task.user_request.source_image_id:
                    raise ValueError('An aggregated score changed the frozen family or seed')
                if row.get('status') != 'scored':
                    missing.append({'method':method,'index':index,'seed':seed,'reason':row.get('error','score_blocked')})
                    continue
                if row.get('evaluation_complete') is not True or row.get('known_scores_calibrated') is not True:
                    raise ValueError('A score lacks calibrated known evidence')
                observed.append(row)
        successes=sum(r['success'] for r in observed)
        measured=[r['protection'] for r in observed if r['preservation_status']=='measured']
        if any(v is None or not math.isfinite(v) for v in measured):raise ValueError('Invalid preservation measurement')
        summaries[method]={'required_observations':320,'observed_scores':len(observed),'families_observed':len({r['family_id'] for r in observed}),
            'successes':successes,'success_rate':successes/320 if len(observed)==320 else None,
            'success_rate_bounds_with_pending':[successes/320,(successes+320-len(observed))/320],
            'dense_mean':sum(r['dense'] for r in observed)/len(observed) if observed else None,
            'dense_mean_scope':'observed_only_until_complete','preservation_measured_count':len(measured),
            'preservation_measured_mean':sum(measured)/len(measured) if measured else None,
            'preservation_not_applicable_count':sum(r['preservation_status']=='not_applicable' for r in observed),
            'observations_with_unknown_atoms':sum(bool(r['unknown_atom_ids']) for r in observed),
            'unknown_atoms':sum(len(r['unknown_atom_ids']) for r in observed),
            'unconfirmed_routes':sorted({kind for r in observed for kind in r['unconfirmed_routes']})}
        rows.extend(observed)
    comparisons=None
    if len(rows)==640 and not missing:
        paired={}
        for row in rows:paired.setdefault((row['index'],row['eval_seed']),{})[row['method']]=row
        protection=[]
        for pair in paired.values():
            left,right=pair['sft200'],pair['baseline']
            if any(left[k] != right[k] for k in ['bundle_id','source_image_id','source_value_id','plan','condition_recipe']):
                raise ValueError('Renderer comparison changed its original fixed conditions')
            if left['binding']['private_evaluation'] != right['binding']['private_evaluation'] or left['binding']['calibration'] != right['binding']['calibration']:
                raise ValueError('Renderer comparison changed its private evaluator or calibration')
            if left['preservation_status']==right['preservation_status']=='measured':protection.extend(pair.values())
            elif not (left['preservation_status']==right['preservation_status']=='not_applicable' and left['protection'] is None and right['protection'] is None):raise ValueError('Protection availability changed between configurations')
        comparisons={metric:paired_bootstrap(rows,'sft200','baseline',metric,iterations) for metric in ['success','dense']}
        comparisons['protection']=paired_bootstrap(protection,'sft200','baseline','protection',iterations) if protection else None
    report={'status':'completed' if not missing and len(rows)==640 else 'incomplete','scope':SCOPE,
        'controller_episode_evaluation':False,'fixed_families':64,'image_seeds':list(SEEDS),'required_observations_per_configuration':320,
        'trained_checkpoint_replicates':1,'image_seeds_are_not_training_replicates':True,'summaries':summaries,
        'paired_comparisons':comparisons,'paired_ci_cluster':'family/session; five image seeds remain within each family',
        'missing':missing,'severe_defect_review_performed':False,'accepted_as_controller_evaluation':False}
    atomic_json(output/'aggregate.json',report)
    return report
