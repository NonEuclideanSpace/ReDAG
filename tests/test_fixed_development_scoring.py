import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from agentic_image.core.types import PlanSpec
from agentic_image.runtime.fixed_evaluation import SEEDS, aggregate_fixed_scores, atomic_json, fixed_inputs


def tasks():
    return [SimpleNamespace(task_id=f'task:{i}',family_id=f'family:{i}',user_request=SimpleNamespace(text=f'Edit object {i}',source_image_id=f'source:{i}')) for i in range(64)]


class FakeArtifacts:
    def __init__(self):self.refs={}
    def get_ref(self,key):
        if key.startswith('source:'):return SimpleNamespace(value_id=key+':value')
        return self.refs[key]


class FixedDevelopmentScoringTests(unittest.TestCase):
    def test_missing_existing_images_wait_without_inventing_success(self):
        with tempfile.TemporaryDirectory() as d:
            images,missing=fixed_inputs(tasks(),d,0,8,'baseline',FakeArtifacts(),'revision',d)
            self.assertEqual(images,[])
            self.assertEqual(len(missing),40)
            report=aggregate_fixed_scores(tasks(),d,100)
            self.assertEqual(report['status'],'incomplete')
            self.assertEqual(len(report['missing']),640)
            self.assertIsNone(report['summaries']['baseline']['success_rate'])
            self.assertIsNone(report['paired_comparisons'])

    def test_exact_render_bindings_accept_and_changed_seed_or_pixels_do_not(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);ts=tasks();store=FakeArtifacts()
            for seed in SEEDS:
                folder=root/'000-007'/f'seed{seed}';records=[]
                metadata={'steps':50,'eta':.3,'cfg_scale':4.,'seed':seed,'optimizer_updates':0,'renderer_revision':'revision',
                    'tasks':[t.task_id for t in ts[:8]],'development_cohort':{'fixed_families':64,'indices':list(range(8)),'registered_seeds':list(SEEDS)}}
                for i,t in enumerate(ts[:8]):
                    plan=PlanSpec(global_text=t.user_request.text).model_dump(mode='json');bundle=f'bundle:{i}';image_id=f'image:{i}:{seed}'
                    image=folder/f'{i:03d}.png';image.parent.mkdir(parents=True,exist_ok=True);image.write_bytes(b'explicit synthetic fixture')
                    recipe={'bundle_id':bundle,'plan':plan,'source_image_id':t.user_request.source_image_id,'model_versions':{'renderer':'revision'}}
                    atomic_json(folder/'bundles'/(bundle.replace(':','_')+'.json'),recipe)
                    records.append({'index':i,'task_id':t.task_id,'family_id':t.family_id,'request':t.user_request.text,'source_image_id':t.user_request.source_image_id,
                        'plan':plan,'result':{'image_id':image_id,'bundle_id':bundle,'metadata':{'seed':seed,'behavior_version':'initial'}},'image':str(image),'seconds':1.})
                    store.refs[image_id]=SimpleNamespace(artifact_id=image_id,value_id=image_id+':value',shape=(512,512,3),metadata={'bundle_id':bundle,'seed':seed})
                atomic_json(folder/'metadata.json',metadata);atomic_json(folder/'results.json',records);atomic_json(folder/'summary.json',{'status':'completed','completed':8})
            bound,missing=fixed_inputs(ts,root,0,8,'baseline',store,'revision',root)
            self.assertEqual(len(bound),40);self.assertFalse(missing)
            store.refs['image:0:11'].metadata['seed']=999
            with self.assertRaisesRegex(ValueError,'artifact identity'):fixed_inputs(ts,root,0,8,'baseline',store,'revision',root)
            store.refs['image:0:11'].metadata['seed']=11
            (root/'000-007/seed11/000.png').unlink()
            bound,missing=fixed_inputs(ts,root,0,8,'baseline',store,'revision',root)
            self.assertEqual(len(bound),39);self.assertEqual(missing[0]['reason'],'render_pixels_or_recipe_missing')

    def test_complete_denominator_keeps_unknown_and_five_image_seeds_clustered(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);ts=tasks()
            for method in ('baseline','sft200'):
                for i,t in enumerate(ts):
                    for seed in SEEDS:
                        row={'method':method,'index':i,'task_id':t.task_id,'family_id':t.family_id,'eval_seed':seed,'training_seed':0,
                            'source_image_id':t.user_request.source_image_id,'source_value_id':f'source:{i}:value','bundle_id':f'bundle:{i}',
                            'plan':{'original_request':t.user_request.text},'condition_recipe':{'fixed':i},'status':'scored',
                            'evaluation_complete':True,'known_scores_calibrated':True,'success':float(i!=0),'dense':.8 if i else 0.,
                            'protection':.9,'preservation_status':'measured','unknown_atom_ids':['count'] if i==0 else [],
                            'unconfirmed_routes':['count'] if i==0 else [],'binding':{'private_evaluation':{'fixed':i},'calibration':{'fixed':True}}}
                        atomic_json(root/'scores'/method/f'{i:03d}_{seed}.json',row)
            report=aggregate_fixed_scores(ts,root,100)
            self.assertEqual(report['status'],'completed')
            self.assertEqual(report['summaries']['baseline']['success_rate'],63/64)
            self.assertEqual(report['summaries']['baseline']['observations_with_unknown_atoms'],5)
            self.assertEqual(report['paired_comparisons']['success']['observations'],320)
            self.assertEqual(report['paired_comparisons']['success']['families'],64)
            self.assertEqual(report['paired_comparisons']['success']['training_seeds'],1)
            path=root/'scores/sft200/000_11.json';changed=json.loads(path.read_text());changed['bundle_id']='different';atomic_json(path,changed)
            with self.assertRaisesRegex(ValueError,'fixed conditions'):aggregate_fixed_scores(ts,root,100)


if __name__=='__main__':unittest.main()
