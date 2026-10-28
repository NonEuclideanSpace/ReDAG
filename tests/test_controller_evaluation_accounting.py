import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest


spec=importlib.util.spec_from_file_location('controller_evaluation_entry',Path(__file__).parents[1]/'scripts/evaluate_development_controller.py')
entry=importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)


class ControllerAccountingTests(unittest.TestCase):
    def test_missing_or_failed_episodes_keep_the_fixed_denominator(self):
        tasks=[SimpleNamespace(task_id=f'task:{i}',family_id=f'family:{i}') for i in range(64)]
        with tempfile.TemporaryDirectory() as directory:
            report=entry.aggregate(tasks,directory)
            self.assertIsNone(report['configurations']['base']['success_rate'])
            self.assertEqual(len(report['configurations']['base']['missing']),320)
            for index,task in enumerate(tasks):
                for seed in entry.SEEDS:
                    row={'method':'base','task_id':task.task_id,'family_id':task.family_id,'seed':seed,
                        'terminal':{'success':False,'atoms':[{'known':False}]},'protocol_failure':True,
                        'delivered_image_id':None,'budget':{'actions':2,'renders':0},
                        'costs':{'historical_measurements_complete':False,'observed_gpu_seconds':1.}}
                    path=Path(directory)/'base/episodes'/f'{index:03d}'/f'seed{seed}'/'episode.json'
                    entry.atomic_json(path,row)
            report=entry.aggregate(tasks,directory)
            base=report['configurations']['base']
            self.assertEqual(base['observed_episodes'],320)
            self.assertEqual(base['success_rate'],0.)
            self.assertEqual(base['protocol_failures'],320)
            self.assertEqual(base['no_delivered_image'],320)
            self.assertEqual(base['cost_ledger_complete_episodes'],0)
            self.assertFalse(report['cross_configuration_complete'])

    def test_checkpoint_identity_requires_an_existing_explicit_weight(self):
        self.assertIsNone(entry.policy_identity(None,'actor'))
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):entry.policy_identity(directory,'actor')
            path=Path(directory)/'policy/adapter_model.safetensors';path.parent.mkdir();path.write_bytes(b'synthetic fixture')
            (path.parent/'adapter_config.json').write_text('{}')
            identity=entry.policy_identity(directory,'actor')
            self.assertEqual(identity['path'],str(path.resolve()))
            self.assertEqual(identity['bytes'],len(b'synthetic fixture'))


if __name__=='__main__':unittest.main()
