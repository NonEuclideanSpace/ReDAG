import importlib.util
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
SPEC = importlib.util.spec_from_file_location('first_batch_export', ROOT / 'scripts/export_first_controller_batch.py')
export = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(export)


class FirstControllerBatchExportTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.addCleanup(patch.stopall)
        patch.object(export, 'comparison', return_value={'scope': 'tested_by_analysis_tests'}).start()
        self.tasks = [{'task_id': f'development64:sample{i}', 'family_id': f'family{i}', 'split_id': 'development',
            'user_request': {'text': f'Change object{i}', 'source_image_id': self.asset(i + 1)}} for i in range(64)]
        self.reserved = [{'sample_id': f'sample{i}', 'family_id': f'family{i}', 'split_id': 'development',
            'evaluation_index': i, 'instruction': f'Change object{i}', 'target_image_id': f'artifact:{i + 1000:064x}',
            'edit_mask_id': None} for i in range(64)]
        private = [{'task_id': row['task_id'], 'family_id': row['family_id'], 'source_image_id': row['user_request']['source_image_id'],
            'evaluation': {'original_protected_mask_id': self.asset(100 + i)}} for i, row in enumerate(self.tasks)]
        for name, rows in [('data/evaluation/development64/public_tasks.jsonl', self.tasks),
                ('data/evaluation/development64/reserved.frozen.jsonl', self.reserved),
                ('data/private_eval_v1/development64/private_tasks.jsonl', private),
                ('data/public/expanded_training128.tasks.jsonl', [{'task_id': 'train:x', 'family_id': 'train:x'}])]:
            self.write(name, ''.join(json.dumps(row) + '\n' for row in rows))
        self.common = {'tasks': self.tasks, 'image_seeds': list(export.SEEDS), 'private_target_used_for_actor_actions': False,
            'training_updates_during_evaluation': 0, 'sampler_protocol': {'parameters': {'steps': 50}}}
        self.write(export.EVALUATION / 'common_protocol.json', self.common)
        targets, prepared = [], []
        for i in range(8):
            artifact = self.asset(200 + i)
            ref_name = Path('refs') / (artifact.split(':')[1] + '.json')
            ref = json.loads((self.root / 'data/artifacts' / ref_name).read_text())
            ref['metadata'] = {'role': 'private_target', 'original_artifact_id': self.reserved[i]['target_image_id'],
                'sample_id': f'sample{i}', 'dataset_profile': 'official-viewer-reencoded-v1',
                'viewer_revision': 'viewer1', 'original_revision': 'original1'}
            self.write(Path('data/artifacts') / ref_name, ref)
            self.write(export.TARGETS / ref_name, ref)
            self.write(export.TARGETS / ref['relative_path'], 'target')
            target = {'index': i, 'sample_id': f'sample{i}', 'family_id': f'family{i}',
                'original_target_artifact_id': self.reserved[i]['target_image_id'], 'artifact_id': artifact,
                'value_id': ref['value_id'], 'reference_path': str(ref_name), 'payload_path': ref['relative_path'],
                'payload_bytes': 6, 'reencoded': True}
            targets.append(target)
            prepared.append({'evaluation_index': i, 'sample_id': f'sample{i}', 'family_id': f'family{i}',
                'split_id': 'development', 'images': [{'candidate_role': 'target', 'original_artifact_id': self.reserved[i]['target_image_id'],
                    'original_value_id': f'value:original{i}', 'artifact_id': artifact, 'value_id': ref['value_id'],
                    'relative_path': ref['relative_path'], 'viewer_revision': 'viewer1'}]})
        self.write(export.TARGETS / 'targets.json', {'schema': 'first_batch_private_targets_v1',
            'indices': list(export.INDICES), 'seed': 11, 'targets': targets})
        self.write('data/evaluation/development64/prepared.jsonl', ''.join(json.dumps(row) + '\n' for row in prepared))
        self.checkpoint = self.root / 'runs/full-actor-rl/rl_joint/checkpoints/final'
        self.archive = str(self.root / 'delivery/joint-final.tar.gz')
        for method in export.METHODS:
            self.method(method)
        for name in ('deployment/development64-controller-joint_rl-000-seed11.exit.json', 'deployment/env-restored-gpu-check.exit.json'):
            self.write(name, {'returncode': 0})
        self.write('deployment/env-restored-gpu-check.json', {'status': 'verified', 'gpu_verified': True,
            'checkpoint_recovery_verified': True, 'recovery_archive': self.archive})
        self.write('deployment/full-joint-rl32-export.json', {'status': 'completed', 'stage': 'joint',
            'effective_target_met': True, 'required_effective_updates': 32, 'checkpoint': str(self.checkpoint), 'source': self.archive})

    def write(self, relative, value):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value if isinstance(value, str) else json.dumps(value))
        return path

    def asset(self, number):
        artifact = f'artifact:{number:064x}'
        self.write(f'data/artifacts/values/{number}.png', 'payload')
        self.write(f'data/artifacts/refs/{number:064x}.json', {'artifact_id': artifact,
            'value_id': f'value:{number:064x}', 'relative_path': f'values/{number}.png', 'kind': 'image'})
        return artifact

    def method(self, method):
        source = {'base': 'untrained_base', 'sft': 'completed_sft_exports'}.get(method, 'committed_rl_pair')
        policy = {'method': method, 'source': source}
        if method == 'joint_rl':
            policy.update(commit_path=str(self.checkpoint / 'COMMITTED.json'), commit={'status': 'COMMITTED'},
                completion={'effective_target_met': True})
        self.write(export.EVALUATION / method / 'registration.json', {'method': method,
            'common_protocol': self.common, 'evaluated_policy': policy})
        worker = self.root / export.EVALUATION / method / 'workers/000_seed11'
        self.write(worker / 'policy_load.json', {'source': source})
        for i in range(8):
            folder = self.root / export.EVALUATION / method / 'episodes' / f'{i:03d}' / 'seed11'
            visual = self.write(worker / f'actor_images/{i}.pt', 'visual tensor')
            output = self.asset(400 + i)
            bundle_id = f'bundle:{i:064x}'
            trace = worker / f'renders/{i}/trace.pt'
            render = {'image_id': output, 'bundle_id': bundle_id, 'trajectory_id': str(trace),
                'metadata': {'seed': 11, 'behavior_version': method + ':32'}}
            action = self.write(folder / 'action_00.pt', {'snapshot': {'visual_files': [str(visual)]},
                'text': '{"name":"RENDER"}', 'seed': 11, 'behavior_version': method + ':32'})
            self.write(folder / 'cost_00.json', {'observed_gpu_seconds': 1})
            self.write(folder / 'result_00.json', render)
            self.write(trace, {'bundle_id': bundle_id, 'behavior_version': method + ':32'})
            self.write(trace.with_name('result.json'), render)
            bundle = worker / f'bundles/bundle_{i:064x}.pt'
            self.write(bundle, {'bundle_id': bundle_id, 'artifact_bindings': {'source': self.tasks[i]['user_request']['source_image_id']}})
            self.write(bundle.with_suffix('.json'), {'bundle_id': bundle_id})
            terminal = {'success': True, 'preservation_status': 'measured', 'preservation': .8, 'atoms': [{'known': True}]}
            self.write(folder / 'state.json', {'done': True, 'task': self.tasks[i], 'delivered_image_id': output})
            self.write(folder / 'terminal_private.json', terminal)
            self.write(folder / 'episode.json', {'scope': 'actual_controller_episode', 'method': method, 'index': i,
                'task_id': self.tasks[i]['task_id'], 'family_id': self.tasks[i]['family_id'], 'seed': 11,
                'private_target_used_for_actions': False, 'training_updates': 0, 'sampler_parameters': {'steps': 50},
                'protocol_failure': False, 'delivered_image_id': output, 'budget': {'actions': 1, 'renders': 1, 'invalid': 0},
                'terminal': terminal, 'loaded_state_verification': {'source': source, 'zero_output_projections_verified': True,
                    'actor': {'exact_tensor_equality': True}, 'renderer': {'exact_tensor_equality': True}},
                'episode_record': {'action_paths': [str(action)], 'environment_path': str(folder / 'state.json'), 'render_records': [render]}})

    def plan(self):
        return export.collect_plan(self.root, loader=lambda path: json.loads(path.read_text()))

    def test_archive_contains_selected_complete_replay_closure(self):
        unrelated = self.write(export.EVALUATION / 'joint_rl/workers/008_seed11/unrelated.pt', 'never include')
        plan = self.plan()
        self.assertEqual(plan['status'], 'ready')
        self.assertEqual(plan['observed_episodes'], 32)
        self.assertNotIn(str(unrelated), plan['files'])
        for category in ('condition_tensor', 'render_trace', 'actor_visual_replay', 'action_replay', 'target_payload'):
            self.assertGreater(plan['bytes_by_category'][category], 0)
        output, receipt = self.root / 'first.tar.gz', self.root / 'receipt.json'
        result = export.write_archive(plan, output, receipt, rate_mib=32, lower_priority=False)
        self.assertEqual(result['schema'], 'first_batch_archive_receipt_v1')
        self.assertEqual(result['source'], str(output.resolve()))
        self.assertEqual(result['mtime'], int(output.stat().st_mtime))
        with tarfile.open(output) as archive:
            self.assertEqual(len(archive.getnames()), len(plan['files']) + 2)
            self.assertIn('FIRST_BATCH/summary.json', archive.getnames())
        self.assertFalse(result['full1280_complete'])
        with self.assertRaises(FileExistsError):
            export.write_archive(plan, output, receipt, lower_priority=False)

    def test_missing_fourth_policy_never_becomes_complete(self):
        for index in range(8):
            (self.root / export.EVALUATION / 'joint_rl/episodes' / f'{index:03d}' / 'seed11/episode.json').unlink()
        plan = self.plan()
        self.assertEqual(plan['observed_episodes'], 24)
        self.assertEqual(plan['status'], 'not_ready')
        self.assertIsNone(plan['summary']['configurations']['joint_rl']['automatic_success_rate'])
        with self.assertRaisesRegex(RuntimeError, 'not ready'):
            export.write_archive(plan, self.root / 'no.tar.gz', self.root / 'no.json', lower_priority=False)
        self.assertFalse((self.root / 'no.tar.gz.part').exists())

    def test_missing_replay_dependencies_are_explicit_and_block_packaging(self):
        for category, name in [('condition_tensor', 'bundles/bundle_' + '0' * 64 + '.pt'),
                ('actor_visual_replay', 'actor_images/0.pt')]:
            path = self.root / export.EVALUATION / 'sft/workers/000_seed11' / name
            data = path.read_bytes()
            path.unlink()
            plan = self.plan()
            self.assertEqual(plan['status'], 'not_ready')
            self.assertIn(category, [row['category'] for row in plan['unresolved_dependencies']])
            path.write_bytes(data)

    def test_failed_or_mismatched_recovery_proof_is_not_accepted(self):
        self.write('deployment/env-restored-gpu-check.exit.json', {'returncode': 1})
        self.assertEqual(self.plan()['status'], 'not_ready')
        self.write('deployment/env-restored-gpu-check.json', {'status': 'verified', 'gpu_verified': True,
            'checkpoint_recovery_verified': True, 'recovery_archive': 'different.tar.gz'})
        with self.assertRaisesRegex(ValueError, 'different recovery archive'):
            self.plan()

    def test_saved_failure_with_no_output_remains_a_failure(self):
        folder = self.root / export.EVALUATION / 'base/episodes/001/seed11'
        row = json.loads((folder / 'episode.json').read_text())
        row.update(protocol_failure=True, delivered_image_id=None)
        row['terminal'].update(success=False, preservation_status='not_evaluated', preservation=None)
        self.write(folder / 'episode.json', row)
        self.write(folder / 'terminal_private.json', row['terminal'])
        state = json.loads((folder / 'state.json').read_text())
        state['delivered_image_id'] = None
        self.write(folder / 'state.json', state)
        summary = self.plan()['summary']['configurations']['base']
        self.assertEqual(summary['automatic_successes'], 7)
        self.assertEqual(summary['protocol_failures'], 1)
        self.assertEqual(summary['missing_delivered_images'], 1)

    def test_foreign_action_reference_and_symlink_are_rejected(self):
        path = self.root / export.EVALUATION / 'base/episodes/000/seed11/action_00.pt'
        row = json.loads(path.read_text())
        row['snapshot']['visual_files'] = [str(self.root / export.EVALUATION / 'sft/workers/000_seed11/actor_images/0.pt')]
        self.write(path, row)
        with self.assertRaisesRegex(ValueError, 'outside its selected worker'):
            self.plan()
        path.unlink()
        path.symlink_to(self.root / export.EVALUATION / 'sft/episodes/000/seed11/action_00.pt')
        with self.assertRaisesRegex(ValueError, 'symlinks'):
            self.plan()

    def test_target_preparation_preserves_provenance_and_rejects_wrong_family(self):
        plan = export.prepare_targets(self.root)
        self.assertEqual(len(plan['manifest']['targets']), 8)
        self.assertEqual(len(plan['files']), 16)
        self.assertTrue(all(row['archive_path'].startswith(str(export.TARGETS)) for row in plan['files'].values()))
        path = self.root / 'data/evaluation/development64/prepared.jsonl'
        rows = export.lines(path)
        rows[0]['family_id'] = 'other-family'
        path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        with self.assertRaisesRegex(ValueError, 'identity differs'):
            export.prepare_targets(self.root)

    def test_changed_selected_source_cannot_publish_archive(self):
        plan = self.plan()
        self.write(export.EVALUATION / 'sft/workers/000_seed11/actor_images/0.pt', 'changed tensor')
        with self.assertRaisesRegex(RuntimeError, 'changed after planning'):
            export.write_archive(plan, self.root / 'no.tar.gz', self.root / 'no.json', lower_priority=False)
        self.assertFalse((self.root / 'no.tar.gz').exists())


if __name__ == '__main__':
    unittest.main()
