import argparse
import ast
from collections import Counter
import contextlib
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('visual_admission', ROOT / 'src/agentic_image/data/visual_admission.py')
admission = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(admission)


def case(index=0):
    task, family, state = f'task:{index}', f'family:{index}', f'public_rule:{index}'
    images = {role: 'artifact:' + hashlib.sha256(f'{index}:{role}'.encode()).hexdigest() for role in admission.IMAGE_KEYS}
    bundle = {'task_id': task, 'family_id': family, 'state_id': state,
        'public_source_image_id': images['SOURCE'], 'public_request': 'Make the cup green.',
        'private_outcomes': [{'stage': 'initial', 'image_id': images['INITIAL REQUEST']},
            {'stage': 'injected', 'image_id': images['CONTROLLED WRONG REQUEST']},
            {'stage': 'recovery', 'image_id': images['RECOVERY'], 'before_image_id': images['CONTROLLED WRONG REQUEST'], 'positive': True}],
        'verified_record_ids': []}
    records = []
    for name, phase, visual in [('plan', 0, False), ('initial_render', 0, False),
        ('error_recovery_read', 3, False), ('injected_contract_rollback', 2, False),
        ('positive_draft', 1, False), ('positive_apply', 1, False),
        ('successful_initial_finish', 2, True), ('terminal_finish', 2, True)]:
        proof = {'execution_evidence': f'/fixture/{index}/events.jsonl', 'visual_success_label': visual}
        if phase == 1:
            proof.update(repair_success_label=True, all_atoms_known=True,
                no_previously_passed_requirement_regressed=True, minimum_dense_improvement=.1,
                before_image_id=images['CONTROLLED WRONG REQUEST'], after_image_id=images['RECOVERY'])
        record = {'record_id': state + ':' + name, 'state_id': state, 'family_id': family, 'phase': phase,
            'verifier_provenance': proof, 'verified': True, 'target_ids': [1, 248046], 'snapshot_path': '/fixture/snapshot.pt'}
        records.append(record)
        bundle['verified_record_ids'].append(record['record_id'])
    review = {'task_id': task, 'family_id': family, 'public_request': bundle['public_request'], 'images': images,
        'labels': dict.fromkeys(admission.LABELS, True),
        'evidence': {'repair_positive': 'The edited object recovers the requested attribute and retains the surrounding scene.',
            'initial_success': 'The requested attribute is visible on the specified object in the initial image.',
            'recovery_success': 'The recovery shows the requested object attribute without an unrelated scene change.'},
        'reviewer': 'fixture_reviewer', 'review_type': 'ai', 'human_reviewed': False,
        'actual_visual_inspection': True, 'reviewed_image_paths': ['/Users/reviewer/actually-viewed-four-column-contact.jpg']}
    events = []
    for name, role in [('successful_initial_finish', 'INITIAL REQUEST'), ('terminal_finish', 'RECOVERY')]:
        task_value = {'task_id': task, 'family_id': family}
        events.append({'name': name, 'internal_action': {'kind': 'FINISH'}, 'result': {'error': None},
            'public_before_state': {'task': task_value, 'done': False},
            'public_after_state': {'task': task_value, 'done': True, 'delivered_image_id': images[role]}})
    return records, bundle, review, events


class VisualAdmissionTests(unittest.TestCase):
    def test_false_and_unknown_remove_only_visual_labels_and_preserve_inputs(self):
        records, bundle, review, events = case()
        budget_records, budget_bundle, _, _ = case(1)
        budget = budget_records[-1]
        budget['verifier_provenance'].update(visual_success_label=False, verification='actual_render_budget_exhausted')
        records.append(budget)
        review['labels'].update(repair_positive=False, initial_success=None)
        before = deepcopy((records, bundle, budget_bundle, review, events))
        included, audit, summary = admission.filter_visual_sft_records(records, [bundle, budget_bundle], [review], lambda _: events)
        self.assertEqual([row['record_id'] for row in included], [row['record_id'] for row in records[:4]] + [records[7]['record_id'], budget['record_id']])
        excluded = [item for item in audit if item['decision'] == 'exclude']
        self.assertEqual(len(excluded), 3)
        self.assertEqual(Counter(item['reason'] for item in excluded), {
            'ai_visual_repair_positive_false': 2, 'ai_visual_initial_success_unknown': 1})
        self.assertIsNone(next(item['review_value'] for item in excluded if item['visual_label'] == 'initial_success'))
        self.assertEqual((records, bundle, budget_bundle, review, events), before)
        self.assertEqual(summary['included_families'], 2)
        self.assertFalse(summary['labels_or_records_added'])
        self.assertEqual(len(audit), len(records))
        for row in included:
            self.assertTrue(any(row is original for original in records))

    def test_true_review_keeps_existing_records_without_creating_new_positive_actions(self):
        records, bundle, review, events = case()
        included, _, summary = admission.filter_visual_sft_records(records, [bundle], [review], lambda _: events)
        self.assertEqual(included, records)
        self.assertEqual(summary['included_positive_states'], 1)
        included, _, _ = admission.filter_visual_sft_records(records[:4], [bundle], [review], lambda _: events)
        self.assertEqual(included, records[:4])
        self.assertFalse(any(row['phase'] == 1 for row in included))

    def test_missing_duplicate_and_conflicting_reviews_fail_closed(self):
        records, bundle, review, events = case()
        for reviews in ([], [review, deepcopy(review)], [review, review | {'labels': dict.fromkeys(admission.LABELS, False)}]):
            with self.subTest(count=len(reviews)), self.assertRaisesRegex(ValueError, 'Missing required|Duplicate or conflicting'):
                admission.filter_visual_sft_records(records, [bundle], reviews, lambda _: events)

    def test_artifact_and_actual_repair_pair_mismatch_reject_even_false_labels(self):
        for mutation in ('review_image', 'before', 'after', 'bundle_before', 'family'):
            records, bundle, review, events = case()
            review['labels']['repair_positive'] = False
            if mutation == 'review_image':
                review['images']['RECOVERY'] = 'artifact:wrong'
            elif mutation in {'before', 'after'}:
                records[4]['verifier_provenance'][mutation + '_image_id'] = 'artifact:wrong'
            elif mutation == 'bundle_before':
                bundle['private_outcomes'][-1]['before_image_id'] = 'artifact:wrong'
            else:
                review['family_id'] = 'family:wrong'
            with self.subTest(mutation=mutation), self.assertRaisesRegex(ValueError, 'identity mismatch|actual injected/recovery pair'):
                admission.filter_visual_sft_records(records, [bundle], [review], lambda _: events)

    def test_finish_requires_actual_terminal_delivery_of_the_matching_review_image(self):
        for mutation in ('not_done', 'wrong_image', 'wrong_task', 'not_finish', 'duplicate_event'):
            records, bundle, review, events = case()
            review['labels']['initial_success'] = False
            if mutation == 'not_done':
                events[0]['public_after_state']['done'] = False
            elif mutation == 'wrong_image':
                events[0]['public_after_state']['delivered_image_id'] = review['images']['RECOVERY']
            elif mutation == 'wrong_task':
                events[0]['public_after_state']['task'] = {'task_id': 'wrong', 'family_id': bundle['family_id']}
            elif mutation == 'not_finish':
                events[0]['internal_action']['kind'] = 'SELECT'
            else:
                events.append(deepcopy(events[0]))
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                admission.filter_visual_sft_records(records, [bundle], [review], lambda _: events)

    def test_review_attestation_schema_is_strict_but_mac_paths_are_not_opened(self):
        records, bundle, review, events = case()
        self.assertFalse(Path(review['reviewed_image_paths'][0]).exists())
        self.assertEqual(len(admission.filter_visual_sft_records(records, [bundle], [review], lambda _: events)[0]), 8)
        for key, value in [('actual_visual_inspection', False), ('human_reviewed', True), ('reviewer', ''), ('reviewed_image_paths', [])]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                admission.filter_visual_sft_records(records, [bundle], [review | {key: value}], lambda _: events)
        for value in (1, 'true', 'unknown'):
            invalid = deepcopy(review)
            invalid['labels']['repair_positive'] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'bool or null'):
                admission.filter_visual_sft_records(records, [bundle], [invalid], lambda _: events)


class TrainingEntryTests(unittest.TestCase):
    def test_actual_entry_registration_is_stable_for_resume_with_bound_review_and_audit(self):
        source = ROOT / 'scripts/train_full_actor_sft.py'
        main = next(node for node in ast.parse(source.read_text()).body if isinstance(node, ast.FunctionDef) and node.name == 'main')
        sampling_spec = importlib.util.spec_from_file_location('actual_phase_sampling', ROOT / 'src/agentic_image/learning/sampling.py')
        sampling = importlib.util.module_from_spec(sampling_spec)
        sampling_spec.loader.exec_module(sampling)
        class ReadyForTraining(Exception):
            pass
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            collection = root / 'collection'
            collection.mkdir()
            snapshot, event = root / 'snapshot.pt', root / 'events.jsonl'
            snapshot.write_bytes(b'fixture snapshot')
            event.write_text('{}\n')
            records, bundles, reviews = [], [], []
            for index in range(128):
                current, bundle, review, _ = case(index)
                chosen = current[:6] if index < 4 else current[:4]
                for record in chosen:
                    record['snapshot_path'] = str(snapshot)
                    record['verifier_provenance']['execution_evidence'] = str(event)
                records.extend(chosen)
                bundles.append(bundle)
                if index < 4:
                    reviews.append(review)
            def write_rows(path, values):
                path = Path(path)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(''.join(json.dumps(value) + '\n' for value in values))
            def read_rows(path):
                return [json.loads(line) for line in Path(path).read_text().splitlines() if line]
            write_rows(collection / 'actor_records.jsonl', records)
            write_rows(collection / 'state_bundles.jsonl', bundles)
            (collection / 'completed.json').write_text('{"status":"completed"}')
            review_path = root / 'reviews.jsonl'
            write_rows(review_path, reviews)
            write_rows(root / 'data/training_expansion/training_selection_v1.frozen.jsonl',
                [{'family_id': f'family:{index}'} for index in range(128)])
            calls = []
            def train(components, stage, updates, resume):
                calls.append({'stage': stage, 'updates': updates, 'resume': resume})
                raise ReadyForTraining()
            namespace = {'argparse': argparse, 'Counter': Counter, 'hashlib': hashlib, 'json': json,
                'Path': Path, 'read_jsonl': read_rows, 'write_jsonl': write_rows,
                'atomic_json': lambda path, value: Path(path).write_text(json.dumps(value, indent=2)),
                'filter_visual_sft_records': admission.filter_visual_sft_records,
                'prepare_full_actor_admission': admission.prepare_full_actor_admission,
                'load_config': lambda _: {'devices': {}, 'sft': {}, 'data': {}, 'models': {'actor_policy': 'fixture-policy'}, 'seed': 11},
                'Components': lambda value: SimpleNamespace(config=value), 'train_actor_sft': train,
                'PhaseBalancedSampler': sampling.PhaseBalancedSampler,
                'yaml': SimpleNamespace(safe_dump=lambda value, **_: json.dumps(value))}
            exec(compile(ast.Module(body=[main], type_ignores=[]), str(source), 'exec'), namespace)
            argv = ['train_full_actor_sft.py', '--collection', str(collection), '--output', str(root / 'filtered'),
                    '--visual-reviews', str(review_path)]
            for resume in (False, True):
                with contextlib.chdir(root), patch.object(sys, 'argv', argv + (['--resume'] if resume else [])), self.assertRaises(ReadyForTraining):
                    namespace['main']()
            registration = json.loads((root / 'filtered/registration.json').read_text())
            self.assertEqual(registration['records'], 520)
            self.assertEqual(registration['phase_counts'], [5, 7, 5, 3])
            self.assertEqual(registration['updates'], 200)
            visual = registration['visual_admission']
            self.assertEqual(visual['reviews']['sha256'], hashlib.sha256(review_path.read_bytes()).hexdigest())
            self.assertEqual(visual['audit_sha256'], hashlib.sha256(Path(visual['audit_path']).read_bytes()).hexdigest())
            self.assertEqual([call['resume'] for call in calls], [False, True])
            self.assertEqual(len(list((root / 'filtered/visual-admission').glob('*.json'))), 1)
            devices = {'actor': 'cuda:1', 'renderer': 'cuda:0', 'encoders': 'cuda:0', 'judge': 'cuda:1'}
            namespace['load_config'] = lambda _: {'devices': dict(devices), 'sft': {}, 'data': {},
                'models': {'actor_policy': 'fixture-policy'}, 'seed': 11, 'renderer': {}}
            def finish_training(components, stage, updates, resume):
                self.assertEqual(components.config['devices']['actor'], 'cuda:0')
                write_rows(Path('runs/actor-full-200/sft_actor/metrics.jsonl'),
                    [{'step': step, 'record_ids': ['fixture-action'], 'family_ids': ['fixture-family']}
                     for step in range(1, updates + 1)])
                return 'fixture-trained-policy'
            namespace['train_actor_sft'] = finish_training
            (collection / 'sampler_selection.json').write_text(json.dumps({'steps': 50, 'eta': .15, 'cfg_scale': 4.0}))
            with contextlib.chdir(root), patch.object(sys, 'argv', argv + ['--device', 'cuda:0']):
                namespace['main']()
            self.assertEqual(json.loads((root / 'filtered/training.yaml').read_text())['devices']['actor'], 'cuda:0')
            self.assertEqual(json.loads((root / 'filtered/rl_unfrozen.yaml').read_text())['devices'], devices)

    def test_actual_entry_checks_512_and_four_positive_states_after_filtering(self):
        source = ROOT / 'scripts/train_full_actor_sft.py'
        main = next(node for node in ast.parse(source.read_text()).body if isinstance(node, ast.FunctionDef) and node.name == 'main')
        for original_count, expected_message in ((512, 'at least 512'), (520, 'at least four')):
            with self.subTest(original_count=original_count), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                collection = root / 'collection'
                collection.mkdir()
                snapshot, event = root / 'snapshot.pt', root / 'events.jsonl'
                snapshot.write_bytes(b'fixture snapshot')
                event.write_text('{}\n')
                all_records, bundles, reviews = [], [], []
                for index in range(128):
                    records, bundle, review, _ = case(index)
                    chosen = records[:4]
                    if index < 4:
                        if original_count == 512:
                            chosen = [records[0], records[2]]
                        chosen += records[4:6]
                        review['labels']['repair_positive'] = False
                        reviews.append(review)
                    for record in chosen:
                        record['snapshot_path'] = str(snapshot)
                        record['verifier_provenance']['execution_evidence'] = str(event)
                    all_records.extend(chosen)
                    bundles.append(bundle)
                def write_rows(path, values):
                    path = Path(path)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(''.join(json.dumps(value) + '\n' for value in values))
                def read_rows(path):
                    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]
                write_rows(collection / 'actor_records.jsonl', all_records)
                write_rows(collection / 'state_bundles.jsonl', bundles)
                (collection / 'completed.json').write_text('{"status":"completed"}')
                review_path = root / 'reviews.jsonl'
                write_rows(review_path, reviews)
                write_rows(root / 'data/training_expansion/training_selection_v1.frozen.jsonl',
                    [{'family_id': f'family:{index}'} for index in range(128)])
                def atomic_json(path, value):
                    Path(path).write_text(json.dumps(value, indent=2))
                namespace = {'argparse': argparse, 'Counter': Counter, 'hashlib': hashlib, 'json': json,
                    'Path': Path, 'read_jsonl': read_rows, 'write_jsonl': write_rows, 'atomic_json': atomic_json,
                    'filter_visual_sft_records': admission.filter_visual_sft_records,
                    'prepare_full_actor_admission': admission.prepare_full_actor_admission,
                    'load_config': lambda _: {'devices': {}, 'sft': {}, 'models': {'actor_policy': 'fixture-policy'}, 'seed': 11},
                    'Components': lambda _: self.fail('Model loading must not happen after rejected admission'),
                    'train_actor_sft': lambda *a, **kw: self.fail('Training must not start after rejected admission')}
                exec(compile(ast.Module(body=[main], type_ignores=[]), str(source), 'exec'), namespace)
                argv = ['train_full_actor_sft.py', '--collection', str(collection), '--output', str(root / 'filtered'),
                        '--visual-reviews', str(review_path)]
                with contextlib.chdir(root), patch.object(sys, 'argv', argv), self.assertRaisesRegex(ValueError, expected_message):
                    namespace['main']()
                audits = list((root / 'filtered/visual-admission').glob('*.json'))
                self.assertEqual(len(audits), 1)
                audit = json.loads(audits[0].read_text())
                self.assertEqual(audit['summary']['input_records'], original_count)
                self.assertEqual(audit['summary']['included_records'], original_count - 8)
                self.assertEqual(audit['summary']['included_families'], 128)
                self.assertEqual(audit['summary']['included_positive_states'], 0)
                self.assertEqual(read_rows(collection / 'actor_records.jsonl'), all_records)
                self.assertFalse((root / 'filtered/registration.json').exists())


class CPUReadyTests(unittest.TestCase):
    def fixture(self, root):
        snapshot, event = root / 'snapshot.pt', root / 'events.jsonl'
        snapshot.write_bytes(b'fixture snapshot')
        event.write_text('{}\n')
        shards = [root / 'shard-0', root / 'shard-1']
        records, bundles, reviews = [[], []], [[], []], []
        for index in range(128):
            current, bundle, review, _ = case(index)
            chosen = current[:6] if index < 4 else current[:4]
            for record in chosen:
                record['snapshot_path'] = str(snapshot)
                record['verifier_provenance']['execution_evidence'] = str(event)
            records[index % 2].extend(chosen)
            bundles[index % 2].append(bundle)
            if index < 4:
                reviews.append(review)
        def write_rows(path, values):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(''.join(json.dumps(value) + '\n' for value in values))
        for index, shard in enumerate(shards):
            write_rows(shard / 'actor_records.jsonl', records[index])
            write_rows(shard / 'state_bundles.jsonl', bundles[index])
            (shard / 'completed.json').write_text('{"status":"completed"}')
        review_path, selection = root / 'reviews.jsonl', root / 'selection.jsonl'
        write_rows(review_path, reviews)
        write_rows(selection, [{'family_id': f'family:{index}'} for index in range(128)])
        return shards, review_path, selection

    def test_real_cpu_only_cli_is_idempotent_and_rejects_changed_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shards, reviews, selection = self.fixture(root)
            ready = root / 'ready.json'
            command = [sys.executable, str(ROOT / 'scripts/prepare_full_actor_sft_admission.py'),
                '--collection', str(shards[0]), '--collection', str(shards[1]),
                '--visual-reviews', str(reviews), '--selection', str(selection),
                '--output', str(root / 'admission'), '--ready', str(ready)]
            first = subprocess.run(command, capture_output=True, text=True, timeout=20, check=True)
            value = json.loads(ready.read_text())
            self.assertTrue(json.loads(first.stdout)['created'])
            self.assertEqual(value['status'], 'completed')
            self.assertEqual(value['counts']['records'], 520)
            self.assertEqual(value['counts']['independent_families'], 128)
            self.assertEqual(value['counts']['positive_states'], 4)
            self.assertEqual(value['counts']['phase_counts'], [5, 7, 5, 3])
            self.assertEqual(value['identity']['planned_optimizer_updates'], 200)
            before, timestamp = ready.read_bytes(), ready.stat().st_mtime_ns
            again = subprocess.run(command, capture_output=True, text=True, timeout=20, check=True)
            self.assertFalse(json.loads(again.stdout)['created'])
            self.assertEqual(ready.stat().st_mtime_ns, timestamp)
            self.assertEqual(ready.read_bytes(), before)
            reviews.write_bytes(reviews.read_bytes() + b'\n')
            changed = subprocess.run(command, capture_output=True, text=True, timeout=20)
            self.assertNotEqual(changed.returncode, 0)
            self.assertIn('different identity', changed.stderr)
            self.assertEqual(ready.read_bytes(), before)

    def test_incomplete_collection_and_missing_review_cannot_publish_ready(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shards, reviews, selection = self.fixture(root)
            ready = root / 'ready.json'
            with self.assertRaisesRegex(ValueError, 'at least 200'):
                result = admission.prepare_full_actor_admission(shards, root / 'output', reviews,
                    selection_path=selection, updates=199)
                admission.publish_visual_admission_ready(result, ready)
            self.assertFalse(ready.exists())
            (shards[1] / 'completed.json').write_text('{"status":"running"}')
            with self.assertRaisesRegex(ValueError, 'not completed'):
                result = admission.prepare_full_actor_admission(shards, root / 'output', reviews, selection_path=selection)
                admission.publish_visual_admission_ready(result, ready)
            self.assertFalse(ready.exists())
            (shards[1] / 'completed.json').write_text('{"status":"completed"}')
            reviews.write_text('\n'.join(reviews.read_text().splitlines()[1:]) + '\n')
            with self.assertRaisesRegex(ValueError, 'Missing required actual visual review'):
                result = admission.prepare_full_actor_admission(shards, root / 'output', reviews, selection_path=selection)
                admission.publish_visual_admission_ready(result, ready)
            self.assertFalse(ready.exists())

    def test_shared_admission_preserves_evidence_eos_and_actual_phase_sampler_gates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shards, reviews, selection = self.fixture(root)
            path = shards[0] / 'actor_records.jsonl'
            original = path.read_bytes()
            rows = [json.loads(line) for line in original.splitlines() if line]
            rows[0]['target_ids'][-1] = 0
            path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
            with self.assertRaisesRegex(ValueError, 'snapshot or execution evidence'):
                admission.prepare_full_actor_admission(shards, root / 'output', reviews, selection_path=selection)
            path.write_bytes(original)
            (root / 'snapshot.pt').unlink()
            with self.assertRaisesRegex(ValueError, 'snapshot or execution evidence'):
                admission.prepare_full_actor_admission(shards, root / 'output', reviews, selection_path=selection)
            (root / 'snapshot.pt').write_bytes(b'fixture snapshot')
            for shard in shards:
                path = shard / 'actor_records.jsonl'
                rows = [json.loads(line) for line in path.read_text().splitlines() if line]
                for row in rows:
                    if row['phase'] == 3:
                        row['phase'] = 0
                path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
            with self.assertRaisesRegex(ValueError, 'phase 3 has no verified actions'):
                admission.prepare_full_actor_admission(shards, root / 'output', reviews, selection_path=selection)


if __name__ == '__main__':
    unittest.main()
