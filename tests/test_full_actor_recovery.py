import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from PIL import Image
import yaml

from agentic_image.core import ArtifactStore, CheckSpec, EpisodeEnvironment, Observation, PlanSpec, PublicCheckAtom, PublicTask, RenderResult, UserRequest
from agentic_image.core.artifacts import canonical_json, content_id


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('full_actor_recovery', ROOT / 'scripts/recover_full_actor_collection.py')
recovery = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(recovery)
SAMPLING_SPEC = importlib.util.spec_from_file_location('actual_sampling', ROOT / 'src/agentic_image/learning/sampling.py')
sampling = importlib.util.module_from_spec(SAMPLING_SPEC)
SAMPLING_SPEC.loader.exec_module(sampling)
stream_seed = sampling.stream_seed


def write_rows(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(value) + '\n' for value in values))


class Compiler:
    def __init__(self):
        self.bundles = {}

    def compile(self, plan, source_image_id):
        identifier = content_id('bundle', canonical_json({'plan': plan, 'source': source_image_id}))
        bundle = SimpleNamespace(bundle_id=identifier, compiler_version='dag-agent-v3')
        self.bundles[identifier] = bundle
        return bundle


class Observer:
    def check(self, task, image_id):
        return [Observation(observation_id='observation:' + image_id.split(':')[1], image_id=image_id,
            check_id=task.check_spec.atoms[0].check_id, known=True, score=.9, passed=True)]


class FixtureBridge:
    def __init__(self, path, artifacts, config):
        self.directory = Path(path)
        self.artifacts = artifacts
        self.config = config
        self.behavior_version = 'renderer-fixed'
        self.fresh_calls = []
        self.decode_calls = []

    def render(self, bundle, seed, job_id):
        directory = self.directory / job_id.replace(':', '_').replace('/', '_')
        directory.mkdir(parents=True, exist_ok=True)
        result_path = directory / 'result.json'
        if result_path.exists():
            return RenderResult.model_validate(recovery.read_json(result_path))
        trace_path = directory / 'trace.pt'
        if trace_path.exists():
            self.decode_calls.append(job_id)
        else:
            self.fresh_calls.append(job_id)
            recovery.atomic_json(trace_path, trace_values(bundle, seed, job_id, self.config))
        color = tuple(hashlib.sha256(job_id.encode()).digest()[:3])
        ref = self.artifacts.put_image(Image.new('RGB', (512, 512), color), producer_id=job_id)
        result = RenderResult(image_id=ref.artifact_id, bundle_id=bundle.bundle_id,
            trajectory_id=str(trace_path.resolve()), metadata={'seed': seed, 'behavior_version': self.behavior_version})
        recovery.atomic_json(result_path, result.model_dump(mode='json'))
        return result


def trace_values(bundle, seed, job_id, config):
    return {'render_id': job_id, 'bundle_id': bundle.bundle_id, 'behavior_version': 'renderer-fixed',
            'steps': config['renderer']['steps'], 'eta': config['renderer']['eta'],
            'cfg_scale': config['renderer']['cfg_scale'], 'seed_init': stream_seed(seed, 'init'),
            'seed_steps': [stream_seed(seed, index, 'sde') for index in range(config['renderer']['steps'])],
            'metadata': {'attention_backend': 'fixed-attention', 'model_revision': 'fixed-revision',
                         'compiler_version': 'dag-agent-v3'}}


def trace_loader(path):
    value = recovery.read_json(path)
    value['seed_steps'] = tuple(value['seed_steps'])
    return SimpleNamespace(**value)


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.artifacts = ArtifactStore(self.root / 'artifacts')
        self.source_id = self.artifacts.put_image(Image.new('RGB', (512, 512), 'blue')).artifact_id
        self.task = PublicTask(task_id='train:1', family_id='family:1', split_id='train', task_mode='edit',
            user_request=UserRequest(text='Make the cup green.', source_image_id=self.source_id),
            check_spec=CheckSpec(extractor_version='fixture', atoms=(PublicCheckAtom(check_id='check:1', predicate='The cup is green.'),)))
        self.config = {'renderer': {'steps': 50, 'eta': .15, 'cfg_scale': 4., 'attention_backend': 'fixed-attention'},
                       'models': {'renderer_revision': 'fixed-revision'}}

    def bridge(self, fresh, source):
        return recovery.RecoveryRenderBridge(fresh, source, self.config, self.root / 'audit.jsonl',
            trace_loader, stream_seed, RenderResult)

    def execute_chain(self, bridge):
        compiler = Compiler()
        environment = EpisodeEnvironment(self.task, self.artifacts, compiler, bridge, Observer(),
            episode_id='public_rule:1', versions={'compiler': 'dag-agent-v3'})
        initial_revision = environment.state.manifest.revision_id
        def step(action, seed=0):
            result = environment.step(action, seed)
            self.assertIsNone(result.error)
            return result
        step({'kind': 'PLAN', 'plan': PlanSpec(global_text=self.task.user_request.text).model_dump(mode='json')})
        first_seed = stream_seed(11, self.task.task_id, 'controlled_initial')
        first = step({'kind': 'RENDER', 'control_set_id': environment.state.manifest.active_control_set_id,
            'expected_revision': environment.state.manifest.revision_id}, first_seed)
        step({'kind': 'SELECT', 'image_id': self.source_id})
        images = [first.render.image_id]
        for text, seed in [('Keep the input image unchanged.', first_seed),
                           (self.task.user_request.text, stream_seed(11, self.task.task_id, 'controlled_recovery'))]:
            state = environment.state
            step({'kind': 'DRAFT_REPAIR', 'source_image_id': state.manifest.selected_image_id,
                'evidence_ids': [key for key, value in state.observations.items() if value.image_id == state.manifest.selected_image_id][-4:],
                'suspect_node_ids': ['plan'], 'patch_ops': [{'target_logical_node_id': 'plan',
                    'expected_old_version_id': state.manifest.bindings['plan'], 'op': 'replace',
                    'field_path': ['global_text'], 'typed_value_or_artifact_id': text}],
                'instruction': text, 'scope': 'global'})
            candidate = next(iter(environment.state.candidates.values()))
            result = step({'kind': 'APPLY', 'candidate_id': candidate.candidate_id,
                           'expected_revision': environment.state.manifest.revision_id}, seed)
            images.append(result.render.image_id)
        return initial_revision, images

    def test_actual_environment_chain_reuses_three_exact_results_despite_new_revisions(self):
        old = self.root / 'old'
        first = FixtureBridge(old / 'renders', self.artifacts, self.config)
        revision, images = self.execute_chain(first)
        before = {str(path.relative_to(old)): path.read_bytes() for path in old.rglob('*') if path.is_file()}
        fresh = FixtureBridge(self.root / 'new/renders', self.artifacts, self.config)
        new_revision, new_images = self.execute_chain(self.bridge(fresh, old))
        self.assertNotEqual(revision, new_revision)
        self.assertEqual(new_images, images)
        self.assertEqual(len(first.fresh_calls), 3)
        self.assertEqual(fresh.fresh_calls, [])
        self.assertEqual(fresh.decode_calls, [])
        self.assertEqual([row['mode'] for row in recovery.rows(self.root / 'audit.jsonl')], ['exact_saved_result'] * 3)
        self.assertEqual(before, {str(path.relative_to(old)): path.read_bytes() for path in old.rglob('*') if path.is_file()})

    def test_trace_only_reuses_saved_trajectory_and_only_decodes(self):
        old = self.root / 'old'
        bundle = SimpleNamespace(bundle_id='bundle:exact', compiler_version='dag-agent-v3')
        first = FixtureBridge(old / 'renders', self.artifacts, self.config)
        expected = first.render(bundle, 19, 'render:exact')
        (old / 'renders/render_exact/result.json').unlink()
        fresh = FixtureBridge(self.root / 'new/renders', self.artifacts, self.config)
        actual = self.bridge(fresh, old).render(bundle, 19, 'render:exact')
        self.assertEqual(actual.image_id, expected.image_id)
        self.assertEqual(fresh.fresh_calls, [])
        self.assertEqual(fresh.decode_calls, ['render:exact'])
        self.assertEqual(recovery.rows(self.root / 'audit.jsonl')[0]['mode'], 'exact_saved_trace_decode')

    def test_wrong_seed_sampler_and_model_fail_without_rendering(self):
        old = self.root / 'old'
        bundle = SimpleNamespace(bundle_id='bundle:exact', compiler_version='dag-agent-v3')
        first = FixtureBridge(old / 'renders', self.artifacts, self.config)
        first.render(bundle, 19, 'render:exact')
        trace_path = old / 'renders/render_exact/trace.pt'
        original = recovery.read_json(trace_path)
        for field, value in [('seed_init', 0), ('eta', .3), ('bundle_id', 'bundle:wrong'), ('steps', 28)]:
            recovery.atomic_json(trace_path, original | {field: value})
            fresh = FixtureBridge(self.root / ('new-' + field) / 'renders', self.artifacts, self.config)
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'identity mismatch'):
                self.bridge(fresh, old).render(bundle, 19, 'render:exact')
            self.assertEqual(fresh.fresh_calls, [])
        changed = original | {'metadata': original['metadata'] | {'model_revision': 'other'}}
        recovery.atomic_json(trace_path, changed)
        fresh = FixtureBridge(self.root / 'new-model/renders', self.artifacts, self.config)
        with self.assertRaisesRegex(ValueError, 'metadata mismatch'):
            self.bridge(fresh, old).render(bundle, 19, 'render:exact')
        self.assertEqual(fresh.fresh_calls, [])

    def prepare_fixture(self):
        old = self.root / 'old'
        old.mkdir()
        tasks = [self.task.model_copy(update={'task_id': f'train:{index}', 'family_id': f'family:{index}'}).model_dump(mode='json') for index in range(3)]
        original_tasks = self.root / 'all-tasks.jsonl'
        write_rows(original_tasks, tasks)
        write_rows(old / 'tasks.jsonl', tasks)
        sampler = {'status': 'accepted', 'sampler_id': 'sde50_eta015', 'steps': 50, 'eta': .15, 'cfg_scale': 4.}
        inputs = {'original_tasks': str(original_tasks)}
        for key in ('selection', 'private_ready', 'sampler_selection'):
            path = self.root / (key + '.json')
            recovery.atomic_json(path, sampler if key == 'sampler_selection' else {'frozen': key})
            inputs[key] = str(path)
        recovery.atomic_json(old / 'sampler_selection.json', sampler)
        config = json.loads(json.dumps(self.config))
        config['data'] = {'private_tasks': str(self.root / 'private.jsonl')}
        config['judge'] = {'calibration': str(self.root / 'calibration.json')}
        config['run_dir'] = str(old)
        for path in (config['data']['private_tasks'], config['judge']['calibration']):
            recovery.atomic_json(path, {'fixed': True})
        registration = {'selected_tasks': [task['task_id'] for task in tasks], 'renderer_version': 'renderer-fixed'}
        for role in ('actor', 'renderer'):
            policy = self.root / (role + '.pt')
            policy.write_bytes(b'fixture-checkpoint')
            config['models'][role + '_policy'] = str(policy)
            registration[role + '_policy'] = str(policy)
            registration[role + '_weight_identity'] = recovery.file_identity(policy)
        for name, path in {'task_manifest_sha256': original_tasks, 'training_selection_sha256': inputs['selection'],
                           'private_ready_sha256': inputs['private_ready'], 'sampler_selection_sha256': inputs['sampler_selection'],
                           'private_manifest_sha256': config['data']['private_tasks'], 'calibration_sha256': config['judge']['calibration']}.items():
            registration[name] = recovery.digest(path)
        recovery.atomic_json(old / 'registration.json', registration)
        (old / 'collection.yaml').write_text(yaml.safe_dump(config))
        completed = old / 'tasks/0000'
        completed.mkdir(parents=True)
        snapshot = completed / 'snapshot.pt'
        snapshot.write_bytes(b'fixture-snapshot')
        record = {'record_id': 'unique:0', 'family_id': 'family:0', 'snapshot_path': str(snapshot)}
        write_rows(completed / 'actor_records.jsonl', [record])
        recovery.atomic_json(completed / 'state_bundle.json', {'task_id': 'train:0', 'family_id': 'family:0', 'verified_record_ids': ['unique:0']})
        recovery.atomic_json(completed / 'completed.json', {'task_id': 'train:0', 'family_id': 'family:0', 'counts': {'failed_tasks': 0}})
        event = {'name': 'plan', 'public_before_state': {'task': tasks[1]}, 'public_after_state': {'task': tasks[1]}}
        write_rows(old / 'tasks/0001/events.jsonl', [event])
        write_rows(old / 'tasks/0001/actor_records.jsonl', [{'record_id': 'old-partial-do-not-count'}])
        return old, inputs

    def test_preparation_preserves_old_attempt_and_imports_completed_family_once(self):
        old, inputs = self.prepare_fixture()
        before = {str(path.relative_to(old)): path.read_bytes() for path in old.rglob('*') if path.is_file()}
        output = self.root / 'recovery'
        plan = recovery.prepare(old, output, inputs, True)
        self.assertEqual(plan['completed_indices'], [0])
        self.assertEqual(plan['interrupted_index'], 1)
        self.assertEqual(plan['unstarted_indices'], [2])
        self.assertEqual(plan['completed_records_retained'], 1)
        self.assertFalse(plan['old_partial_actions_counted'])
        self.assertTrue((output / 'tasks/0000').is_symlink())
        self.assertFalse((output / 'tasks/0001').exists())
        self.assertEqual(recovery.rows(output / 'tasks/0000/actor_records.jsonl')[0]['record_id'], 'unique:0')
        self.assertEqual(before, {str(path.relative_to(old)): path.read_bytes() for path in old.rglob('*') if path.is_file()})

    def test_changed_identity_and_active_source_are_rejected_before_new_attempt(self):
        old, inputs = self.prepare_fixture()
        with self.assertRaisesRegex(ValueError, 'source-stopped'):
            recovery.prepare(old, self.root / 'new', inputs, False)
        Path(inputs['sampler_selection']).write_text('{}')
        with self.assertRaisesRegex(ValueError, 'frozen input changed'):
            recovery.prepare(old, self.root / 'new', inputs, True)
        self.assertFalse((self.root / 'new').exists())
        plan = recovery.inspect_source(old)
        with (old / 'tasks/0001/events.jsonl').open('a') as stream:
            stream.write('{"unfinished":')
        with self.assertRaisesRegex(RuntimeError, 'source attempt changed'):
            recovery.check_guards(plan['source_guards'])
        self.assertTrue(recovery.inspect_source(old)['torn_final_event_preserved'])

    def test_changed_prepared_tasks_are_rejected_before_model_imports(self):
        old, inputs = self.prepare_fixture()
        output = self.root / 'recovery'
        recovery.prepare(old, output, inputs, True)
        (output / 'tasks.jsonl').write_text('{}\n')
        with self.assertRaisesRegex(ValueError, 'prepared recovery input changed'):
            recovery.run(output, True)


if __name__ == '__main__':
    unittest.main()
