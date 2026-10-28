import ast
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from PIL import Image

from agentic_image.checks.reward import AtomScore
from agentic_image.core.artifacts import ArtifactStore


SOURCE = Path(__file__).resolve().parents[1] / 'src/agentic_image/runtime/training.py'
FUNCTION = next(node for node in ast.parse(SOURCE.read_text()).body
    if isinstance(node, ast.FunctionDef) and node.name == 'evaluate_development')


class DevelopmentUnknownRoutesTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.store = ArtifactStore(self.root / 'artifacts')
        self.image_id = self.store.put_image(Image.new('RGB', (16, 16), 'blue')).artifact_id
        private = self.root / 'private.jsonl'
        private.write_text(json.dumps({'task_id': 'dev:1', 'evaluation': {
            'private_task_key': 'dev:1', 'evaluator_version': 'frozen-test-v1',
            'eval_atoms': [
                {'eval_id': 'count', 'raw_span': 'two cups', 'predicate': 'There are two cups.',
                 'checker_route': 'count', 'expected_value': {'kind': 'count', 'object': 'cup', 'count': 2}, 'threshold': .8},
                {'eval_id': 'color', 'raw_span': 'blue', 'predicate': 'The cups are blue.',
                 'checker_route': 'semantic', 'expected_value': 'blue', 'threshold': .8},
            ],
        }}) + '\n')
        self.calls = []
        def check(**kwargs):
            self.calls.append(kwargs['kind'])
            if kwargs['kind'] == 'count':
                raise ValueError('The count judge route has no confirmed calibration')
            return AtomScore(kwargs['atom_id'], .95, True, .5)
        self.components = SimpleNamespace(config={
            'data': {'development_tasks': 'fixed-development.jsonl', 'development_private': str(private)},
            'rl': {'development_tasks': 64}, 'judge': {'unconfirmed_route_policy': 'unknown'},
        }, judge=SimpleNamespace(check=check, require_calibration=True,
            thresholds={'semantic': {'confirmed': True, 'threshold': .5}, 'count': {'confirmed': False}}),
            artifacts=self.store, observer=SimpleNamespace(tools=SimpleNamespace(
                call=lambda *args, **kwargs: {'known': True, 'detections': [{}, {}]})))
        task = SimpleNamespace(task_id='dev:1', split_id='development',
            user_request=SimpleNamespace(source_image_id=None, text='Create two blue cups.'),
            limits=SimpleNamespace(max_renders=3, max_inspects=2, max_actions=10))
        state = SimpleNamespace(delivered_image_id=self.image_id, protocol_failure=False,
            budget=SimpleNamespace(renders=1, inspects=0, actions=2, invalid=0))
        class Collector:
            def __init__(self, components): pass
            def infer(self, actual_task, seed, directory):
                return SimpleNamespace(state=state)
        namespace = {'Path': Path, 'read_tasks': lambda path: [task], 'RolloutCollector': Collector,
            'atomic_json': lambda path, value: path.write_text(json.dumps(value))}
        exec(compile(ast.Module(body=[FUNCTION], type_ignores=[]), str(SOURCE), 'exec'), namespace)
        self.evaluate = namespace['evaluate_development']

    def test_configured_unknown_route_survives_actual_development_evaluation(self):
        result = self.evaluate(self.components, 32, self.root / 'development')
        self.assertEqual(self.calls, ['semantic'])
        self.assertEqual(len(result), 1)
        self.assertFalse(result[0]['success'])
        atoms = {atom['atom_id']: atom for atom in result[0]['atoms']}
        self.assertEqual(atoms['count']['value'], 0)
        self.assertFalse(atoms['count']['known'])
        self.assertEqual(atoms['count']['threshold'], .8)
        self.assertEqual(atoms['count']['evidence'], ('not_evaluated_calibration:count',))
        self.assertTrue(atoms['color']['known'])
        self.assertEqual(atoms['color']['threshold'], .8)
        self.assertEqual(atoms['color']['value'], .95)
        saved = json.loads((self.root / 'development/32/results.json').read_text())
        self.assertEqual(len(saved[0]['atoms']), 2)

    def test_legacy_default_still_rejects_unconfirmed_route(self):
        self.components.config['judge'].pop('unconfirmed_route_policy')
        with self.assertRaisesRegex(ValueError, 'no confirmed calibration'):
            self.evaluate(self.components, 32, self.root / 'development')
        self.assertEqual(self.calls, ['count'])


if __name__ == '__main__':
    unittest.main()
