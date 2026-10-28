import ast
import json
from pathlib import Path
import threading
from types import SimpleNamespace
import unittest

import numpy as np

from agentic_image.checks.private import PrivateEvaluator
from agentic_image.checks.reward import AtomScore, terminal_reward


SOURCE = Path(__file__).resolve().parents[1] / 'src/agentic_image/checks/judge.py'
TREE = ast.parse(SOURCE.read_text())
CLASS = next(node for node in TREE.body if isinstance(node, ast.ClassDef) and node.name == 'FrozenJudge')
TEMPLATES = ast.literal_eval(next(node.value for node in TREE.body
    if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == 'TEMPLATES'
        for target in node.targets)))


def actual_method(name, **namespace):
    node = next(node for node in CLASS.body if isinstance(node, ast.FunctionDef) and node.name == name)
    node.decorator_list = []
    values = {'AtomScore': AtomScore, 'TEMPLATES': TEMPLATES, **namespace}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), 'exec'), values)
    return values[name]


class ProbabilityAuditTests(unittest.TestCase):
    def judge(self, probabilities):
        pending = iter(probabilities)
        calls = []
        def probability(text, images, **kwargs):
            value = next(pending)
            calls.append((text, images))
            if kwargs.get('audit') is not None:
                kwargs['audit'].update(executed=True, probability=value,
                    image_grid_thw=[[1, 32, 32], [1, 24, 42]], input_tokens=520)
            return value
        judge = SimpleNamespace(config=SimpleNamespace(threshold=.8), thresholds={},
            require_calibration=False, answer_probability=probability)
        return judge, calls

    def run_check(self, probabilities, audit, **kwargs):
        judge, calls = self.judge(probabilities)
        result = actual_method('check')(judge, atom_id='a', request='Add a white keyboard',
            property_text='A white keyboard is present', images=['candidate', 'source'],
            roles='candidate, source', audit=audit, **kwargs)
        return result, calls

    def test_unexecuted_second_is_null_and_public_return_is_unchanged(self):
        audit = {}
        actual, calls = self.run_check([.95], audit)
        public, public_calls = self.run_check([.95], None)
        self.assertEqual(actual, public)
        self.assertEqual(calls, public_calls)
        self.assertEqual(len(calls), 1)
        self.assertEqual(actual.evidence, ())
        self.assertFalse(audit['second_required'])
        self.assertFalse(audit['second_executed'])
        self.assertIsNone(audit['second']['probability'])
        self.assertIsNone(audit['second']['image_grid_thw'])
        self.assertEqual(audit['decision_second_value'], .95)
        self.assertEqual(audit['first']['image_grid_thw'], [[1, 32, 32], [1, 24, 42]])

    def test_disagreement_keeps_raw_values_and_original_minimum(self):
        audit = {}
        actual, calls = self.run_check([.65, .25], audit)
        public, public_calls = self.run_check([.65, .25], None)
        self.assertEqual(actual, public)
        self.assertEqual(calls, public_calls)
        self.assertEqual(len(calls), 2)
        self.assertEqual(actual.value, .25)
        self.assertFalse(actual.known)
        self.assertEqual(audit['first']['probability'], .65)
        self.assertEqual(audit['second']['probability'], .25)
        self.assertEqual(audit['known_rejection_reasons'], ['template_disagreement'])

    def test_below_threshold_is_not_a_known_rejection(self):
        audit = {}
        actual, calls = self.run_check([.75, .77], audit)
        self.assertTrue(actual.known)
        self.assertFalse(actual.passed)
        self.assertEqual(actual.value, .75)
        self.assertEqual(audit['known_rejection_reasons'], [])
        self.assertEqual(len(calls), 2)

    def test_structured_and_semantic_rejection_rules_are_unchanged(self):
        cases = [
            ({'structured': AtomScore('a', .1, True)}, ['conflict', 'structured_semantic_disagreement']),
            ({'structured': AtomScore('a', .9, False)}, ['structured_unknown', 'structured_semantic_disagreement']),
            ({'semantic_sufficient': False}, ['semantic_insufficient']),
        ]
        for kwargs, reasons in cases:
            with self.subTest(kwargs=kwargs):
                audit = {}
                actual, calls = self.run_check([.95, .95], audit, **kwargs)
                public, public_calls = self.run_check([.95, .95], None, **kwargs)
                self.assertEqual(actual, public)
                self.assertEqual(calls, public_calls)
                self.assertEqual(audit['known_rejection_reasons'], reasons)

    def test_legacy_answer_probability_double_needs_no_new_keyword(self):
        judge, calls = self.judge([.95])
        judge.answer_probability = lambda text, images: .95
        result = actual_method('check')(judge, atom_id='a', request='r', property_text='p',
            images=[], roles='candidate')
        self.assertEqual(result, AtomScore('a', .95, True, .8))

    def test_probability_captures_existing_grid_without_extra_forward(self):
        prepared, forwards = [], []
        grid_values = [[1, 32, 32], [1, 24, 42]]
        grid = SimpleNamespace()
        grid.detach = lambda: grid
        grid.cpu = lambda: grid
        grid.tolist = lambda: [list(row) for row in grid_values]
        def prepare(text, images):
            prepared.append((text, images))
            return {'input_ids': np.array([[1, 2, 3]]), 'attention_mask': np.ones((1, 3)),
                'image_grid_thw': grid, 'mm_token_type_ids': np.zeros((1, 3))}
        class Logits:
            def __init__(self, value): self.value = value
            def float(self): return self
            def log_softmax(self, dimension): return self
            def gather(self, dimension, labels): return self
            def sum(self): return self.value
        class Model:
            model = SimpleNamespace(rope_deltas='previous')
            def __call__(self, **kwargs):
                forwards.append(kwargs)
                return SimpleNamespace(logits=Logits(-2.0 if len(forwards) % 2 else -5.0))
        torch = SimpleNamespace(tensor=lambda value, **kwargs: np.asarray(value), long=np.int64,
            cat=lambda values, dim: np.concatenate(values, axis=dim), ones_like=np.ones_like,
            zeros_like=np.zeros_like,
            arange=lambda start, end, **kwargs: np.arange(start, end),
            sigmoid=lambda value: np.asarray(1 / (1 + np.exp(-value))), isfinite=np.isfinite)
        judge = SimpleNamespace(lock=threading.RLock(), _prepare=prepare, config=SimpleNamespace(device='cpu'),
            processor=SimpleNamespace(tokenizer=SimpleNamespace(convert_tokens_to_ids=lambda token: 151645,
                encode=lambda answer, **kwargs: [5 if answer == 'Yes' else 7])),
            model=Model(), forwards=0, queries=0)
        audit = {}
        method = actual_method('answer_probability', torch=torch)
        value = method(judge, 'unchanged prompt', ['candidate', 'source'], audit=audit)
        self.assertEqual(value, audit['probability'])
        self.assertEqual(audit['image_grid_thw'], grid_values)
        self.assertEqual(audit['input_tokens'], 3)
        self.assertEqual(judge.forwards, 2)
        self.assertEqual(judge.queries, 1)
        self.assertEqual(len(prepared), 1)
        self.assertTrue(all(call['use_cache'] is False for call in forwards))
        plain = method(judge, 'unchanged prompt', ['candidate', 'source'])
        self.assertEqual(value, plain)
        self.assertEqual(judge.forwards, 4)

    def test_private_evidence_roundtrips_and_legacy_check_signature_survives(self):
        def check(*, atom_id, request, property_text, images, roles, kind, structured):
            self.assertEqual(images, ['candidate pixels', 'source pixels'])
            return AtomScore(atom_id, .9, True, .5, evidence=('artifact:structured',))
        atom = SimpleNamespace(eval_id='a', predicate='white keyboard', expected_value='white keyboard',
            checker_route='semantic', threshold=.8)
        task = SimpleNamespace(task_id='task', user_request=SimpleNamespace(text='r', source_image_id='source'))
        judge = SimpleNamespace(check=check, thresholds={'semantic': {'confirmed': True}})
        artifacts = SimpleNamespace(get_image=lambda image_id: image_id + ' pixels')
        evaluator = PrivateEvaluator(judge, artifacts,
            {'task': SimpleNamespace(annotation_provenance={}, eval_atoms=[atom])})
        plain = evaluator.score_image(task, 'candidate')[0]
        def audited_check(*, audit, **kwargs):
            audit.update(schema='frozen_judge_probability_audit_v1', first={'executed': True,
                'probability': .9, 'image_grid_thw': [[1, 32, 32], [1, 24, 42]]},
                second={'executed': False, 'probability': None, 'image_grid_thw': None})
            return check(**kwargs)
        judge.supports_probability_audit = True
        judge.check = audited_check
        scored = evaluator.score_image(task, 'candidate')[0]
        self.assertEqual((plain.value, plain.known, plain.threshold, plain.exact_pass),
            (scored.value, scored.known, scored.threshold, scored.exact_pass))
        self.assertEqual(scored.evidence[0], 'artifact:structured')
        audit = json.loads(scored.evidence[1].removeprefix('judge_score_audit:'))
        self.assertEqual(audit['candidate_image_id'], 'candidate')
        self.assertEqual(audit['reference_image_id'], 'source')
        self.assertEqual(audit['effective_threshold'], .8)
        reward = terminal_reward([scored], renders=1, inspects=0, actions=2, invalids=0)
        self.assertEqual(json.loads(json.dumps(reward.to_dict()))['atoms'][0]['evidence'][1],
            scored.evidence[1])


if __name__ == '__main__':
    unittest.main()
