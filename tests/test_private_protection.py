import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

from agentic_image.checks.private import PrivateEvaluator
from agentic_image.checks.evidence import StructuredChecks
from agentic_image.checks.reward import AtomScore, preservation_score
from agentic_image.core.artifacts import ArtifactStore
from agentic_image.core.geometry import letterbox, make_frame, transform_mask
from agentic_image.core.types import PrivateEvaluation


class PrivateProtectionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.store = ArtifactStore(self.directory.name)
        self.source = np.full((4, 8, 3), 100, dtype=np.uint8)
        self.source_id = self.store.put_image(self.source).artifact_id
        frame = make_frame((4, 8), (8, 8), 'test')
        self.output_id = self.store.put_image(letterbox(self.source, frame)).artifact_id
        self.task = SimpleNamespace(task_id='test', user_request=SimpleNamespace(source_image_id=self.source_id, text='Edit the left object'), canvas=SimpleNamespace(height=8, width=8), limits=SimpleNamespace(max_renders=3, max_inspects=2, max_actions=10))
        self.judge = SimpleNamespace(check=lambda **kwargs: AtomScore(kwargs['atom_id'], .95, True, .9))
        self.tools = SimpleNamespace(call=lambda *args, **kwargs: {})

    def evaluator(self, mask=None, protection=None):
        evaluation = PrivateEvaluation(private_task_key='test', eval_atoms=({'eval_id':'eval:0','raw_span':'Edit','predicate':'The object is edited.','threshold':.8},), original_protected_mask_id=mask, annotation_provenance={'protection':protection or {}}, evaluator_version='test')
        return PrivateEvaluator(self.judge, self.store, {'test':evaluation}, self.tools)

    def evaluate(self, evaluator):
        return evaluator.evaluate(self.task, self.output_id, renders=1, inspects=0, actions=2, invalids=0)

    def test_missing_local_protection_is_not_success(self):
        with self.assertRaisesRegex(ValueError, 'source-grounded protection'):
            self.evaluate(self.evaluator())

    def test_explicit_global_inapplicability_is_not_numeric_perfection(self):
        protection = {'status':'not_applicable','scope':'global_edit','source_image_id':self.source_id,'reason':'Whole-image style changes every pixel.','review_evidence':{'reviewer':'test','observation':'Global request'}}
        result = self.evaluate(self.evaluator(protection=protection))
        self.assertIsNone(result.preservation)
        self.assertEqual(result.preservation_status, 'not_applicable')

    def test_native_source_mask_is_registered_to_canvas(self):
        mask = np.zeros((4, 8), dtype=np.float32)
        mask[:, 4:] = 1
        mask_id = self.store.put_array(mask).artifact_id
        result = self.evaluate(self.evaluator(mask_id))
        self.assertAlmostEqual(result.preservation, 1.)
        self.assertEqual(result.preservation_status, 'measured')
        self.assertEqual(result.atoms[0].threshold, .9)

    def test_delivered_original_uses_registered_source_only_for_preservation(self):
        frame = make_frame((4, 8), (8, 8), 'private-evaluation:test')
        native_mask = np.zeros((4, 8), dtype=np.float32)
        native_mask[:, 4:] = 1
        mask_id = self.store.put_array(
            transform_mask(native_mask, frame), frame=frame,
        ).artifact_id
        judge_calls = []
        feature_calls = []

        def check(**arguments):
            judge_calls.append(arguments)
            return AtomScore(arguments['atom_id'], .2, True, .9)

        def features(name, **arguments):
            self.assertEqual(name, 'preservation')
            feature_calls.append(arguments)
            return {}

        self.judge.check = check
        self.tools.call = features
        evaluator = self.evaluator(mask_id)
        original_ref = self.store.get_ref(self.source_id)
        for renders in (0, 1):
            with self.subTest(renders=renders):
                result = evaluator.evaluate(
                    self.task, self.source_id, renders=renders,
                    inspects=0, actions=2, invalids=0,
                )
                self.assertFalse(result.success)
                self.assertEqual(result.atoms[0].value, .2)
                self.assertEqual(result.atoms[0].threshold, .9)
                self.assertAlmostEqual(result.preservation, 1.)
                self.assertAlmostEqual(result.cost, (renders + .02) / 3.3)
                call = feature_calls[-1]
                self.assertEqual(call['source_image_id'], call['image_id'])
                self.assertNotEqual(call['image_id'], self.source_id)
                aligned = self.store.get_ref(call['image_id'])
                self.assertEqual(aligned.frame, frame)
                self.assertEqual(aligned.metadata['original_image_id'], self.source_id)
                self.assertEqual(aligned.metadata['visibility'], 'private')
                np.testing.assert_array_equal(
                    self.store.get_image(call['image_id']), letterbox(self.source, frame),
                )
                np.testing.assert_array_equal(
                    self.store.get_array(call['mask_id']), transform_mask(native_mask, frame),
                )
        self.assertTrue(all(
            [image.size for image in call['images']] == [(8, 4), (8, 4)]
            for call in judge_calls
        ))
        self.assertEqual(self.store.get_ref(self.source_id), original_ref)
        self.assertEqual(self.task.user_request.source_image_id, self.source_id)
        self.assertEqual(self.store.get_image(self.source_id).size, (8, 4))

    def test_other_native_output_is_not_resized_even_with_identical_pixels(self):
        mask_id = self.store.put_array(np.ones((4, 8), dtype=np.float32)).artifact_id
        candidate = self.store.put_image(self.source, producer_id='different-output').artifact_id
        self.assertNotEqual(candidate, self.source_id)
        with self.assertRaisesRegex(ValueError, 'output differs from the registered canvas'):
            self.evaluator(mask_id).evaluate(
                self.task, candidate, renders=0, inspects=0, actions=1, invalids=0,
            )

    def test_original_finish_at_observed_failure_geometry_and_budget(self):
        source = np.full((768, 1344, 3), 100, dtype=np.uint8)
        source_id = self.store.put_image(source).artifact_id
        self.task.user_request.source_image_id = source_id
        self.task.canvas.height = self.task.canvas.width = 512
        frame = make_frame((768, 1344), (512, 512), 'private-evaluation:test')
        native_mask = np.zeros((768, 1344), dtype=np.float32)
        native_mask[:, 672:] = 1
        mask_id = self.store.put_array(transform_mask(native_mask, frame), frame=frame).artifact_id
        self.judge.check = lambda **args: AtomScore(args['atom_id'], .2, True, .9)
        result = self.evaluator(mask_id).evaluate(
            self.task, source_id, renders=0, inspects=0, actions=3, invalids=1,
        )
        self.assertFalse(result.success)
        self.assertEqual(result.terminal_reason, 'finished')
        self.assertAlmostEqual(result.preservation, 1.)
        self.assertAlmostEqual(result.cost, .03 / 3.3)
        self.assertAlmostEqual(result.reward, .3 * .2 - .1 * (.03 / 3.3) - .02)
        self.assertEqual(self.store.get_ref(source_id).shape, (768, 1344, 3))
        self.assertIsNone(self.store.get_ref(source_id).frame)

    def test_original_delivery_does_not_bypass_mask_frame_validation(self):
        frame = make_frame((4, 8), (8, 8), 'wrong-source-registration')
        mask_id = self.store.put_array(np.ones((8, 8), dtype=np.float32), frame=frame).artifact_id
        with self.assertRaisesRegex(ValueError, 'different source-to-canvas registration'):
            self.evaluator(mask_id).evaluate(
                self.task, self.source_id, renders=0, inspects=0, actions=1, invalids=0,
            )

    def test_empty_or_missing_mask_never_scores_one(self):
        for mask in (None, np.zeros((4, 8), dtype=np.float32)):
            with self.assertRaises(ValueError):
                preservation_score(self.source, self.source, mask)
        mask_id = self.store.put_array(np.zeros((4, 8), dtype=np.float32)).artifact_id
        with self.assertRaisesRegex(ValueError, 'no valid source pixels'):
            self.evaluate(self.evaluator(mask_id))

    def test_unconfirmed_route_remains_unknown_without_judge_call(self):
        evaluator = self.evaluator()
        evaluator.unconfirmed_route_policy = 'unknown'
        self.judge.thresholds = {}
        self.judge.check = lambda **kwargs: self.fail('Unconfirmed route invoked the judge')
        score = evaluator.score_image(self.task, self.output_id)[0]
        self.assertFalse(score.known)
        self.assertFalse(score.exact_pass)
        self.assertEqual(score.evidence, ('not_evaluated_calibration:semantic',))

    def test_source_ambiguity_cannot_be_overruled_by_judge(self):
        evaluator = self.evaluator()
        evaluation = evaluator.evaluations['test']
        evaluator.evaluations['test'] = evaluation.model_copy(update={'annotation_provenance':{'forced_unknown_eval_ids':{'eval:0':'Two possible referents'}}})
        self.judge.check = lambda **kwargs: self.fail('Ambiguous source invoked the judge')
        self.assertFalse(evaluator.score_image(self.task, self.output_id)[0].known)

    def test_phrase_ocr_preserves_literal_words_and_case(self):
        text = ['LOGO ADAM WEST BURT WARD']
        tools = SimpleNamespace(call=lambda *args, **kwargs: {'known':True,'text':text[0]})
        checks = StructuredChecks(tools)
        expected = {'kind':'text','text':'ADAM WEST','match_mode':'contains_exact_phrase'}
        self.assertTrue(checks.evaluate('text','image','text',expected).passed)
        text[0] = 'BADAM WESTERN'
        self.assertFalse(checks.evaluate('text','image','text',expected).passed)
        text[0] = 'Adam West'
        self.assertFalse(checks.evaluate('text','image','text',expected).passed)

    def test_relative_count_uses_original_source_and_retains_unknown(self):
        detections = {'source': {'known': True, 'detections': [{}, {}]}, 'image': {'known': True, 'detections': [{}]}}
        tools = SimpleNamespace(call=lambda *args, **kwargs: detections[kwargs['image_id']])
        checks = StructuredChecks(tools)
        expected = {'kind': 'count', 'delta': -1, 'object': 'jars'}
        self.assertTrue(checks.evaluate('count', 'image', 'count', expected, 'source').passed)
        detections['source'] = {'known': False, 'detections': []}
        score = checks.evaluate('count', 'image', 'count', expected, 'source')
        self.assertFalse(score.known)
        self.assertFalse(score.exact_pass)


if __name__ == '__main__':
    unittest.main()
