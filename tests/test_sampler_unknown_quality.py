import copy
import itertools
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from agentic_image.evaluation.statistics import paired_bootstrap
from agentic_image.runtime.sampler import complete_comparison_rows, sampler_decision_with_protection, severe_review_status


def rows_and_reviews():
    rows, reviews = [], []
    for index in range(64):
        for seed in (101, 211):
            for method in ('ode50', 'sde50'):
                image_id = f'image:{index}:{seed}:{method}'
                row = {'method':method,'training_seed':11,'family_id':f'family:{index}',
                    'task_id':f'task:{index}','eval_seed':seed,'image_id':image_id,'image_value_id':image_id+':value',
                    'source_image_id':f'source:{index}','bundle_id':f'bundle:{index}',
                    'renderer_policy_identity':{'path':'test-fixture'},'status':'scored','calibrated_evaluation':True,
                    'success':1.,'severe_defect':0.,'severe_defect_status':'reviewed','protection':.9,
                    'preservation_status':'measured','seconds':50.}
                review = {'review_id':image_id+':review','image_id':image_id,'candidate_image_value_id':row['image_value_id'],
                    'known':True,'severe_defect':False,'review_type':'ai','human_reviewed':False,
                    'reviewed_image_paths':['synthetic-unit-test-fixture'],
                    'review_evidence':{'actual_visual_inspection':True,'reviewer':'synthetic-unit-test-reviewer',
                        'observation':'Synthetic fixture for gate boundary testing; not a real visual review.'}}
                row['severe_defect_review'] = review
                rows.append(row)
                reviews.append(review)
    return rows, reviews


def mark_unknown(row):
    row.update(severe_defect=None, severe_defect_status='reviewed_unknown')
    row['severe_defect_review'].update(known=False, severe_defect=None)


class SamplerUnknownQualityTests(unittest.TestCase):
    def test_reviewed_unknown_can_pass_only_with_worst_case_and_keeps_null(self):
        rows, _ = rows_and_reviews()
        mark_unknown(rows[1])
        before = copy.deepcopy(rows)
        result = sampler_decision_with_protection(rows, 'sde50', 'ode50', 1000)
        defects = result['severe_defects']
        self.assertTrue(result['accepted'])
        self.assertEqual(rows, before)
        self.assertEqual(defects['reviewed_unknown_count'], 1)
        self.assertEqual(defects['known_label_count_by_method'], {'sde50':127,'ode50':128})
        self.assertEqual(defects['worst_case_difference_upper_bound'], 1/128)
        self.assertLessEqual(defects['worst_case_confidence_upper_bound'], .05)
        self.assertEqual(defects['observations'], 128)
        self.assertEqual(defects['families'], 64)
        self.assertIsNone(defects['reviewed_unknown_images'][0].get('label'))

    def test_too_many_candidate_unknowns_reject_instead_of_dropping(self):
        rows, _ = rows_and_reviews()
        for row in rows:
            if row['method'] == 'sde50' and int(row['family_id'].split(':')[1]) < 8:
                mark_unknown(row)
        result = sampler_decision_with_protection(rows, 'sde50', 'ode50', 1000)
        self.assertFalse(result['accepted'])
        self.assertEqual(result['severe_defects']['reviewed_unknown_count'], 16)
        self.assertGreater(result['severe_defects']['worst_case_confidence_upper_bound'], .05)
        self.assertEqual(result['success']['observations'], 128)

    def test_unknown_reference_uses_zero_and_changes_role_when_reversed(self):
        rows, _ = rows_and_reviews()
        mark_unknown(rows[0])
        forward = sampler_decision_with_protection(rows, 'sde50', 'ode50', 1000)['severe_defects']
        reverse = sampler_decision_with_protection(rows, 'ode50', 'sde50', 1000)['severe_defects']
        self.assertEqual(forward['worst_case_difference_upper_bound'], 0.)
        self.assertEqual(forward['difference_identification_interval'], [-1/128, 0.])
        self.assertEqual(reverse['worst_case_difference_upper_bound'], 1/128)

    def test_upper_bootstrap_dominates_every_binary_completion(self):
        rows, _ = rows_and_reviews()
        mark_unknown(rows[0])
        mark_unknown(rows[5])
        bound = sampler_decision_with_protection(rows, 'sde50', 'ode50', 1000)['severe_defects']
        for values in itertools.product((0., 1.), repeat=2):
            completed = copy.deepcopy(rows)
            completed[0]['severe_defect'], completed[5]['severe_defect'] = values
            actual = paired_bootstrap(completed, 'sde50', 'ode50', 'severe_defect', 1000)
            self.assertLessEqual(actual['difference'], bound['worst_case_difference_upper_bound'])
            self.assertLessEqual(actual['confidence_interval'][1], bound['worst_case_confidence_upper_bound'])
            self.assertGreaterEqual(actual['confidence_interval'][0], bound['robust_confidence_interval'][0])

    def test_missing_inspection_or_identity_does_not_become_reviewed_unknown(self):
        rows, _ = rows_and_reviews()
        mark_unknown(rows[1])
        valid = rows[1]
        self.assertEqual(severe_review_status(valid['severe_defect_review'], valid), 'reviewed_unknown')
        mutations = [lambda r:r['review_evidence'].update(actual_visual_inspection=False),
            lambda r:r.update(review_evidence=None),lambda r:r.update(reviewed_image_paths=[]),
            lambda r:r.update(candidate_image_value_id='different'),lambda r:r.update(image_id='different'),
            lambda r:r.update(known=True),lambda r:r.update(severe_defect=False),
            lambda r:r['review_evidence'].update(observation='')]
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                changed = copy.deepcopy(rows)
                mutate(changed[1]['severe_defect_review'])
                self.assertIsNone(severe_review_status(changed[1]['severe_defect_review'], changed[1]))
                with self.assertRaises(ValueError):
                    sampler_decision_with_protection(changed, 'sde50', 'ode50', 100)
        changed = copy.deepcopy(rows)
        changed[1]['severe_defect_status'] = 'unknown'
        with self.assertRaises(ValueError):
            sampler_decision_with_protection(changed, 'sde50', 'ode50', 100)

    def test_complete_rows_distinguishes_reviewed_unknown_and_missing_pixels(self):
        rows, reviews = rows_and_reviews()
        mark_unknown(rows[1])
        tasks = [SimpleNamespace(task_id=f'task:{i}',family_id=f'family:{i}') for i in range(64)]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for row in rows:
                index = int(row['task_id'].split(':')[1])
                image_path = root/f'{index}_{row["eval_seed"]}_{row["method"]}.png'
                image_path.write_bytes(b'synthetic unit test fixture')
                row['image'] = str(image_path)
                for stage in ('images','scores'):
                    folder = root/stage/row['method']
                    folder.mkdir(parents=True,exist_ok=True)
                    (folder/f'{index:03d}_{row["eval_seed"]}.json').write_text(json.dumps(row))
            review_path = root/'reviews.jsonl'
            original = ''.join(json.dumps(r)+'\n' for r in reviews)
            review_path.write_text(original)
            completed, report = complete_comparison_rows(tasks,root,review_path,('ode50','sde50'))
            self.assertEqual(report['status'],'ready_for_decision')
            self.assertEqual(report['reviewed_unknown_defect_count'],1)
            self.assertEqual(completed[1]['severe_defect_status'],'reviewed_unknown')
            self.assertIsNone(completed[1]['severe_defect'])
            self.assertFalse(completed[1]['severe_defect_review']['known'])
            self.assertEqual(review_path.read_text(),original)
            reviews[1]['review_evidence']['actual_visual_inspection'] = False
            review_path.write_text(''.join(json.dumps(r)+'\n' for r in reviews))
            _, missing = complete_comparison_rows(tasks,root,review_path,('ode50','sde50'))
            self.assertEqual(missing['status'],'incomplete')
            self.assertIn('severe_defect_review',[r['missing'] for r in missing['missing']])
            Path(rows[0]['image']).unlink()
            _, missing = complete_comparison_rows(tasks,root,review_path,('ode50','sde50'))
            self.assertIn('render_pixels',[r['missing'] for r in missing['missing']])


if __name__ == '__main__':
    unittest.main()
