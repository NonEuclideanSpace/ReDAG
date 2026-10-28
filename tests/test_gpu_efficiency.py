import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('gpu_efficiency_entry', ROOT / 'scripts/gpu_efficiency.py')
entry = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(entry)


def sample(timestamp, utilization):
    return {'time': timestamp, 'gpus': {'0': {'utilization': utilization}}}


class GpuEfficiencyTests(unittest.TestCase):
    def test_normal_five_second_samples_preserve_exact_metrics(self):
        samples = [sample(0, 0), sample(5, 80), sample(10, 100), sample(15, 40)]
        report = entry.efficiency(samples, now=20, windows=(20,))
        row = report['gpus']['0']['20']
        self.assertEqual(row['mean_utilization'], 55)
        self.assertEqual(row['high_load_time_percent'], 50)
        self.assertEqual(row['idle_time_percent'], 25)
        self.assertEqual(row['observed_seconds'], 20)
        self.assertEqual(row['coverage_percent'], 100)
        self.assertEqual(row['unknown_seconds'], 0)
        self.assertEqual(row['unknown_time_percent'], 0)
        self.assertEqual(report['maximum_sample_gap_seconds'], 15)

    def test_seven_minute_gap_has_no_high_or_idle_attribution(self):
        for before_gap in (0, 100):
            with self.subTest(before_gap=before_gap):
                samples = [sample(0, before_gap), sample(420, 20),
                           sample(425, 40), sample(430, 60)]
                row = entry.efficiency(samples, now=435, windows=(435,))['gpus']['0']['435']
                self.assertEqual(row['mean_utilization'], 40)
                self.assertEqual(row['high_load_time_percent'], 0)
                self.assertEqual(row['idle_time_percent'], 0)
                self.assertEqual(row['observed_seconds'], 15)
                self.assertAlmostEqual(row['coverage_percent'], 100 * 15 / 435)
                self.assertEqual(row['unknown_seconds'], 420)
                self.assertAlmostEqual(row['unknown_time_percent'], 100 - row['coverage_percent'])

    def test_stale_trailing_samples_are_unknown_but_exact_limit_is_observed(self):
        for timestamp, utilization in ((0, 100), (84, 0)):
            with self.subTest(timestamp=timestamp):
                row = entry.efficiency([sample(timestamp, utilization)], now=100,
                                       windows=(100,))['gpus']['0']['100']
                self.assertIsNone(row['mean_utilization'])
                self.assertIsNone(row['high_load_time_percent'])
                self.assertIsNone(row['idle_time_percent'])
                self.assertEqual(row['observed_seconds'], 0)
                self.assertEqual(row['coverage_percent'], 0)
                self.assertEqual(row['unknown_seconds'], 100)
                self.assertEqual(row['unknown_time_percent'], 100)
        row = entry.efficiency([sample(85, 80)], now=100, windows=(100,))['gpus']['0']['100']
        self.assertEqual(row['mean_utilization'], 80)
        self.assertEqual(row['high_load_time_percent'], 100)
        self.assertEqual(row['idle_time_percent'], 0)
        self.assertEqual(row['observed_seconds'], 15)
        self.assertEqual(row['coverage_percent'], 15)
        self.assertEqual(row['unknown_seconds'], 85)
        self.assertEqual(row['unknown_time_percent'], 85)


if __name__ == '__main__':
    unittest.main()
