import ast
from pathlib import Path
import subprocess
import types
import unittest
from unittest.mock import Mock


SOURCE = Path(__file__).resolve().parents[1] / 'scripts/download_rental_chunks.py'
FUNCTION = next(node for node in ast.parse(SOURCE.read_text()).body
    if isinstance(node, ast.FunctionDef) and node.name == 'ssh_for')


class ExistingMasterTests(unittest.TestCase):
    def context(self, live, checks, worker=None):
        run = Mock(side_effect=checks)
        namespace = {
            'controls': ['a', 'b', 'c'], 'control_offset': 1,
            'worker_connection': types.SimpleNamespace(**({'control': worker} if worker else {})),
            'Path': lambda name: types.SimpleNamespace(is_socket=lambda: name in live),
            'subprocess': types.SimpleNamespace(run=run, TimeoutExpired=subprocess.TimeoutExpired),
        }
        exec(compile(ast.Module(body=[FUNCTION], type_ignores=[]), '<actual ssh_for>', 'exec'), namespace)
        return namespace['ssh_for'], run

    def result(self, code=0):
        return types.SimpleNamespace(returncode=code, stderr='master unavailable')

    def test_stable_source_offset_is_used(self):
        select, run = self.context({'a', 'b', 'c'}, [self.result()])
        self.assertEqual(select(0)[2], 'b')
        self.assertIn('-O', run.call_args.args[0])
        self.assertIn('check', run.call_args.args[0])

    def test_retry_rotates_from_worker_binding(self):
        select, run = self.context({'a', 'b', 'c'}, [self.result()], 'a')
        self.assertEqual(select(0, 1)[2], 'b')

    def test_missing_and_dead_sockets_fall_back_only_to_live_master(self):
        select, run = self.context({'b', 'c'}, [self.result(255), self.result()], 'a')
        self.assertEqual(select(0)[2], 'c')
        self.assertEqual([call.args[0][2] for call in run.call_args_list], ['b', 'c'])
        self.assertTrue(all('-O' in call.args[0] and 'check' in call.args[0]
            for call in run.call_args_list))

    def test_all_missing_sockets_never_execute_ssh(self):
        select, run = self.context(set(), [], 'a')
        with self.assertRaisesRegex(RuntimeError, 'No existing transfer master'):
            select(0)
        run.assert_not_called()

    def test_local_check_timeout_is_bounded_and_falls_back(self):
        select, run = self.context({'a', 'b'},
            [subprocess.TimeoutExpired('ssh', 5), self.result()], 'a')
        self.assertEqual(select(0)[2], 'b')
        self.assertTrue(all(call.kwargs['timeout'] == 5 for call in run.call_args_list))


if __name__ == '__main__':
    unittest.main()
