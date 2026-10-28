import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location(
    "result_pack_nested_scope", ROOT / "scripts/package_rental_results.py"
)
pack = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pack)


class ResultPackNestedScopeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "project"
        self.root.mkdir()
        self.agent = "runs/full-actor-rl/rl_agent"
        self.joint = "runs/full-actor-rl/rl_joint"
        for name in (
            self.agent + "/training_completed.json",
            self.agent + "/checkpoints/committed/actor.pt",
            self.joint + "/metrics.jsonl",
            self.joint + "/checkpoints/live/actor.pt",
            "data/sharedrenders/active/image.png",
        ):
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"actual fixture")

    def assert_rejected(self, selection):
        try:
            rows, errors = pack.inventory(self.root, [], only_groups=[selection])
        except ValueError:
            return
        self.assertTrue(errors)
        self.assertEqual(rows, {})

    def test_nested_scope_excludes_joint_and_shared_renders(self):
        rows, errors = pack.inventory(self.root, [], only_groups=[self.agent])
        self.assertEqual(errors, [])
        self.assertEqual(
            set(rows),
            {
                self.agent,
                self.agent + "/training_completed.json",
                self.agent + "/checkpoints",
                self.agent + "/checkpoints/committed",
                self.agent + "/checkpoints/committed/actor.pt",
            },
        )
        self.assertEqual({row["group"] for row in rows.values()}, {self.agent})

    def test_existing_top_level_scope_keeps_both_stages(self):
        group = "runs/full-actor-rl"
        rows, errors = pack.inventory(self.root, [], only_groups=[group])
        self.assertEqual(errors, [])
        self.assertIn(self.agent + "/training_completed.json", rows)
        self.assertIn(self.joint + "/metrics.jsonl", rows)
        self.assertTrue(all(path == group or path.startswith(group + "/") for path in rows))
        self.assertEqual({row["group"] for row in rows.values()}, {group})

    def test_default_inventory_preserves_all_groups(self):
        rows, errors = pack.inventory(self.root, [])
        self.assertEqual(errors, [])
        self.assertIn(self.agent + "/training_completed.json", rows)
        self.assertIn(self.joint + "/metrics.jsonl", rows)
        self.assertIn("data/sharedrenders/active/image.png", rows)

    def test_missing_nested_scope_is_an_explicit_error(self):
        group = "runs/full-actor-rl/not-present"
        rows, errors = pack.inventory(self.root, [], only_groups=[group])
        self.assertEqual(rows, {})
        self.assertTrue(errors)
        self.assertIn(group, str(errors))

    def test_unknown_group_is_an_explicit_error(self):
        rows, errors = pack.inventory(self.root, [], only_groups=["unrecognized"])
        self.assertEqual(rows, {})
        self.assertTrue(errors)
        self.assertIn("unrecognized", str(errors))

    def test_parent_traversal_and_absolute_paths_are_rejected(self):
        for group in (
            "runs/full-actor-rl/../../../outside",
            "runs/full-actor-rl/../full-actor-rl/rl_agent",
            str(self.root / self.agent),
        ):
            with self.subTest(group=group):
                self.assert_rejected(group)

    def test_symlink_ancestor_cannot_select_outside_files(self):
        outside = Path(self.temporary.name) / "outside"
        (outside / "stage").mkdir(parents=True)
        (outside / "stage" / "private.pt").write_bytes(b"must not be selected")
        (self.root / "runs" / "alias").symlink_to(outside, target_is_directory=True)
        self.assert_rejected("runs/alias/stage")

    def test_excluded_nested_scope_does_not_succeed_empty(self):
        group = self.root / self.agent
        rows, errors = pack.inventory(self.root, [group], only_groups=[self.agent])
        self.assertEqual(rows, {})
        self.assertTrue(errors)
        self.assertIn(self.agent, str(errors))


if __name__ == "__main__":
    unittest.main()
