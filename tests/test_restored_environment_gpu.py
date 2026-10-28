import contextlib
import ast
import importlib.util
import io
import json
from pathlib import Path, PurePosixPath
import tempfile
import unittest
from unittest.mock import patch, Mock
import os
import stat
from types import SimpleNamespace


SPEC = importlib.util.spec_from_file_location(
    "restored_environment_gpu",
    Path(__file__).resolve().parents[1] / "scripts/check_restored_environment_gpu.py",
)
restore = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(restore)

ACTOR_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
RENDERER_REVISION = "a3b4f4849157f664bdbc776fd7453c2783562f4d"


class RestoredEnvironmentGpuTests(unittest.TestCase):
    def test_native_runtime_environment_uses_extracted_prefix_and_keeps_devices(self):
        root = Path('/independent/root')
        base = {'LD_LIBRARY_PATH': '/host/driver', 'CUDA_VISIBLE_DEVICES': '0,1', 'PYTHONPATH': '/project/src'}
        env = restore.native_environment(root, base)
        prefix = '/independent/root/usr/local/miniconda3/envs/py312'
        self.assertEqual(env['PYTHONHOME'], prefix)
        self.assertEqual(env['RESTORED_NATIVE_PREFIX'], prefix)
        self.assertEqual(env['PATH'], prefix + '/bin:/usr/bin:/bin')
        self.assertEqual(env['LD_LIBRARY_PATH'], prefix + '/lib:/host/driver')
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'], '0,1')
        self.assertEqual(env['PYTHONPATH'], '/project/src')

    def test_native_runtime_provenance_checks_physical_file_and_archive_scope(self):
        root = self.root / 'extracted'
        path = root / 'usr/local/miniconda3/envs/py312/lib/module.py'
        path.parent.mkdir(parents=True)
        path.write_text('restored module')
        info = path.stat()
        record = {'path': str(path), 'device': info.st_dev, 'inode': info.st_ino, 'bytes': info.st_size}
        roots = [PurePosixPath('/usr/local/miniconda3/envs/py312')]
        self.assertEqual(restore.check_runtime_file(record, root, roots, 'native'), str(path.resolve()))
        for changed in ({**record, 'inode': info.st_ino + 1}, {**record, 'bytes': info.st_size + 1}):
            with self.assertRaisesRegex(ValueError, 'inode'):
                restore.check_runtime_file(changed, root, roots, 'native')
        with self.assertRaisesRegex(ValueError, 'archive prefixes'):
            restore.check_runtime_file(record, root, [PurePosixPath('/different/archive')], 'native')
        outside = self.root / 'host_module.py'
        outside.write_text('host module')
        with self.assertRaisesRegex(ValueError, 'escaped'):
            restore.check_runtime_file({**record, 'path': str(outside)}, root, roots, 'native')
        path.unlink()
        path.symlink_to(outside)
        with self.assertRaisesRegex(ValueError, 'escaped'):
            restore.check_runtime_file(record, root, roots, 'native')

    def test_probe_source_is_valid_python_and_execution_mode_is_explicit(self):
        compile(restore.PROBE, '<restored-runtime-probe>', 'exec')
        self.assertIsNone(restore.parser().parse_args([]).execution)
        for mode in ('native', 'proot'):
            self.assertEqual(restore.parser().parse_args(['--execution', mode]).execution, mode)
        self.parse_rejected(['--execution', 'host'])

    def test_native_import_guard_rejects_host_modules_and_only_allows_current_jit_cache(self):
        prefix = self.root / 'independent'
        own = self.output / 'cache/triton_cache_dir/cuda_utils.so'
        foreign = self.root / 'host_cache/cuda_utils.so'
        module = prefix / 'lib/python3.12/site-packages/torch/__init__.py'
        for path in (own, foreign, module):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('test file')
        nodes = [node for node in ast.parse(restore.PROBE).body if isinstance(node, ast.FunctionDef)
            and node.name in ('record', 'native_inventory')]
        namespace = {'Path': Path, 'os': SimpleNamespace(environ={'RESTORED_NATIVE_PREFIX': str(prefix),
            'PYTHONPATH': str(self.project / 'src')}), 'origins': self.output / 'origins.json', 'evidence': {},
            'sys': SimpleNamespace(modules={'torch': SimpleNamespace(__file__=str(module)),
                'cuda_utils': SimpleNamespace(__file__=str(own))})}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), '<guard>', 'exec'), namespace)
        namespace['native_inventory']()
        self.assertEqual(len(namespace['evidence']['native_loaded_files']), 1)
        self.assertEqual(len(namespace['evidence']['generated_runtime_extensions']), 1)
        namespace['sys'].modules['cuda_utils'].__file__ = str(foreign)
        with self.assertRaisesRegex(RuntimeError, 'escaped the independent restored environment'):
            namespace['native_inventory']()

    def test_visible_gpu_index_uses_driver_minor_instead_of_ordinal(self):
        driver0 = Mock()
        driver0.read_text.return_value = 'GPU UUID: GPU-a\nDevice Minor: 0\n'
        driver1 = Mock()
        driver1.read_text.return_value = 'GPU UUID: GPU-b\nDevice Minor: 4\n'
        def device_info(path):
            minor = int(path.name.removeprefix('nvidia'))
            return Mock(st_mode=stat.S_IFCHR, st_rdev=os.makedev(195, minor))
        with patch.object(restore.subprocess, 'check_output', return_value='0, GPU-a\n1, GPU-b\n'), patch.object(restore.Path, 'glob', return_value=[driver0, driver1]), patch.object(restore.Path, 'stat', device_info):
            result = restore.gpu_device_nodes([0, 1])
        self.assertEqual([row['device_node'] for row in result], ['/dev/nvidia0', '/dev/nvidia4'])
        self.assertEqual([row['nvidia_index'] for row in result], [0, 1])

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.project = self.root / "project"
        (self.project / "runs").mkdir(parents=True)
        self.output = self.project / "runs/env-restored-gpu-check-unit"

    def model_fixture(self):
        actor = self.project / "models/actor-4b-nf4"
        actor.mkdir(parents=True)
        manifest = actor / "base_manifest.json"
        manifest.write_text(json.dumps({
            "model_id": "Qwen/Qwen3.5-4B",
            "revision": ACTOR_REVISION,
        }))
        renderer = self.project / "models/renderer" / RENDERER_REVISION
        for component in ("tokenizer", "text_encoder", "vae", "transformer"):
            (renderer / component).mkdir(parents=True)
        return manifest, renderer

    def parse_rejected(self, arguments):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                restore.parser().parse_args(arguments)
        self.assertEqual(error.exception.code, 2)

    def test_parser_defaults_to_plan_for_both_components_on_gpu_one(self):
        args = restore.parser().parse_args([])
        self.assertFalse(args.execute)
        self.assertEqual(str(args.gpu), "1")
        self.assertEqual(list(args.components), ["actor", "renderer"])

    def test_parser_accepts_explicit_execution_and_each_supported_gpu(self):
        for gpu in ("0", "1"):
            with self.subTest(gpu=gpu):
                args = restore.parser().parse_args([
                    "--gpu", gpu, "--components", "actor", "--execute",
                ])
                self.assertEqual(str(args.gpu), gpu)
                self.assertEqual(list(args.components), ["actor"])
                self.assertTrue(args.execute)

    def test_parser_rejects_unsupported_gpus_and_components(self):
        for gpu in ("-1", "2", "all"):
            with self.subTest(gpu=gpu):
                self.parse_rejected(["--gpu", gpu])
        for component in ("judge", "all", "unknown"):
            with self.subTest(component=component):
                self.parse_rejected(["--components", component])
        self.parse_rejected(["--components"])

    def test_output_validation_accepts_new_run_without_creating_it(self):
        restore.validate_output(self.project, self.output)
        self.assertFalse(self.output.exists())

    def test_output_validation_rejects_existing_directory_or_file(self):
        for kind in ("directory", "file"):
            with self.subTest(kind=kind):
                output = self.project / "runs" / ("env-restored-gpu-check-" + kind)
                if kind == "directory":
                    output.mkdir()
                else:
                    output.write_text("existing result")
                with self.assertRaises((ValueError, FileExistsError)):
                    restore.validate_output(self.project, output)
                self.assertTrue(output.exists())
                if kind == "file":
                    self.assertEqual(output.read_text(), "existing result")

    def test_output_validation_rejects_escape_and_unrelated_run_name(self):
        cases = [
            self.root / "env-restored-gpu-check-outside",
            self.project / "env-restored-gpu-check-outside-runs",
            self.project / "runs/../env-restored-gpu-check-traversal",
            self.project / "runs/actor-full-200",
        ]
        for output in cases:
            with self.subTest(output=str(output)):
                with self.assertRaises(ValueError):
                    restore.validate_output(self.project, output)
                self.assertFalse(output.exists())

    def test_output_validation_rejects_parent_symlink_escape(self):
        outside = self.root / "outside"
        outside.mkdir()
        (self.project / "runs/linked").symlink_to(outside, target_is_directory=True)
        output = self.project / "runs/linked/env-restored-gpu-check-escape"
        with self.assertRaises(ValueError):
            restore.validate_output(self.project, output)
        self.assertFalse(output.exists())

    def test_model_inputs_accept_exact_actor_identity_and_renderer_directories(self):
        self.model_fixture()
        models = restore.model_inputs(self.project)
        self.assertIsInstance(models, list)
        self.assertGreaterEqual(len(models), 2)

    def test_model_inputs_reject_different_actor_revision_or_identity(self):
        manifest, _ = self.model_fixture()
        for changed in (
            {"model_id": "Qwen/Qwen3.5-4B", "revision": "0" * 40},
            {"model_id": "Qwen/another-model", "revision": ACTOR_REVISION},
        ):
            with self.subTest(manifest=changed):
                manifest.write_text(json.dumps(changed))
                with self.assertRaises(ValueError):
                    restore.model_inputs(self.project)

    def test_model_inputs_reject_missing_renderer_component(self):
        _, renderer = self.model_fixture()
        (renderer / "transformer").rmdir()
        with self.assertRaises((ValueError, FileNotFoundError)):
            restore.model_inputs(self.project)

    def test_source_identity_changes_only_for_python_sources_under_src(self):
        source = self.project / "src/package/module.py"
        source.parent.mkdir(parents=True)
        source.write_text("value = 1\n")
        unrelated = self.project / "other.py"
        unrelated.write_text("outside = 1\n")
        binary = source.with_suffix(".bin")
        binary.write_bytes(b"unrelated bytes")
        initial = restore.source_identity(self.project)
        self.assertIn("sha256", initial)
        self.assertIn("files", initial)
        self.assertEqual(len(initial["sha256"]), 64)
        self.assertEqual(initial, restore.source_identity(self.project))
        unrelated.write_text("outside = 2\n")
        binary.write_bytes(b"changed unrelated bytes")
        self.assertEqual(initial, restore.source_identity(self.project))
        source.write_text("value = 2\n")
        changed = restore.source_identity(self.project)
        self.assertNotEqual(initial["sha256"], changed["sha256"])

    def test_component_commands_use_separate_outputs_one_gpu_and_no_saved_policies(self):
        output = Path("/workspace/agentic-image-pilot-small/runs/env-restored-gpu-check-unit")
        run_dirs = []
        for component in ("actor", "renderer"):
            with self.subTest(component=component):
                arguments = restore.cli_arguments(component, output)
                self.assertEqual(arguments[:3], ["-m", "agentic_image.cli", "preflight"])
                self.assertEqual(arguments[arguments.index("--component") + 1], component)
                self.assertEqual(arguments[arguments.index("--cycles") + 1], "1")
                self.assertNotIn("all", arguments)
                overrides = dict(
                    arguments[index + 1].split("=", 1)
                    for index, argument in enumerate(arguments) if argument == "--set"
                )
                for device in ("actor", "renderer", "encoders", "judge"):
                    self.assertEqual(overrides["devices." + device], "cuda:0")
                for model in ("actor", "renderer"):
                    for adapter in ("policy", "reference"):
                        self.assertEqual(overrides[f"models.{model}_{adapter}"], "null")
                self.assertEqual(overrides["data.artifacts"], str(output / "inputs/artifacts"))
                self.assertEqual(overrides["run_dir"], str(output / component))
                run_dirs.append(overrides["run_dir"])
        self.assertEqual(len(set(run_dirs)), 2)


if __name__ == "__main__":
    unittest.main()
