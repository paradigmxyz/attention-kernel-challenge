import tempfile
import unittest
from io import StringIO
from pathlib import Path
from unittest import mock

from attention_kernel_challenge.backends import (
    NoBackendConfiguredError,
    modal_cli_available,
    resolve_backend_name,
    run_modal_reference_eval,
)
from attention_kernel_challenge.cli import _handle_backend_command, _local_correctness_only, _local_setup_device, build_parser, main
from attention_kernel_challenge.config import HarnessConfig, ModalBackendConfig, clear_config, load_config, save_config


class BackendConfigTests(unittest.TestCase):
    def test_local_correctness_only_depends_on_device(self) -> None:
        self.assertTrue(_local_correctness_only("cpu"))
        self.assertFalse(_local_correctness_only("cuda"))
        self.assertFalse(_local_correctness_only("mps"))

    def test_local_setup_device_only_changes_in_serverlike_mode(self) -> None:
        self.assertIsNone(_local_setup_device(False))
        self.assertEqual(_local_setup_device(True), "cuda")

    def test_cli_parser_accepts_local_setup_timeout_override(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "eval-submission",
                "--submission-dir",
                "example_submission",
                "--setup-timeout-s",
                "30",
            ]
        )
        self.assertEqual(args.setup_timeout_s, 30.0)

    def test_modal_cli_available_accepts_python_module_without_shell_binary(self) -> None:
        with mock.patch("attention_kernel_challenge.backends.shutil.which", return_value=None):
            with mock.patch("attention_kernel_challenge.backends.importlib.util.find_spec", return_value=object()):
                self.assertTrue(modal_cli_available())

    def test_config_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            original = HarnessConfig(
                default_backend="modal",
                modal=ModalBackendConfig(gpu="H100!:4", timeout_s=1200, python_version="3.11"),
            )
            save_config(root, original)
            loaded = load_config(root)
            self.assertEqual(original, loaded)

    def test_clear_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            save_config(root, HarnessConfig(default_backend="local"))
            clear_config(root)
            self.assertIsNone(load_config(root))

    def test_backend_resolution_requires_config_or_explicit_backend(self) -> None:
        with self.assertRaises(NoBackendConfiguredError):
            resolve_backend_name(None, None)
        self.assertEqual(resolve_backend_name(None, "local"), "local")

    def test_setup_modal_failure_does_not_persist_broken_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            args = build_parser().parse_args(["backend", "setup-modal"])
            stderr = StringIO()
            with mock.patch("attention_kernel_challenge.cli.deploy_modal_app", side_effect=RuntimeError("boom")):
                with mock.patch("sys.stderr", stderr):
                    result = _handle_backend_command(args, root, None)
            self.assertEqual(result, 1)
            self.assertIsNone(load_config(root))
            self.assertIn("boom", stderr.getvalue())

    def test_distribution_suite_prints_distribution_metadata(self) -> None:
        stdout = StringIO()
        with mock.patch("sys.stdout", stdout):
            result = main(["show-suite", "--suite", "full"])
        self.assertEqual(result, 0)
        self.assertIn('"profile": "full"', stdout.getvalue())
        self.assertIn('"seed_policy": "hidden-site-sampled"', stdout.getvalue())

    @mock.patch("attention_kernel_challenge.backends.modal_profile_current", return_value=(True, "default"))
    @mock.patch("attention_kernel_challenge.backends.modal_cli_available", return_value=True)
    def test_modal_reference_eval_prefers_deployed_function(self, *_mocks) -> None:
        fake_remote = mock.Mock()
        fake_remote.remote.return_value = "deployed-ok"
        with mock.patch("modal.Function.from_name", return_value=fake_remote) as from_name:
            with mock.patch("attention_kernel_challenge.modal_backend.app.run") as app_run:
                result = run_modal_reference_eval(
                    repo_root=Path.cwd(),
                    suite="smoke",
                    suite_manifest_json=None,
                    warmup_iters=0,
                    measure_iters=1,
                    modal_config=ModalBackendConfig(),
                )
        self.assertEqual(result, "deployed-ok")
        from_name.assert_called_once()
        fake_remote.hydrate.assert_called_once()
        app_run.assert_not_called()

    @mock.patch("attention_kernel_challenge.backends.modal_profile_current", return_value=(True, "default"))
    @mock.patch("attention_kernel_challenge.backends.modal_cli_available", return_value=True)
    def test_modal_reference_eval_falls_back_to_ephemeral_app(self, *_mocks) -> None:
        context = mock.MagicMock()
        context.__enter__.return_value = None
        context.__exit__.return_value = False
        with mock.patch("modal.Function.from_name", side_effect=RuntimeError("not deployed")):
            with mock.patch("attention_kernel_challenge.modal_backend.app.run", return_value=context) as app_run:
                with mock.patch(
                    "attention_kernel_challenge.modal_backend.run_reference_eval.remote",
                    return_value="fallback-ok",
                ) as remote:
                    result = run_modal_reference_eval(
                        repo_root=Path.cwd(),
                        suite="smoke",
                        suite_manifest_json=None,
                        warmup_iters=0,
                        measure_iters=1,
                        modal_config=ModalBackendConfig(),
                    )
        self.assertEqual(result, "fallback-ok")
        app_run.assert_called_once()
        remote.assert_called_once()

    @mock.patch("attention_kernel_challenge.backends.modal_profile_current", return_value=(True, "default"))
    @mock.patch("attention_kernel_challenge.backends.modal_cli_available", return_value=True)
    def test_modal_reference_eval_can_force_one_shot_mode(self, *_mocks) -> None:
        context = mock.MagicMock()
        context.__enter__.return_value = None
        context.__exit__.return_value = False
        with mock.patch("modal.Function.from_name") as from_name:
            with mock.patch("attention_kernel_challenge.modal_backend.app.run", return_value=context) as app_run:
                with mock.patch(
                    "attention_kernel_challenge.modal_backend.run_reference_eval.remote",
                    return_value="one-shot-ok",
                ) as remote:
                    result = run_modal_reference_eval(
                        repo_root=Path.cwd(),
                        suite="smoke",
                        suite_manifest_json=None,
                        warmup_iters=0,
                        measure_iters=1,
                        modal_config=ModalBackendConfig(),
                        prefer_deployed=False,
                    )
        self.assertEqual(result, "one-shot-ok")
        from_name.assert_not_called()
        app_run.assert_called_once()
        remote.assert_called_once()


if __name__ == "__main__":
    unittest.main()
