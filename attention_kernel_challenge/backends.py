from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

from .config import HarnessConfig, ModalBackendConfig


class NoBackendConfiguredError(RuntimeError):
    pass


def resolve_backend_name(config: Optional[HarnessConfig], explicit_backend: Optional[str]) -> str:
    if explicit_backend:
        return explicit_backend
    if config is not None:
        return config.default_backend
    raise NoBackendConfiguredError(
        "No evaluation backend is configured.\n"
        "Set one up with:\n"
        "  python -m attention_kernel_challenge.cli backend setup-modal\n"
        "or run a non-official local eval explicitly with:\n"
        "  python -m attention_kernel_challenge.cli eval-reference --backend local"
    )


def modal_cli_available() -> bool:
    return _modal_command_prefix() is not None


def _modal_command_prefix() -> list[str] | None:
    modal_binary = shutil.which("modal")
    if modal_binary is not None:
        return [modal_binary]
    if importlib.util.find_spec("modal") is not None:
        return [sys.executable, "-m", "modal"]
    return None


def modal_profile_current() -> tuple[bool, str]:
    command_prefix = _modal_command_prefix()
    if command_prefix is None:
        return False, "Modal is not installed in this Python environment."

    completed = subprocess.run(
        [*command_prefix, "profile", "current"],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode == 0:
        return True, completed.stdout.strip()
    stderr = completed.stderr.strip() or completed.stdout.strip() or "unknown error"
    return False, stderr


def modal_deployment_current(
    app_name: str = "attention-kernel-challenge-harness",
    function_name: str = "run_submission_eval",
    environment_name: str | None = None,
) -> tuple[bool, str]:
    if not modal_cli_available():
        return False, "Modal is not installed in this Python environment."

    ok, profile_text = modal_profile_current()
    if not ok:
        return False, profile_text

    try:
        import modal

        deployed_fn = modal.Function.from_name(
            app_name,
            function_name,
            environment_name=environment_name,
        )
        deployed_fn.hydrate()
    except Exception as exc:
        return False, str(exc)
    return True, f"{app_name}/{function_name}"


def deploy_modal_app(
    app_ref: str = "attention_kernel_challenge.modal_backend",
    modal_config: Optional[ModalBackendConfig] = None,
) -> str:
    command_prefix = _modal_command_prefix()
    if command_prefix is None:
        raise RuntimeError("Modal is not installed in this Python environment.")

    ok, profile_text = modal_profile_current()
    if not ok:
        raise RuntimeError(
            "Modal is not ready. Run `modal setup` or authenticate first. "
            f"Details: {profile_text}"
        )

    env = os.environ.copy()
    if modal_config is not None:
        env["ATTENTION_KERNEL_CHALLENGE_MODAL_GPU"] = modal_config.gpu
        env["ATTENTION_KERNEL_CHALLENGE_MODAL_TIMEOUT_S"] = str(modal_config.timeout_s)
        env["ATTENTION_KERNEL_CHALLENGE_MODAL_PYTHON_VERSION"] = modal_config.python_version

    completed = subprocess.run(
        [*command_prefix, "deploy", "-m", app_ref],
        text=True,
        capture_output=True,
        check=False,
        env=env,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.strip() or completed.stdout.strip() or "unknown error"
        raise RuntimeError(f"Modal deploy failed: {stderr}")
    return completed.stdout.strip()


def run_modal_reference_eval(
    repo_root: Path,
    suite: str,
    suite_manifest_json: Optional[str],
    warmup_iters: int,
    measure_iters: int,
    modal_config: ModalBackendConfig,
    prefer_deployed: bool = True,
) -> str:
    request_payload = {
        "suite": suite,
        "suite_manifest_json": suite_manifest_json,
        "device": "cuda",
        "warmup_iters": warmup_iters,
        "measure_iters": measure_iters,
    }
    return _run_modal_function(
        "run_reference_eval",
        request_payload,
        modal_config,
        prefer_deployed=prefer_deployed,
    )


def run_modal_submission_eval(
    suite: str,
    suite_manifest_json: Optional[str],
    submission_archive: bytes,
    warmup_iters: int,
    measure_iters: int,
    modal_config: ModalBackendConfig,
    prefer_deployed: bool = True,
) -> str:
    request_payload = {
        "suite": suite,
        "suite_manifest_json": suite_manifest_json,
        "device": "cuda",
        "warmup_iters": warmup_iters,
        "measure_iters": measure_iters,
    }
    return _run_modal_function(
        "run_submission_eval",
        request_payload,
        modal_config,
        submission_archive,
        prefer_deployed=prefer_deployed,
    )


def _run_modal_function(
    function_name: str,
    request_payload: dict,
    modal_config: ModalBackendConfig,
    submission_archive: Optional[bytes] = None,
    prefer_deployed: bool = True,
) -> str:
    if not modal_cli_available():
        raise RuntimeError("Modal is not installed in this Python environment.")

    ok, profile_text = modal_profile_current()
    if not ok:
        raise RuntimeError(
            "Modal is not ready. Run `modal setup` or authenticate first. "
            f"Details: {profile_text}"
        )

    os.environ["ATTENTION_KERNEL_CHALLENGE_MODAL_GPU"] = modal_config.gpu
    os.environ["ATTENTION_KERNEL_CHALLENGE_MODAL_TIMEOUT_S"] = str(modal_config.timeout_s)
    os.environ["ATTENTION_KERNEL_CHALLENGE_MODAL_PYTHON_VERSION"] = modal_config.python_version

    try:
        import modal

        from .modal_backend import app, run_reference_eval, run_submission_eval
    except Exception as exc:
        raise RuntimeError(f"Failed to import Modal backend module: {exc}") from exc

    remote_fn = run_reference_eval if function_name == "run_reference_eval" else run_submission_eval
    request_json = json.dumps(request_payload)

    deployed_app_name = os.environ.get("ATTENTION_KERNEL_CHALLENGE_MODAL_APP_NAME", app.name)
    deployed_environment_name = os.environ.get("ATTENTION_KERNEL_CHALLENGE_MODAL_ENVIRONMENT")
    if os.environ.get("ATTENTION_KERNEL_CHALLENGE_MODAL_PREFER_DEPLOYED") is not None:
        prefer_deployed = os.environ.get("ATTENTION_KERNEL_CHALLENGE_MODAL_PREFER_DEPLOYED", "1") != "0"

    if prefer_deployed:
        try:
            deployed_fn = modal.Function.from_name(
                deployed_app_name,
                function_name,
                environment_name=deployed_environment_name,
            )
            deployed_fn.hydrate()
            if submission_archive is None:
                return deployed_fn.remote(request_json)
            return deployed_fn.remote(request_json, submission_archive)
        except Exception:
            pass

    with app.run():
        if submission_archive is None:
            return remote_fn.remote(request_json)
        return remote_fn.remote(request_json, submission_archive)
