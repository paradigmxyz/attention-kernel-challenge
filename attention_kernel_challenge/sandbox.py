from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class SandboxResult:
    returncode: int
    stdout: str
    stderr: str
    used_sandbox: bool


def nsjail_available() -> bool:
    return platform.system() == "Linux" and shutil.which("nsjail") is not None


def run_python_module(
    module: str,
    module_args: Sequence[str],
    repo_root: str,
) -> SandboxResult:
    if not nsjail_available():
        raise RuntimeError(
            "Sandbox mode was requested, but nsjail is unavailable on this machine."
        )

    return _run_nsjail(module, module_args, repo_root)


def build_nsjail_command(
    module: str,
    module_args: Sequence[str],
    repo_root: str,
    scratch_dir: str,
    jail_root: str,
) -> list[str]:
    repo_path = Path(repo_root).resolve()
    command = [
        "nsjail",
        "--config",
        str(repo_path / "sandbox" / "nsjail.cfg"),
        "--chroot",
        jail_root,
        "--bindmount_ro",
        f"{repo_path}:/workspace",
        "--bindmount",
        f"{scratch_dir}:/tmp",
        "--cwd",
        "/workspace",
        "--env",
        "HOME=/tmp",
        "--env",
        "TMPDIR=/tmp",
        "--env",
        "PYTHONNOUSERSITE=1",
        "--env",
        "PYTHONPATH=/workspace",
    ]

    for path in _readonly_mounts(repo_path):
        command.extend(["--bindmount_ro", f"{path}:{path}"])
    for path in _writable_device_mounts():
        command.extend(["--bindmount", f"{path}:{path}"])

    command.extend(
        [
            "--",
            sys.executable,
            "-m",
            module,
            *module_args,
        ]
    )
    return command


def _run_nsjail(module: str, module_args: Sequence[str], repo_root: str) -> SandboxResult:
    scratch_dir = tempfile.mkdtemp(prefix="attention-kernel-challenge-sandbox-tmp-")
    jail_root = tempfile.mkdtemp(prefix="attention-kernel-challenge-sandbox-root-")
    try:
        command = build_nsjail_command(module, module_args, repo_root, scratch_dir, jail_root)
        completed = subprocess.run(
            command,
            cwd=repo_root,
            text=True,
            capture_output=True,
            check=False,
        )
        return SandboxResult(
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            used_sandbox=True,
        )
    finally:
        shutil.rmtree(scratch_dir, ignore_errors=True)
        shutil.rmtree(jail_root, ignore_errors=True)


def _readonly_mounts(repo_root: Path) -> list[str]:
    mounts = {
        str(path)
        for path in (
            Path("/usr"),
            Path("/bin"),
            Path("/lib"),
            Path("/lib64"),
            Path("/sbin"),
            Path("/etc"),
        )
        if path.exists()
    }

    interpreter_path = Path(sys.executable).resolve()
    for path in {interpreter_path, interpreter_path.parent} | _python_runtime_paths():
        if not path.exists():
            continue
        if _is_within(path, repo_root):
            continue
        mounts.add(str(path))

    for device_path in (
        Path("/dev/null"),
        Path("/dev/zero"),
        Path("/dev/random"),
        Path("/dev/urandom"),
    ):
        if device_path.exists():
            mounts.add(str(device_path))

    return sorted(mounts)


def _python_runtime_paths() -> set[Path]:
    import site
    import sysconfig

    paths = {
        Path(value).resolve()
        for value in sysconfig.get_paths().values()
        if value
    }
    for value in site.getsitepackages():
        paths.add(Path(value).resolve())
    return paths


def _writable_device_mounts() -> list[str]:
    device_paths = []
    for pattern in (
        "/dev/nvidiactl",
        "/dev/nvidia-uvm",
        "/dev/nvidia-uvm-tools",
        "/dev/nvidia-modeset",
    ):
        path = Path(pattern)
        if path.exists():
            device_paths.append(str(path))
    for path in sorted(Path("/dev").glob("nvidia[0-9]*")):
        if path.exists():
            device_paths.append(str(path))
    return device_paths


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True
