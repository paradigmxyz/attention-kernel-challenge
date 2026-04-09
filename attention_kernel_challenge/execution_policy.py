from __future__ import annotations

import contextvars
import ctypes
import os
import pty
import shutil
import socket
import subprocess
import sys
import tempfile
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator
from unittest.mock import patch


class PolicyViolationError(RuntimeError):
    pass


_AUDIT_GUARD_ACTIVE: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "attention_kernel_challenge_audit_guard_active",
    default=False,
)
_ALLOWED_CTYPES_HANDLES: contextvars.ContextVar[set[int] | None] = contextvars.ContextVar(
    "attention_kernel_challenge_allowed_ctypes_handles",
    default=None,
)
_AUDIT_HOOK_INSTALLED = False
_ALLOWED_CTYPES_LIBRARY_PREFIXES = (
    "libcuda.so",
    "libcudart.so",
    "libnvidia-ml.so",
    "libnvrtc.so",
    "libnvJitLink.so",
)
_ALLOWED_SUBPROCESS_COMMANDS = {
    ("ldconfig", "-p"),
}


def _blocked_callable(name: str) -> Callable:
    def _raise(*args, **kwargs):
        raise PolicyViolationError(f"{name} is not permitted during challenge evaluation.")

    return _raise


def _is_allowed_ctypes_library(name: object) -> bool:
    if name is None:
        return False
    if isinstance(name, bytes):
        candidate = name.decode(errors="ignore")
    else:
        candidate = os.fspath(name)
    basename = os.path.basename(candidate)
    return any(basename.startswith(prefix) for prefix in _ALLOWED_CTYPES_LIBRARY_PREFIXES)


def _guarded_ctypes_loader(original: Callable, label: str) -> Callable:
    def _wrapped(name, *args, **kwargs):
        if _AUDIT_GUARD_ACTIVE.get() and not _is_allowed_ctypes_library(name):
            raise PolicyViolationError(f"{label} is not permitted during challenge evaluation.")
        library = original(name, *args, **kwargs)
        handles = _ALLOWED_CTYPES_HANDLES.get()
        if handles is not None and hasattr(library, "_handle"):
            handles.add(int(library._handle))
        return library

    return _wrapped


def _normalize_subprocess_argv(argv: object) -> tuple[str, ...] | None:
    if isinstance(argv, (list, tuple)):
        return tuple(str(part) for part in argv)
    return None


def _is_allowed_subprocess_argv(argv: object) -> bool:
    normalized = _normalize_subprocess_argv(argv)
    if not normalized:
        return False
    command = normalized[0]
    basename = os.path.basename(command)
    if (basename, *normalized[1:]) in _ALLOWED_SUBPROCESS_COMMANDS and _is_trusted_subprocess_binary(
        command,
        {basename},
    ):
        return True
    if basename == "uname" and normalized[1:] == ("-p",) and _is_trusted_subprocess_binary(
        command,
        {"uname"},
    ):
        return True
    if basename == "ptxas":
        return _is_allowed_ptxas_invocation(normalized)
    return False


def _is_allowed_file_probe_target(target: str) -> bool:
    try:
        resolved_target = Path(target).resolve()
        resolved_python = Path(sys.executable).resolve()
    except OSError:
        return False
    if resolved_target == resolved_python:
        return True
    return resolved_target.parent == resolved_python.parent and resolved_target.name.startswith("python")


def _should_simulate_missing_subprocess(argv: object) -> bool:
    normalized = _normalize_subprocess_argv(argv)
    if not normalized:
        return False
    command = os.path.basename(normalized[0])
    if command == "nvcc" and normalized[1:] == ("--version",):
        return True
    if command == "file" and len(normalized) == 3 and normalized[1] == "-b":
        return _is_allowed_file_probe_target(normalized[2])
    return False


def _is_allowed_ptxas_invocation(argv: tuple[str, ...]) -> bool:
    if not _is_trusted_subprocess_binary(argv[0], {"ptxas"}):
        return False
    if argv[1:] == ("--version",):
        return True
    return len(argv) >= 2


def _is_trusted_subprocess_binary(command: str, expected_basenames: set[str]) -> bool:
    path = _resolve_subprocess_command(command)
    if path is None or path.name not in expected_basenames:
        return False
    for root in _trusted_subprocess_roots():
        try:
            path.relative_to(root)
        except ValueError:
            continue
        return True
    return False


def _resolve_subprocess_command(command: str) -> Path | None:
    if os.path.isabs(command):
        candidate = Path(command)
    else:
        discovered = shutil.which(command)
        if discovered is not None:
            candidate = Path(discovered)
        else:
            candidate = _fallback_subprocess_command_path(command)
            if candidate is None:
                return None
    try:
        return candidate.resolve()
    except OSError:
        return None


def _trusted_subprocess_roots() -> tuple[Path, ...]:
    roots = {
        Path("/bin"),
        Path("/sbin"),
        Path("/usr/bin"),
        Path("/usr/sbin"),
        Path("/usr/local/bin"),
        Path("/usr/local/cuda/bin"),
    }
    try:
        import triton
    except ImportError:
        pass
    else:
        roots.add(Path(triton.__file__).resolve().parent)
    return tuple(sorted(roots))


def _fallback_subprocess_command_path(command: str) -> Path | None:
    basename = os.path.basename(command)
    for root in _trusted_subprocess_roots():
        candidate = root / basename
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    return None


def _guarded_subprocess_callable(original: Callable, label: str) -> Callable:
    def _wrapped(*args, **kwargs):
        argv = None
        if args:
            argv = args[0]
        elif "args" in kwargs:
            argv = kwargs["args"]
        if _AUDIT_GUARD_ACTIVE.get() and not _is_allowed_subprocess_argv(argv):
            if _should_simulate_missing_subprocess(argv):
                raise FileNotFoundError(f"{label} is unavailable in the submission sandbox.")
            raise PolicyViolationError(f"{label} is not permitted during challenge evaluation.")
        return original(*args, **kwargs)

    return _wrapped


def _guarded_posix_spawn_callable(original: Callable, label: str) -> Callable:
    def _wrapped(path, argv, env, *args, **kwargs):
        if _AUDIT_GUARD_ACTIVE.get() and not _is_allowed_subprocess_argv(argv):
            if _should_simulate_missing_subprocess(argv):
                raise FileNotFoundError(f"{label} is unavailable in the submission sandbox.")
            raise PolicyViolationError(f"{label} is not permitted during challenge evaluation.")
        return original(path, argv, env, *args, **kwargs)

    return _wrapped


def _install_submission_audit_hook() -> None:
    global _AUDIT_HOOK_INSTALLED
    if _AUDIT_HOOK_INSTALLED:
        return
    sys.addaudithook(_submission_audit_hook)
    _AUDIT_HOOK_INSTALLED = True


def _submission_audit_hook(event: str, args) -> None:
    if not _AUDIT_GUARD_ACTIVE.get():
        return

    blocked_events = {
        "ctypes.dlsym": "ctypes.dlsym",
        "os.exec": "os.exec",
        "os.system": "os.system",
        "socket.__new__": "socket.socket",
    }
    if event == "ctypes.dlopen":
        library_name = args[0] if args else None
        if not _is_allowed_ctypes_library(library_name):
            raise PolicyViolationError(
                "ctypes.dlopen is not permitted during challenge evaluation."
            )
        return
    if event == "ctypes.dlsym":
        library = args[0] if args else None
        allowed_handles = _ALLOWED_CTYPES_HANDLES.get() or set()
        if library is None or not hasattr(library, "_handle") or int(library._handle) not in allowed_handles:
            raise PolicyViolationError(
                "ctypes.dlsym is not permitted during challenge evaluation."
            )
        return
    if event == "subprocess.Popen":
        argv = args[1] if len(args) > 1 else (args[0] if args else None)
        if not _is_allowed_subprocess_argv(argv):
            raise PolicyViolationError(
                "subprocess.Popen is not permitted during challenge evaluation."
            )
        return
    if event == "os.posix_spawn":
        argv = args[1] if len(args) > 1 else None
        if not _is_allowed_subprocess_argv(argv):
            raise PolicyViolationError(
                "os.posix_spawn is not permitted during challenge evaluation."
            )
        return
    blocked_name = blocked_events.get(event)
    if blocked_name is not None:
        raise PolicyViolationError(
            f"{blocked_name} is not permitted during challenge evaluation."
        )


@contextmanager
def submission_runtime_guard() -> Iterator[None]:
    _install_submission_audit_hook()
    guard_token = _AUDIT_GUARD_ACTIVE.set(True)
    handles_token = _ALLOWED_CTYPES_HANDLES.set(set())
    with ExitStack() as stack:
        try:
            stack.enter_context(patch.object(socket, "socket", _blocked_callable("socket.socket")))
            stack.enter_context(
                patch.object(socket, "create_connection", _blocked_callable("socket.create_connection"))
            )
            if hasattr(socket, "socketpair"):
                stack.enter_context(patch.object(socket, "socketpair", _blocked_callable("socket.socketpair")))
            if hasattr(socket, "fromfd"):
                stack.enter_context(patch.object(socket, "fromfd", _blocked_callable("socket.fromfd")))

            stack.enter_context(
                patch.object(subprocess, "Popen", _guarded_subprocess_callable(subprocess.Popen, "subprocess.Popen"))
            )
            stack.enter_context(
                patch.object(subprocess, "run", _guarded_subprocess_callable(subprocess.run, "subprocess.run"))
            )
            stack.enter_context(patch.object(subprocess, "call", _blocked_callable("subprocess.call")))
            stack.enter_context(patch.object(subprocess, "check_call", _blocked_callable("subprocess.check_call")))
            stack.enter_context(
                patch.object(
                    subprocess,
                    "check_output",
                    _guarded_subprocess_callable(subprocess.check_output, "subprocess.check_output"),
                )
            )
            stack.enter_context(patch.object(subprocess, "getoutput", _blocked_callable("subprocess.getoutput")))
            stack.enter_context(
                patch.object(subprocess, "getstatusoutput", _blocked_callable("subprocess.getstatusoutput"))
            )

            stack.enter_context(patch.object(os, "system", _blocked_callable("os.system")))
            stack.enter_context(patch.object(os, "popen", _blocked_callable("os.popen")))
            for name in (
                "execl",
                "execle",
                "execlp",
                "execlpe",
                "execv",
                "execve",
                "execvp",
                "execvpe",
                "fork",
                "forkpty",
                "posix_spawn",
                "posix_spawnp",
                "spawnl",
                "spawnle",
                "spawnlp",
                "spawnlpe",
                "spawnv",
                "spawnve",
                "spawnvp",
                "spawnvpe",
            ):
                if hasattr(os, name):
                    if name in {"posix_spawn", "posix_spawnp"}:
                        stack.enter_context(
                            patch.object(
                                os,
                                name,
                                _guarded_posix_spawn_callable(getattr(os, name), f"os.{name}"),
                            )
                        )
                    else:
                        stack.enter_context(patch.object(os, name, _blocked_callable(f"os.{name}")))
            if hasattr(pty, "fork"):
                stack.enter_context(patch.object(pty, "fork", _blocked_callable("pty.fork")))
            stack.enter_context(patch.object(pty, "spawn", _blocked_callable("pty.spawn")))

            stack.enter_context(patch.object(ctypes, "CDLL", _guarded_ctypes_loader(ctypes.CDLL, "ctypes.CDLL")))
            stack.enter_context(patch.object(ctypes, "PyDLL", _guarded_ctypes_loader(ctypes.PyDLL, "ctypes.PyDLL")))
            stack.enter_context(
                patch.object(
                    ctypes.cdll,
                    "LoadLibrary",
                    _guarded_ctypes_loader(ctypes.cdll.LoadLibrary, "ctypes.cdll.LoadLibrary"),
                )
            )
            stack.enter_context(
                patch.object(
                    ctypes.pydll,
                    "LoadLibrary",
                    _guarded_ctypes_loader(ctypes.pydll.LoadLibrary, "ctypes.pydll.LoadLibrary"),
                )
            )

            yield
        finally:
            _ALLOWED_CTYPES_HANDLES.reset(handles_token)
            _AUDIT_GUARD_ACTIVE.reset(guard_token)


@dataclass(frozen=True)
class CacheSnapshot:
    files: dict[str, tuple[int, int]]


class CompilationCacheMonitor:
    def __init__(self) -> None:
        self.root = Path(tempfile.mkdtemp(prefix="attention-kernel-challenge-cache-"))
        self._env_updates = {
            "TRITON_CACHE_DIR": str(self.root / "triton"),
            "TORCHINDUCTOR_CACHE_DIR": str(self.root / "torchinductor"),
            "CUDA_CACHE_PATH": str(self.root / "cuda"),
            "XDG_CACHE_HOME": str(self.root / "xdg"),
            # Keep Inductor compilation in-process so setup-time compile support
            # does not require relaxing the subprocess sandbox for worker pools.
            "TORCHINDUCTOR_COMPILE_THREADS": "1",
        }
        self._old_env: dict[str, str | None] = {}
        self._baseline: CacheSnapshot | None = None

    def __enter__(self) -> "CompilationCacheMonitor":
        for value in self._env_updates.values():
            Path(value).mkdir(parents=True, exist_ok=True)
        for key, value in self._env_updates.items():
            self._old_env[key] = os.environ.get(key)
            os.environ[key] = value
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        for key, value in self._old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        shutil.rmtree(self.root, ignore_errors=True)

    def freeze(self) -> None:
        self._baseline = self._snapshot()

    def assert_unchanged(self, context: str) -> None:
        if self._baseline is None:
            raise RuntimeError("Compilation cache baseline was not frozen before use.")
        current = self._snapshot()
        if current.files != self._baseline.files:
            diff = _summarize_cache_diff(self._baseline, current)
            raise PolicyViolationError(
                f"Post-setup cache mutation detected during {context}. {diff}"
            )

    def _snapshot(self) -> CacheSnapshot:
        files: dict[str, tuple[int, int]] = {}
        for path in sorted(self.root.rglob("*")):
            if not path.is_file():
                continue
            stat = path.stat()
            files[str(path.relative_to(self.root))] = (stat.st_size, stat.st_mtime_ns)
        return CacheSnapshot(files=files)


def prepare_compile_runtime_support(device: str) -> None:
    if device != "cuda":
        return
    try:
        from triton.backends.nvidia.driver import CudaUtils
    except ImportError:
        return
    # Seed Triton's fixed runtime helper into the monitored cache outside the
    # submission sandbox so user code does not need native compiler privileges.
    CudaUtils()


def _summarize_cache_diff(before: CacheSnapshot, after: CacheSnapshot) -> str:
    before_keys = set(before.files)
    after_keys = set(after.files)
    added = sorted(after_keys - before_keys)
    removed = sorted(before_keys - after_keys)
    changed = sorted(
        key
        for key in before_keys & after_keys
        if before.files[key] != after.files[key]
    )
    fragments = []
    if added:
        fragments.append(f"added={added[:3]}")
    if removed:
        fragments.append(f"removed={removed[:3]}")
    if changed:
        fragments.append(f"changed={changed[:3]}")
    if not fragments:
        fragments.append("cache contents changed")
    return "; ".join(fragments)
