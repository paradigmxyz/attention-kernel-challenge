from __future__ import annotations

import multiprocessing.connection
import time
import traceback
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
import torch.multiprocessing as mp

from .cases import MaterializedCase, materialize_case
from .execution_policy import submission_runtime_guard
from .spec import CaseSpec, VariantSpec


@dataclass
class TimedInvocationResult:
    call_index: int
    latency_ms: float


class IsolatedSubmissionError(RuntimeError):
    pass


class IsolatedSubmissionRunner(AbstractContextManager["IsolatedSubmissionRunner"]):
    def __init__(self, submission_dir: str | Path, device: str) -> None:
        self._submission_dir = str(Path(submission_dir).resolve())
        self._device = device
        self._ctx = mp.get_context("spawn")
        self._parent_conn, child_conn = self._ctx.Pipe()
        self._process = self._ctx.Process(
            target=_submission_worker_main,
            args=(child_conn, self._submission_dir, self._device),
        )
        self._process.start()
        child_conn.close()
        response = self._recv()
        self.variants = tuple(response["variants"])

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            if self._process.is_alive():
                self._send({"command": "close"})
        except Exception:
            pass
        finally:
            self._parent_conn.close()
            self._process.join(timeout=1.0)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=1.0)
        return None

    def run_setup(
        self,
        suite_specs: Sequence[CaseSpec],
        setup_warmup_iters: int,
        setup_device: str,
    ) -> None:
        self._send(
            {
                "command": "setup",
                "setup_device": setup_device,
                "setup_warmup_iters": setup_warmup_iters,
                "suite_specs": list(suite_specs),
            }
        )
        self._recv()

    def run_public_warmups(self, warmup_specs: Sequence[CaseSpec]) -> None:
        self._send(
            {
                "command": "run_public_warmups",
                "warmup_specs": list(warmup_specs),
            }
        )
        self._recv()

    def run_timed_call(self, case: MaterializedCase) -> TimedInvocationResult:
        self._send(
            {
                "command": "run_timed_call",
                "q": case.q,
                "k": case.k,
                "v": case.v,
                "row_ptr": case.row_ptr,
                "col_idx": case.col_idx,
                "seq_lens": case.seq_lens,
            }
        )
        response = self._recv()
        return TimedInvocationResult(
            call_index=int(response["call_index"]),
            latency_ms=float(response["latency_ms"]),
        )

    def fetch_timed_output(self, call_index: int) -> tuple[torch.Tensor, torch.Tensor]:
        self._send(
            {
                "command": "fetch_timed_output",
                "call_index": int(call_index),
            }
        )
        response = self._recv()
        return response["output"], response["lse"]

    def clear_timed_outputs(self) -> None:
        self._send({"command": "clear_timed_outputs"})
        self._recv()

    def _send(self, payload: dict) -> None:
        self._parent_conn.send(payload)

    def _recv(self) -> dict:
        try:
            response = self._parent_conn.recv()
        except EOFError:
            self._raise_process_failure("Isolated submission worker exited unexpectedly.")
        if response.get("ok"):
            return response
        raise IsolatedSubmissionError(response.get("error", "Isolated submission worker failed."))

    def _raise_process_failure(self, message: str) -> None:
        exitcode = self._process.exitcode
        if exitcode is None and self._process.is_alive():
            raise IsolatedSubmissionError(message)
        raise IsolatedSubmissionError(f"{message} Worker exit code: {exitcode}.")


def _submission_worker_main(
    conn: multiprocessing.connection.Connection,
    submission_dir: str,
    device: str,
) -> None:
    try:
        _scrub_preloaded_harness_modules()
        from .submission_loader import load_submission

        with submission_runtime_guard():
            loaded = load_submission(submission_dir)
        conn.send({"ok": True, "variants": loaded.variants})
        timed_outputs: list[tuple[torch.Tensor, torch.Tensor]] = []

        while True:
            request = conn.recv()
            command = request["command"]
            if command == "close":
                return
            if command == "setup":
                _run_setup_command(
                    loaded=loaded,
                    suite_specs=request["suite_specs"],
                    setup_device=str(request["setup_device"]),
                    setup_warmup_iters=int(request["setup_warmup_iters"]),
                    device=device,
                )
                conn.send({"ok": True})
                continue
            if command == "run_public_warmups":
                _run_public_warmups_command(
                    loaded=loaded,
                    warmup_specs=request["warmup_specs"],
                    device=device,
                )
                conn.send({"ok": True})
                continue
            if command == "run_timed_call":
                latency_ms, output, lse = _run_timed_call_command(
                    loaded=loaded,
                    q=request["q"],
                    k=request["k"],
                    v=request["v"],
                    row_ptr=request["row_ptr"],
                    col_idx=request["col_idx"],
                    seq_lens=request["seq_lens"],
                )
                # Some compiled CUDA paths hand back storage that cannot be sent
                # over multiprocessing IPC directly. Clone outside the timed
                # region so correctness fetches stay transport-safe.
                timed_outputs.append((output.clone(), lse.clone()))
                conn.send(
                    {
                        "ok": True,
                        "call_index": len(timed_outputs) - 1,
                        "latency_ms": latency_ms,
                    }
                )
                continue
            if command == "fetch_timed_output":
                output, lse = timed_outputs[int(request["call_index"])]
                conn.send(
                    {
                        "ok": True,
                        "output": output,
                        "lse": lse,
                    }
                )
                continue
            if command == "clear_timed_outputs":
                timed_outputs.clear()
                conn.send({"ok": True})
                continue
            raise ValueError(f"Unknown worker command {command!r}.")
    except BaseException:
        try:
            conn.send({"ok": False, "error": traceback.format_exc()})
        except Exception:
            pass
    finally:
        conn.close()


def _scrub_preloaded_harness_modules() -> None:
    import sys

    root = sys.modules.get("attention_kernel_challenge")
    if root is not None:
        for attr in (
            "reference_block_sparse_attn_fwd",
            "dense_reference_block_sparse_attn_fwd",
            "evaluate_reference_suite",
            "evaluate_callable",
            "EvaluationSummary",
        ):
            if hasattr(root, attr):
                delattr(root, attr)
    for module_name in (
        "attention_kernel_challenge.reference",
        "attention_kernel_challenge.evaluator",
    ):
        sys.modules.pop(module_name, None)


def _run_setup_command(
    loaded,
    suite_specs: Sequence[CaseSpec],
    setup_device: str,
    setup_warmup_iters: int,
    device: str,
) -> None:
    with submission_runtime_guard():
        loaded.run_setup(suite_specs, setup_device)
    for case_spec in suite_specs:
        case = materialize_case(case_spec, device=device)
        for _ in range(setup_warmup_iters):
            _invoke_submission_entrypoint(
                loaded,
                case.q,
                case.k,
                case.v,
                case.row_ptr,
                case.col_idx,
                case.seq_lens,
            )
            _synchronize(case.q.device)


def _run_public_warmups_command(
    loaded,
    warmup_specs: Sequence[CaseSpec],
    device: str,
) -> None:
    for case_spec in warmup_specs:
        case = materialize_case(case_spec, device=device)
        _invoke_submission_entrypoint(
            loaded,
            case.q,
            case.k,
            case.v,
            case.row_ptr,
            case.col_idx,
            case.seq_lens,
        )
        _synchronize(case.q.device)


def _run_timed_call_command(
    loaded,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    row_ptr: torch.Tensor,
    col_idx: torch.Tensor,
    seq_lens: torch.Tensor,
) -> tuple[float, torch.Tensor, torch.Tensor]:
    start = time.perf_counter()
    output, lse = _invoke_submission_entrypoint(loaded, q, k, v, row_ptr, col_idx, seq_lens)
    _synchronize(q.device)
    return (time.perf_counter() - start) * 1000.0, output, lse


def _invoke_submission_entrypoint(
    loaded,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    row_ptr: torch.Tensor,
    col_idx: torch.Tensor,
    seq_lens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    with submission_runtime_guard():
        return loaded.entrypoint(q, k, v, row_ptr, col_idx, seq_lens)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()
