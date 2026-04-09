from __future__ import annotations

import json
import os
import traceback

import modal

from .evaluator import EvaluationSummary, evaluate_reference_suite, evaluate_submission_dir
from .spec import EvaluationConfig
from .submission_loader import unpack_submission_archive


MODAL_GPU = os.environ.get("ATTENTION_KERNEL_CHALLENGE_MODAL_GPU", "H100!:1")
MODAL_TIMEOUT_S = int(os.environ.get("ATTENTION_KERNEL_CHALLENGE_MODAL_TIMEOUT_S", "120"))
MODAL_PYTHON_VERSION = os.environ.get("ATTENTION_KERNEL_CHALLENGE_MODAL_PYTHON_VERSION", "3.11")
MODAL_ENABLE_MEMORY_SNAPSHOT = (
    os.environ.get("ATTENTION_KERNEL_CHALLENGE_MODAL_ENABLE_MEMORY_SNAPSHOT", "1") != "0"
)

app = modal.App("attention-kernel-challenge-harness")

image = (
    modal.Image.debian_slim(python_version=MODAL_PYTHON_VERSION)
    .pip_install(
        "torch==2.8.0",
        index_url="https://download.pytorch.org/whl/cu128",
    )
    .pip_install("triton==3.4.0")
    .pip_install("numpy==2.0.2")
    .add_local_python_source("attention_kernel_challenge")
)


@app.function(
    image=image,
    gpu=MODAL_GPU,
    timeout=MODAL_TIMEOUT_S,
    enable_memory_snapshot=MODAL_ENABLE_MEMORY_SNAPSHOT,
    block_network=True,
    restrict_modal_access=True,
    max_inputs=1,
)
def run_reference_eval(request_json: str) -> str:
    payload = json.loads(request_json)
    try:
        config = EvaluationConfig(
            device=payload.get("device", "cuda"),
            warmup_iters=int(payload.get("warmup_iters", 1)),
            measure_iters=int(payload.get("measure_iters", 3)),
            check_correctness=True,
            isolate_submission_process=True,
            suite_manifest_json=payload.get("suite_manifest_json"),
        )
        summary = evaluate_reference_suite(payload["suite"], config)
        return summary.to_json()
    except Exception:
        return _remote_failure_summary(payload, traceback.format_exc())


@app.function(
    image=image,
    gpu=MODAL_GPU,
    timeout=MODAL_TIMEOUT_S,
    enable_memory_snapshot=MODAL_ENABLE_MEMORY_SNAPSHOT,
    block_network=True,
    restrict_modal_access=True,
    max_inputs=1,
)
def run_submission_eval(request_json: str, submission_archive: bytes) -> str:
    payload = json.loads(request_json)
    try:
        config = EvaluationConfig(
            device=payload.get("device", "cuda"),
            warmup_iters=int(payload.get("warmup_iters", 1)),
            measure_iters=int(payload.get("measure_iters", 3)),
            check_correctness=True,
            isolate_submission_process=True,
            suite_manifest_json=payload.get("suite_manifest_json"),
        )
        submission_root = unpack_submission_archive(submission_archive)
        summary = evaluate_submission_dir(
            submission_dir=str(submission_root),
            suite_name=payload["suite"],
            config=config,
        )
        return summary.to_json()
    except Exception:
        return _remote_failure_summary(payload, traceback.format_exc())


def _remote_failure_summary(payload: dict, trace: str) -> str:
    return EvaluationSummary(
        suite=payload.get("suite", "remote"),
        device=payload.get("device", "cuda"),
        overall_valid=False,
        geometric_mean_family_latency_ms=None,
        worst_family_latency_ms=None,
        case_results=[],
        failure_reason=f"Remote evaluation failed:\n{trace}",
    ).to_json()
