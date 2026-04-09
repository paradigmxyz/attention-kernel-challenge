from __future__ import annotations

import hashlib
import json
import math
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from statistics import median
from typing import Callable, Dict, List, Optional, Sequence

import torch

from .cases import MaterializedCase, materialize_case, resolve_suite
from .execution_policy import (
    CompilationCacheMonitor,
    prepare_compile_runtime_support,
    submission_runtime_guard,
)
from .isolated_submission import IsolatedSubmissionRunner
from .reference import reference_block_sparse_attn_fwd
from .spec import CaseSpec, EvaluationConfig, VariantSpec
from .submission_loader import (
    LoadedSubmission,
    find_matching_variant,
    load_submission,
)
from .validation import ValidationResult, validate_outputs


EntryPoint = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor],
]
SetupCallable = Callable[[List[CaseSpec], str], None]
PublicWarmupKey = tuple[object, ...]


@dataclass
class CaseResult:
    case_id: str
    family: str
    latency_ms: float | None
    density: float
    variant: str | None
    validation: ValidationResult


@dataclass
class EvaluationSummary:
    suite: str
    device: str
    overall_valid: bool
    geometric_mean_family_latency_ms: Optional[float]
    worst_family_latency_ms: Optional[float]
    case_results: List[CaseResult]
    scored_case_count: int | None = None
    failure_reason: str | None = None

    def to_json(self) -> str:
        payload = asdict(self)
        return json.dumps(payload, indent=2)

    def redacted(self) -> "EvaluationSummary":
        return EvaluationSummary(
            suite=self.suite,
            device=self.device,
            overall_valid=self.overall_valid,
            geometric_mean_family_latency_ms=self.geometric_mean_family_latency_ms,
            worst_family_latency_ms=self.worst_family_latency_ms,
            case_results=[],
            scored_case_count=self.scored_case_count,
            failure_reason=_redact_failure_reason(self.failure_reason),
        )

    @classmethod
    def from_json(cls, payload_json: str) -> "EvaluationSummary":
        payload = json.loads(payload_json)
        case_results = []
        for case_payload in payload["case_results"]:
            validation = ValidationResult(**case_payload["validation"])
            case_results.append(
                CaseResult(
                    case_id=case_payload["case_id"],
                    family=case_payload["family"],
                    latency_ms=case_payload["latency_ms"],
                    density=case_payload["density"],
                    variant=case_payload.get("variant"),
                    validation=validation,
                )
            )
        return cls(
            suite=payload["suite"],
            device=payload["device"],
            overall_valid=payload["overall_valid"],
            geometric_mean_family_latency_ms=payload["geometric_mean_family_latency_ms"],
            worst_family_latency_ms=payload["worst_family_latency_ms"],
            case_results=case_results,
            scored_case_count=payload.get("scored_case_count", len(case_results)),
            failure_reason=payload.get("failure_reason"),
        )


def evaluate_reference_suite(suite_name: str, config: EvaluationConfig) -> EvaluationSummary:
    suite = resolve_suite(suite_name, config.suite_manifest_json)
    return _evaluate_suite(
        entrypoint=reference_block_sparse_attn_fwd,
        suite=suite,
        suite_label=suite_name,
        config=config,
    )


def evaluate_callable(
    entrypoint: EntryPoint,
    suite_name: str,
    config: EvaluationConfig,
    setup: Optional[SetupCallable] = None,
) -> EvaluationSummary:
    suite = resolve_suite(suite_name, config.suite_manifest_json)
    return _evaluate_suite(
        entrypoint=entrypoint,
        suite=suite,
        suite_label=suite_name,
        config=config,
        setup=setup,
    )


def evaluate_submission_dir(
    submission_dir: str,
    suite_name: str,
    config: EvaluationConfig,
) -> EvaluationSummary:
    if config.isolate_submission_process:
        return _evaluate_submission_dir_isolated(submission_dir, suite_name, config)

    suite = resolve_suite(suite_name, config.suite_manifest_json)
    warmup_suite = _build_public_warmup_descriptors(suite)
    cache_context = CompilationCacheMonitor() if config.enforce_post_setup_cache_stability else nullcontext()

    try:
        with cache_context as cache_monitor:
            prepare_compile_runtime_support(_effective_setup_device(config))
            setup_start = time.perf_counter()
            with submission_runtime_guard():
                loaded = load_submission(submission_dir)
                loaded.run_setup(warmup_suite, _effective_setup_device(config))
                _run_setup_warmups(loaded, warmup_suite, config)
            setup_duration = time.perf_counter() - setup_start
            if setup_duration > config.setup_timeout_s:
                return _failed_summary(
                    suite_label=suite_name,
                    config=config,
                    scored_case_count=len(suite),
                    reason=(
                        f"Setup exceeded cap: {setup_duration:.3f}s > {config.setup_timeout_s:.3f}s"
                    ),
                )

            if cache_monitor is not None and config.enforce_post_setup_cache_stability:
                cache_monitor.freeze()

            return _evaluate_suite(
                entrypoint=loaded.entrypoint,
                suite=suite,
                suite_label=suite_name,
                config=config,
                variants=loaded.variants,
                runtime_guarded=True,
                cache_monitor=cache_monitor if isinstance(cache_monitor, CompilationCacheMonitor) else None,
            )
    except Exception as exc:
        return _failed_summary(
            suite_name,
            config,
            f"Setup failed: {exc}",
            scored_case_count=len(suite),
        )


def _evaluate_submission_dir_isolated(
    submission_dir: str,
    suite_name: str,
    config: EvaluationConfig,
) -> EvaluationSummary:
    suite = resolve_suite(suite_name, config.suite_manifest_json)
    warmup_suite = _build_public_warmup_descriptors(suite)
    cache_context = CompilationCacheMonitor() if config.enforce_post_setup_cache_stability else nullcontext()

    try:
        with cache_context as cache_monitor:
            prepare_compile_runtime_support(_effective_setup_device(config))
            setup_start = time.perf_counter()
            with IsolatedSubmissionRunner(submission_dir, config.device) as runner:
                runner.run_setup(
                    warmup_suite,
                    config.setup_warmup_iters,
                    _effective_setup_device(config),
                )
                setup_duration = time.perf_counter() - setup_start
                if setup_duration > config.setup_timeout_s:
                    return _failed_summary(
                        suite_label=suite_name,
                        config=config,
                        scored_case_count=len(suite),
                        reason=(
                            f"Setup exceeded cap: {setup_duration:.3f}s > {config.setup_timeout_s:.3f}s"
                        ),
                    )

                if cache_monitor is not None and config.enforce_post_setup_cache_stability:
                    cache_monitor.freeze()

                return _evaluate_isolated_submission_suite(
                    runner=runner,
                    suite=suite,
                    suite_label=suite_name,
                    config=config,
                    variants=runner.variants,
                    cache_monitor=cache_monitor if isinstance(cache_monitor, CompilationCacheMonitor) else None,
                )
    except Exception as exc:
        return _failed_summary(
            suite_name,
            config,
            f"Setup failed: {exc}",
            scored_case_count=len(suite),
        )


def _evaluate_suite(
    entrypoint: EntryPoint,
    suite: Sequence[CaseSpec],
    suite_label: str,
    config: EvaluationConfig,
    setup: Optional[SetupCallable] = None,
    variants: Sequence[VariantSpec] | None = None,
    runtime_guarded: bool = False,
    cache_monitor: CompilationCacheMonitor | None = None,
) -> EvaluationSummary:
    if setup is not None:
        setup_start = time.perf_counter()
        setup(suite, _effective_setup_device(config))
        setup_duration = time.perf_counter() - setup_start
        if setup_duration > config.setup_timeout_s:
            return _failed_summary(
                suite_label,
                config,
                f"Setup exceeded cap: {setup_duration:.3f}s > {config.setup_timeout_s:.3f}s",
            )
        if cache_monitor is not None and config.enforce_post_setup_cache_stability:
            cache_monitor.freeze()

    case_results: List[CaseResult] = []
    family_latencies: Dict[str, List[float]] = {}
    overall_valid = True
    warmed_runtime_descriptors: set[PublicWarmupKey] = set()

    for case_spec in suite:
        variant_name = None
        if variants is not None:
            try:
                variant_name = find_matching_variant(case_spec, variants).name
            except Exception as exc:
                return _failed_summary(
                    suite_label,
                    config,
                    str(exc),
                    case_results=case_results,
                    scored_case_count=len(suite),
                )

        case = materialize_case(case_spec, device=config.device)

        try:
            if config.correctness_only:
                validation = validate_case_entrypoint(
                    entrypoint,
                    case=case,
                    tolerances=config.tolerances,
                    runtime_guarded=runtime_guarded,
                )
                latency_ms = None
            else:
                runtime_warmups = _build_runtime_warmup_cases(
                    case_spec,
                    config,
                    already_warmed=_public_warmup_key(case_spec) in warmed_runtime_descriptors,
                )
                measure_cases = _build_measure_cases(case_spec, config)
                latency_ms, validation = benchmark_entrypoint(
                    entrypoint,
                    warmup_cases=runtime_warmups,
                    measure_cases=measure_cases,
                    tolerances=config.tolerances,
                    check_correctness=config.check_correctness,
                    runtime_guarded=runtime_guarded,
                    cache_monitor=cache_monitor,
                )
                warmed_runtime_descriptors.add(_public_warmup_key(case_spec))
        except BenchmarkValidationError as exc:
            overall_valid = False
            case_results.append(
                CaseResult(
                    case_id=case_spec.case_id,
                    family=case_spec.family,
                    latency_ms=None if config.correctness_only else float("inf"),
                    density=case.density,
                    variant=variant_name,
                    validation=exc.validation,
                )
            )
            return _failed_summary(
                suite_label,
                config,
                f"Benchmark validation failed for case {case_spec.case_id!r}.",
                case_results=case_results,
                scored_case_count=len(suite),
            )
        except Exception as exc:
            overall_valid = False
            validation = ValidationResult(
                passed=False,
                output_max_abs_diff=float("inf"),
                lse_max_abs_diff=float("inf"),
                message=f"benchmark failed: {exc}",
            )
            case_results.append(
                CaseResult(
                    case_id=case_spec.case_id,
                    family=case_spec.family,
                    latency_ms=None if config.correctness_only else float("inf"),
                    density=case.density,
                    variant=variant_name,
                    validation=validation,
                )
            )
            return _failed_summary(
                suite_label,
                config,
                f"Benchmark for case {case_spec.case_id!r} failed: {exc}",
                case_results=case_results,
                scored_case_count=len(suite),
            )

        if latency_ms is not None:
            family_latencies.setdefault(case_spec.family, []).append(latency_ms)
        case_results.append(
            CaseResult(
                case_id=case_spec.case_id,
                family=case_spec.family,
                latency_ms=latency_ms,
                density=case.density,
                variant=variant_name,
                validation=validation,
            )
        )

    geometric_mean = None
    worst_family = None
    if overall_valid and family_latencies:
        family_medians = {
            family: median(latencies)
            for family, latencies in family_latencies.items()
        }
        geometric_mean = math.exp(
            sum(math.log(value) for value in family_medians.values()) / len(family_medians)
        )
        worst_family = max(family_medians.values())

    return EvaluationSummary(
        suite=suite_label,
        device=config.device,
        overall_valid=overall_valid,
        geometric_mean_family_latency_ms=geometric_mean,
        worst_family_latency_ms=worst_family,
        case_results=case_results,
        scored_case_count=len(suite),
        failure_reason=None if overall_valid else "Correctness validation failed.",
    )


def _evaluate_isolated_submission_suite(
    runner: IsolatedSubmissionRunner,
    suite: Sequence[CaseSpec],
    suite_label: str,
    config: EvaluationConfig,
    variants: Sequence[VariantSpec],
    cache_monitor: CompilationCacheMonitor | None = None,
) -> EvaluationSummary:
    case_results: List[CaseResult] = []
    family_latencies: Dict[str, List[float]] = {}
    overall_valid = True
    warmed_runtime_descriptors: set[PublicWarmupKey] = set()

    for case_spec in suite:
        try:
            variant_name = find_matching_variant(case_spec, variants).name
        except Exception as exc:
            return _failed_summary(
                suite_label,
                config,
                str(exc),
                case_results=case_results,
                scored_case_count=len(suite),
            )

        case = materialize_case(case_spec, device=config.device)

        try:
            if config.correctness_only:
                validation = validate_case_isolated_entrypoint(
                    runner,
                    case=case,
                    tolerances=config.tolerances,
                )
                latency_ms = None
            else:
                warmup_specs = _build_runtime_warmup_specs(
                    case_spec,
                    config,
                    already_warmed=_public_warmup_key(case_spec) in warmed_runtime_descriptors,
                )
                latency_ms, validation = benchmark_isolated_entrypoint(
                    runner,
                    warmup_specs=warmup_specs,
                    measure_cases=_build_measure_cases(case_spec, config),
                    tolerances=config.tolerances,
                    check_correctness=config.check_correctness,
                    cache_monitor=cache_monitor,
                )
                warmed_runtime_descriptors.add(_public_warmup_key(case_spec))
        except BenchmarkValidationError as exc:
            overall_valid = False
            case_results.append(
                CaseResult(
                    case_id=case_spec.case_id,
                    family=case_spec.family,
                    latency_ms=None if config.correctness_only else float("inf"),
                    density=case.density,
                    variant=variant_name,
                    validation=exc.validation,
                )
            )
            return _failed_summary(
                suite_label,
                config,
                f"Benchmark validation failed for case {case_spec.case_id!r}.",
                case_results=case_results,
                scored_case_count=len(suite),
            )
        except Exception as exc:
            overall_valid = False
            validation = ValidationResult(
                passed=False,
                output_max_abs_diff=float("inf"),
                lse_max_abs_diff=float("inf"),
                message=f"benchmark failed: {exc}",
            )
            case_results.append(
                CaseResult(
                    case_id=case_spec.case_id,
                    family=case_spec.family,
                    latency_ms=None if config.correctness_only else float("inf"),
                    density=case.density,
                    variant=variant_name,
                    validation=validation,
                )
            )
            return _failed_summary(
                suite_label,
                config,
                f"Benchmark for case {case_spec.case_id!r} failed: {exc}",
                case_results=case_results,
                scored_case_count=len(suite),
            )

        if latency_ms is not None:
            family_latencies.setdefault(case_spec.family, []).append(latency_ms)
        case_results.append(
            CaseResult(
                case_id=case_spec.case_id,
                family=case_spec.family,
                latency_ms=latency_ms,
                density=case.density,
                variant=variant_name,
                validation=validation,
            )
        )

    geometric_mean = None
    worst_family = None
    if overall_valid and family_latencies:
        family_medians = {
            family: median(latencies)
            for family, latencies in family_latencies.items()
        }
        geometric_mean = math.exp(
            sum(math.log(value) for value in family_medians.values()) / len(family_medians)
        )
        worst_family = max(family_medians.values())

    return EvaluationSummary(
        suite=suite_label,
        device=config.device,
        overall_valid=overall_valid,
        geometric_mean_family_latency_ms=geometric_mean,
        worst_family_latency_ms=worst_family,
        case_results=case_results,
        scored_case_count=len(suite),
        failure_reason=None if overall_valid else "Correctness validation failed.",
    )


class BenchmarkValidationError(RuntimeError):
    def __init__(self, validation: ValidationResult):
        super().__init__(validation.message)
        self.validation = validation


def validate_case_entrypoint(
    entrypoint: EntryPoint,
    case: MaterializedCase,
    tolerances,
    runtime_guarded: bool = False,
) -> ValidationResult:
    candidate_output, candidate_lse = _invoke_candidate(
        entrypoint,
        case,
        runtime_guarded=runtime_guarded,
    )
    _synchronize(case.q.device)
    reference_output, reference_lse = reference_block_sparse_attn_fwd(
        case.q,
        case.k,
        case.v,
        case.row_ptr,
        case.col_idx,
        case.seq_lens,
    )
    validation = validate_outputs(
        candidate_output,
        candidate_lse,
        reference_output,
        reference_lse,
        tolerances,
    )
    if not validation.passed:
        raise BenchmarkValidationError(validation)
    return ValidationResult(
        passed=True,
        output_max_abs_diff=validation.output_max_abs_diff,
        lse_max_abs_diff=validation.lse_max_abs_diff,
        message="ok (local correctness check)",
    )


def validate_case_isolated_entrypoint(
    runner: IsolatedSubmissionRunner,
    case: MaterializedCase,
    tolerances,
) -> ValidationResult:
    timed = runner.run_timed_call(case)
    validation_output, validation_lse = runner.fetch_timed_output(timed.call_index)
    runner.clear_timed_outputs()
    reference_output, reference_lse = reference_block_sparse_attn_fwd(
        case.q,
        case.k,
        case.v,
        case.row_ptr,
        case.col_idx,
        case.seq_lens,
    )
    validation = validate_outputs(
        validation_output,
        validation_lse,
        reference_output,
        reference_lse,
        tolerances,
    )
    if not validation.passed:
        raise BenchmarkValidationError(validation)
    return ValidationResult(
        passed=True,
        output_max_abs_diff=validation.output_max_abs_diff,
        lse_max_abs_diff=validation.lse_max_abs_diff,
        message="ok (local correctness check)",
    )


def benchmark_entrypoint(
    entrypoint: EntryPoint,
    warmup_cases: Sequence[MaterializedCase],
    measure_cases: Sequence[MaterializedCase],
    tolerances,
    check_correctness: bool,
    runtime_guarded: bool = False,
    cache_monitor: CompilationCacheMonitor | None = None,
) -> tuple[float, ValidationResult]:
    for case in warmup_cases:
        _invoke_candidate(entrypoint, case, runtime_guarded=runtime_guarded)
        _synchronize(case.q.device)
        if cache_monitor is not None:
            cache_monitor.assert_unchanged(f"warmup for case {case.spec.case_id}")

    if not measure_cases:
        raise ValueError("measure_iters must be at least 1.")

    latencies = []
    validation_case = None
    validation_output = None
    validation_lse = None
    validation_index = _validation_case_index(measure_cases)
    for index, case in enumerate(measure_cases):
        start = time.perf_counter()
        candidate_output, candidate_lse = _invoke_candidate(
            entrypoint,
            case,
            runtime_guarded=runtime_guarded,
        )
        _synchronize(case.q.device)
        latencies.append((time.perf_counter() - start) * 1000.0)
        if cache_monitor is not None:
            cache_monitor.assert_unchanged(f"timed run for case {case.spec.case_id}")
        if check_correctness and index == validation_index:
            validation_case = case
            validation_output = candidate_output
            validation_lse = candidate_lse

    if not check_correctness:
        return (
            float(median(latencies)),
            ValidationResult(
                passed=True,
                output_max_abs_diff=0.0,
                lse_max_abs_diff=0.0,
                message="skipped (use --check-correctness)",
            ),
        )

    if validation_case is None or validation_output is None or validation_lse is None:
        raise ValueError("Expected one measured iteration to be selected for correctness validation.")

    reference_output, reference_lse = reference_block_sparse_attn_fwd(
        validation_case.q,
        validation_case.k,
        validation_case.v,
        validation_case.row_ptr,
        validation_case.col_idx,
        validation_case.seq_lens,
    )
    validation = validate_outputs(
        validation_output,
        validation_lse,
        reference_output,
        reference_lse,
        tolerances,
    )
    if not validation.passed:
        raise BenchmarkValidationError(validation)

    return (
        float(median(latencies)),
        ValidationResult(
            passed=True,
            output_max_abs_diff=validation.output_max_abs_diff,
            lse_max_abs_diff=validation.lse_max_abs_diff,
            message=f"ok (checked 1 of {len(measure_cases)})",
        ),
    )


def benchmark_isolated_entrypoint(
    runner: IsolatedSubmissionRunner,
    warmup_specs: Sequence[CaseSpec],
    measure_cases: Sequence[MaterializedCase],
    tolerances,
    check_correctness: bool,
    cache_monitor: CompilationCacheMonitor | None = None,
) -> tuple[float, ValidationResult]:
    if warmup_specs:
        runner.run_public_warmups(warmup_specs)
        if cache_monitor is not None:
            for case_spec in warmup_specs:
                cache_monitor.assert_unchanged(f"warmup for case {case_spec.case_id}")

    if not measure_cases:
        raise ValueError("measure_iters must be at least 1.")

    latencies = []
    validation_call_index = _validation_case_index(measure_cases)
    timed_call_index = None
    for case in measure_cases:
        timed = runner.run_timed_call(case)
        latencies.append(timed.latency_ms)
        if cache_monitor is not None:
            cache_monitor.assert_unchanged(f"timed run for case {case.spec.case_id}")
        if len(latencies) - 1 == validation_call_index:
            timed_call_index = timed.call_index

    if not check_correctness:
        runner.clear_timed_outputs()
        return (
            float(median(latencies)),
            ValidationResult(
                passed=True,
                output_max_abs_diff=0.0,
                lse_max_abs_diff=0.0,
                message="skipped (use --check-correctness)",
            ),
        )

    if timed_call_index is None:
        raise ValueError("Expected one measured iteration to be selected for correctness validation.")

    validation_case = measure_cases[validation_call_index]
    validation_output, validation_lse = runner.fetch_timed_output(timed_call_index)
    runner.clear_timed_outputs()

    reference_output, reference_lse = reference_block_sparse_attn_fwd(
        validation_case.q,
        validation_case.k,
        validation_case.v,
        validation_case.row_ptr,
        validation_case.col_idx,
        validation_case.seq_lens,
    )
    validation = validate_outputs(
        validation_output,
        validation_lse,
        reference_output,
        reference_lse,
        tolerances,
    )
    if not validation.passed:
        raise BenchmarkValidationError(validation)

    return (
        float(median(latencies)),
        ValidationResult(
            passed=True,
            output_max_abs_diff=validation.output_max_abs_diff,
            lse_max_abs_diff=validation.lse_max_abs_diff,
            message=f"ok (checked 1 of {len(measure_cases)})",
        ),
    )


def _invoke_candidate(
    entrypoint: EntryPoint,
    case: MaterializedCase,
    runtime_guarded: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if runtime_guarded:
        with submission_runtime_guard():
            return entrypoint(case.q, case.k, case.v, case.row_ptr, case.col_idx, case.seq_lens)
    return entrypoint(case.q, case.k, case.v, case.row_ptr, case.col_idx, case.seq_lens)


def _run_setup_warmups(
    loaded: LoadedSubmission,
    suite: Sequence[CaseSpec],
    config: EvaluationConfig,
) -> None:
    for case_spec in suite:
        case = materialize_case(case_spec, device=config.device)
        for _ in range(config.setup_warmup_iters):
            loaded.entrypoint(case.q, case.k, case.v, case.row_ptr, case.col_idx, case.seq_lens)
            _synchronize(case.q.device)


def _failed_summary(
    suite_label: str,
    config: EvaluationConfig,
    reason: str,
    case_results: Sequence[CaseResult] | None = None,
    scored_case_count: int | None = None,
) -> EvaluationSummary:
    return EvaluationSummary(
        suite=suite_label,
        device=config.device,
        overall_valid=False,
        geometric_mean_family_latency_ms=None,
        worst_family_latency_ms=None,
        case_results=list(case_results or []),
        scored_case_count=scored_case_count,
        failure_reason=reason,
    )


def _effective_setup_device(config: EvaluationConfig) -> str:
    return config.setup_device or config.device


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()


def _build_public_warmup_descriptors(suite: Sequence[CaseSpec]) -> list[CaseSpec]:
    descriptors: list[CaseSpec] = []
    seen_keys: set[PublicWarmupKey] = set()
    for case in suite:
        key = _public_warmup_key(case)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        descriptors.append(
            _canonical_public_warmup_case(
                case,
                case_id=f"warmup-key-{len(descriptors) + 1}",
                profile="setup",
                seed=0,
            )
        )
    return descriptors


def _build_runtime_warmup_cases(
    case_spec: CaseSpec,
    config: EvaluationConfig,
    already_warmed: bool,
) -> list[MaterializedCase]:
    if already_warmed:
        return []

    base_spec = _canonical_public_warmup_case(
        case_spec,
        case_id=f"{case_spec.case_id}-runtime-warmup",
        profile="warmup",
        seed=0,
    )
    return [
        materialize_case(
            _seeded_public_warmup_case_spec(base_spec, phase="warmup", index=index),
            device=config.device,
        )
        for index in range(config.warmup_iters)
    ]


def _build_runtime_warmup_specs(
    case_spec: CaseSpec,
    config: EvaluationConfig,
    already_warmed: bool,
) -> list[CaseSpec]:
    if already_warmed:
        return []

    base_spec = _canonical_public_warmup_case(
        case_spec,
        case_id=f"{case_spec.case_id}-runtime-warmup",
        profile="warmup",
        seed=0,
    )
    return [
        _seeded_public_warmup_case_spec(base_spec, phase="warmup", index=index)
        for index in range(config.warmup_iters)
    ]


def _build_measure_cases(
    case_spec: CaseSpec,
    config: EvaluationConfig,
) -> list[MaterializedCase]:
    return [
        materialize_case(
            _derived_case_spec(case_spec, phase="measure", index=index),
            device=config.device,
        )
        for index in range(config.measure_iters)
    ]


def _public_warmup_key(case: CaseSpec) -> PublicWarmupKey:
    return (
        case.family,
        case.batch_size,
        case.num_heads,
        case.t_max,
        case.window_blocks,
        case.global_blocks,
        case.retrieval_blocks,
    )


def _canonical_public_warmup_case(
    case: CaseSpec,
    case_id: str,
    profile: str,
    seed: int,
) -> CaseSpec:
    return CaseSpec(
        case_id=case_id,
        family=case.family,
        batch_size=case.batch_size,
        num_heads=case.num_heads,
        t_max=case.t_max,
        window_blocks=case.window_blocks,
        global_blocks=case.global_blocks,
        retrieval_blocks=case.retrieval_blocks,
        retrieval_local_bias=0.5 if case.family == "sliding_window_retrieval" else 0.0,
        seq_len_min_ratio=1.0,
        seed=seed,
        profile=profile,
    )


def _seeded_public_warmup_case_spec(case_spec: CaseSpec, phase: str, index: int) -> CaseSpec:
    key_payload = ":".join(str(part) for part in _public_warmup_key(case_spec))
    return CaseSpec(
        case_id=f"{case_spec.case_id}-{phase}-{index + 1}",
        family=case_spec.family,
        batch_size=case_spec.batch_size,
        num_heads=case_spec.num_heads,
        t_max=case_spec.t_max,
        window_blocks=case_spec.window_blocks,
        global_blocks=case_spec.global_blocks,
        retrieval_blocks=case_spec.retrieval_blocks,
        retrieval_local_bias=case_spec.retrieval_local_bias,
        seq_len_min_ratio=case_spec.seq_len_min_ratio,
        seed=_derived_seed(0, f"{phase}:{index}:{key_payload}"),
        profile=case_spec.profile,
    )


def _derived_case_spec(case_spec: CaseSpec, phase: str, index: int) -> CaseSpec:
    return CaseSpec(
        case_id=f"{case_spec.case_id}-{phase}-{index + 1}",
        family=case_spec.family,
        batch_size=case_spec.batch_size,
        num_heads=case_spec.num_heads,
        t_max=case_spec.t_max,
        window_blocks=case_spec.window_blocks,
        global_blocks=case_spec.global_blocks,
        retrieval_blocks=case_spec.retrieval_blocks,
        retrieval_local_bias=case_spec.retrieval_local_bias,
        seq_len_min_ratio=case_spec.seq_len_min_ratio,
        seed=_derived_seed(case_spec.seed, f"{phase}:{index}"),
        profile=case_spec.profile,
    )


def _derived_seed(base_seed: int, salt: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{salt}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**31 - 1)


def _redact_failure_reason(reason: str | None) -> str | None:
    if reason is None:
        return None

    lowered = reason.lower()
    if reason.startswith("Setup exceeded cap"):
        return "Setup exceeded cap."
    if reason.startswith("Setup failed"):
        return "Setup failed."
    if reason.startswith("Benchmark for case") or reason.startswith("Benchmark validation failed"):
        return "Benchmark failed during evaluation."
    if reason.startswith("Case "):
        return "Evaluation failed on a hidden case."
    if "cache mutation" in lowered:
        return "Post-setup cache mutation detected."
    if "candidate raised" in lowered or reason == "Correctness validation failed.":
        return "Correctness validation failed."
    if reason.startswith("Remote evaluation failed"):
        return "Remote evaluation failed."
    return "Evaluation failed."


def _validation_case_index(measure_cases: Sequence[MaterializedCase]) -> int:
    if not measure_cases:
        raise ValueError("measure_iters must be at least 1.")
    case_spec = measure_cases[0].spec
    return _derived_seed(case_spec.seed, "validate") % len(measure_cases)
