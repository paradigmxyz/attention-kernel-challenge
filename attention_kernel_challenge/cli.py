from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import torch

from .backends import (
    NoBackendConfiguredError,
    deploy_modal_app,
    modal_cli_available,
    modal_deployment_current,
    modal_profile_current,
    resolve_backend_name,
    run_modal_reference_eval,
    run_modal_submission_eval,
)
from .cases import (
    build_public_suite_metadata,
    build_suite,
    build_suite_from_manifest_json,
    build_suite_from_manifest_path,
    is_public_distribution_suite,
)
from .config import HarnessConfig, ModalBackendConfig, clear_config, load_config, repo_root_from_file, save_config
from .evaluator import EvaluationSummary, evaluate_reference_suite, evaluate_submission_dir
from .sandbox import run_python_module
from .spec import BUILTIN_SUITE_NAMES, EvaluationConfig
from .submission_loader import pack_submission_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local harness for the Attention Kernel Challenge.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    show_parser = subparsers.add_parser("show-suite", help="Show suite metadata.")
    _add_suite_arguments(show_parser)

    eval_parser = subparsers.add_parser("eval-reference", help="Run the reference implementation through a selected backend.")
    _add_suite_arguments(eval_parser)
    eval_parser.add_argument("--backend", choices=["local", "modal"])
    eval_parser.add_argument("--device", default="cpu")
    eval_parser.add_argument("--warmup-iters", type=int, default=1)
    eval_parser.add_argument("--measure-iters", type=int, default=3)
    eval_parser.add_argument("--setup-timeout-s", type=float, default=30.0)
    eval_parser.add_argument(
        "--modal-one-shot",
        action="store_true",
        help="Ignore any deployed Modal evaluator and launch a transient app for this run.",
    )
    eval_parser.add_argument("--sandbox", action="store_true")
    eval_parser.add_argument("--redact-case-details", action="store_true")
    eval_parser.add_argument("--emit-json", action="store_true")

    submission_parser = subparsers.add_parser("eval-submission", help="Run a submission through a selected backend.")
    submission_parser.add_argument("--submission-dir", required=True)
    _add_suite_arguments(submission_parser)
    submission_parser.add_argument("--backend", choices=["local", "modal"])
    submission_parser.add_argument("--device", default="cpu")
    submission_parser.add_argument("--warmup-iters", type=int, default=1)
    submission_parser.add_argument("--measure-iters", type=int, default=3)
    submission_parser.add_argument("--setup-timeout-s", type=float, default=30.0)
    submission_parser.add_argument(
        "--modal-one-shot",
        action="store_true",
        help="Ignore any deployed Modal evaluator and launch a transient app for this run.",
    )
    submission_parser.add_argument(
        "--serverlike",
        action="store_true",
        help="Use isolated local submission execution and report setup device as cuda to catch server-only setup surprises.",
    )
    submission_parser.add_argument("--sandbox", action="store_true")
    submission_parser.add_argument("--redact-case-details", action="store_true")
    submission_parser.add_argument("--emit-json", action="store_true")

    doctor_parser = subparsers.add_parser("doctor", help="Inspect local setup and backend readiness.")
    doctor_parser.add_argument("--probe-modal", action="store_true")

    backend_parser = subparsers.add_parser("backend", help="Configure or inspect evaluation backends.")
    backend_subparsers = backend_parser.add_subparsers(dest="backend_command", required=True)

    backend_status_parser = backend_subparsers.add_parser("status", help="Show backend configuration and availability.")
    backend_status_parser.add_argument("--probe-modal", action="store_true")

    use_modal_parser = backend_subparsers.add_parser("use-modal", help="Persist Modal as the default backend.")
    use_modal_parser.add_argument("--gpu", default="H100!:1")
    use_modal_parser.add_argument("--timeout-s", type=int, default=120)
    use_modal_parser.add_argument("--python-version", default="3.11")

    setup_modal_parser = backend_subparsers.add_parser(
        "setup-modal",
        help="Persist Modal as the default backend and deploy the evaluator app in one step.",
    )
    setup_modal_parser.add_argument("--gpu", default="H100!:1")
    setup_modal_parser.add_argument("--timeout-s", type=int, default=120)
    setup_modal_parser.add_argument("--python-version", default="3.11")
    setup_modal_parser.add_argument(
        "--app-ref",
        default="attention_kernel_challenge.modal_backend",
        help="Python module path for the Modal app.",
    )

    deploy_modal_parser = backend_subparsers.add_parser(
        "deploy-modal",
        help="Deploy the Modal evaluator app for lower-overhead remote invocations.",
    )
    deploy_modal_parser.add_argument(
        "--app-ref",
        default="attention_kernel_challenge.modal_backend",
        help="Python module path for the Modal app.",
    )

    backend_subparsers.add_parser("use-local", help="Persist local execution as the default backend.")
    backend_subparsers.add_parser("clear", help="Remove persisted backend configuration.")

    internal_parser = subparsers.add_parser("_eval-reference-internal", help=argparse.SUPPRESS)
    _add_suite_arguments(internal_parser, required=True)
    internal_parser.add_argument("--device", default="cpu")
    internal_parser.add_argument("--warmup-iters", type=int, default=1)
    internal_parser.add_argument("--measure-iters", type=int, default=3)
    internal_parser.add_argument("--setup-timeout-s", type=float, default=30.0)
    internal_parser.add_argument("--redact-case-details", action="store_true")
    internal_parser.add_argument("--emit-json", action="store_true")

    internal_submission_parser = subparsers.add_parser("_eval-submission-internal", help=argparse.SUPPRESS)
    internal_submission_parser.add_argument("--submission-dir", required=True)
    _add_suite_arguments(internal_submission_parser, required=True)
    internal_submission_parser.add_argument("--device", default="cpu")
    internal_submission_parser.add_argument("--warmup-iters", type=int, default=1)
    internal_submission_parser.add_argument("--measure-iters", type=int, default=3)
    internal_submission_parser.add_argument("--setup-timeout-s", type=float, default=30.0)
    internal_submission_parser.add_argument("--serverlike", action="store_true")
    internal_submission_parser.add_argument("--redact-case-details", action="store_true")
    internal_submission_parser.add_argument("--emit-json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    repo_root = repo_root_from_file(__file__)
    config = load_config(repo_root)

    if args.command == "show-suite":
        try:
            suite_name, suite_manifest_json = _resolve_suite_inputs(args)
            if suite_manifest_json is not None:
                suite = build_suite_from_manifest_json(suite_manifest_json)
                payload = [asdict(case) for case in suite]
            elif is_public_distribution_suite(suite_name):
                payload = build_public_suite_metadata(suite_name)
            else:
                suite = build_suite(suite_name)
                payload = [asdict(case) for case in suite]
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(json.dumps(payload, indent=2))
        return 0

    if args.command == "backend":
        return _handle_backend_command(args, repo_root, config)

    if args.command == "doctor":
        return _run_doctor(config=config, repo_root=repo_root, probe_modal=args.probe_modal)

    if args.command == "eval-reference":
        suite_name, suite_manifest_json = _resolve_suite_inputs(args)
        try:
            backend_name = resolve_backend_name(config, args.backend)
        except NoBackendConfiguredError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1

        if backend_name == "modal":
            if args.setup_timeout_s != 30.0:
                print("error: --setup-timeout-s is only supported with --backend local.", file=sys.stderr)
                return 1
            if args.sandbox:
                print("error: --sandbox is only supported with --backend local.", file=sys.stderr)
                return 1
            try:
                modal_config = config.modal if config is not None else ModalBackendConfig()
                payload_json = run_modal_reference_eval(
                    repo_root=repo_root,
                    suite=suite_name,
                    suite_manifest_json=suite_manifest_json,
                    warmup_iters=args.warmup_iters,
                    measure_iters=args.measure_iters,
                    modal_config=modal_config,
                    prefer_deployed=not args.modal_one_shot,
                )
            except RuntimeError as exc:
                print(f"error: {exc}", file=sys.stderr)
                return 1
            if args.emit_json:
                summary = EvaluationSummary.from_json(payload_json)
                if args.redact_case_details:
                    summary = summary.redacted()
                print(summary.to_json())
            else:
                summary = EvaluationSummary.from_json(payload_json)
                if args.redact_case_details:
                    summary = summary.redacted()
                _print_summary(summary)
            return 0

        if args.sandbox:
            try:
                sandbox_result = run_python_module(
                    module="attention_kernel_challenge.cli",
                    module_args=[
                        "_eval-reference-internal",
                        *(_suite_args_for_subprocess(args)),
                        "--device",
                        args.device,
                        "--warmup-iters",
                        str(args.warmup_iters),
                        "--measure-iters",
                        str(args.measure_iters),
                        "--setup-timeout-s",
                        str(args.setup_timeout_s),
                        *(["--redact-case-details"] if args.redact_case_details else []),
                        *(["--emit-json"] if args.emit_json else []),
                    ],
                    repo_root=repo_root,
                )
            except RuntimeError as exc:
                print(f"error: {exc}", file=sys.stderr)
                return 1
            if sandbox_result.stdout:
                print(sandbox_result.stdout, end="")
            if sandbox_result.stderr:
                print(sandbox_result.stderr, end="", file=sys.stderr)
            return sandbox_result.returncode

        return _run_reference(
            suite_name=suite_name,
            suite_manifest_json=suite_manifest_json,
            device=args.device,
            warmup_iters=args.warmup_iters,
            measure_iters=args.measure_iters,
            setup_timeout_s=args.setup_timeout_s,
            check_correctness=True,
            correctness_only=_local_correctness_only(args.device),
            redact_case_details=args.redact_case_details,
            emit_json=args.emit_json,
        )

    if args.command == "eval-submission":
        suite_name, suite_manifest_json = _resolve_suite_inputs(args)
        try:
            backend_name = resolve_backend_name(config, args.backend)
        except NoBackendConfiguredError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1

        if backend_name == "modal":
            if args.setup_timeout_s != 30.0:
                print("error: --setup-timeout-s is only supported with --backend local.", file=sys.stderr)
                return 1
            if args.sandbox:
                print("error: --sandbox is only supported with --backend local.", file=sys.stderr)
                return 1
            try:
                modal_config = config.modal if config is not None else ModalBackendConfig()
                archive = pack_submission_dir(args.submission_dir)
                payload_json = run_modal_submission_eval(
                    suite=suite_name,
                    suite_manifest_json=suite_manifest_json,
                    submission_archive=archive,
                    warmup_iters=args.warmup_iters,
                    measure_iters=args.measure_iters,
                    modal_config=modal_config,
                    prefer_deployed=not args.modal_one_shot,
                )
            except Exception as exc:
                print(f"error: {exc}", file=sys.stderr)
                return 1
            if args.emit_json:
                summary = EvaluationSummary.from_json(payload_json)
                if args.redact_case_details:
                    summary = summary.redacted()
                print(summary.to_json())
            else:
                summary = EvaluationSummary.from_json(payload_json)
                if args.redact_case_details:
                    summary = summary.redacted()
                _print_summary(summary)
            return 0

        if args.sandbox:
            try:
                sandbox_result = run_python_module(
                    module="attention_kernel_challenge.cli",
                    module_args=[
                        "_eval-submission-internal",
                        "--submission-dir",
                        args.submission_dir,
                        *(_suite_args_for_subprocess(args)),
                        "--device",
                        args.device,
                        "--warmup-iters",
                        str(args.warmup_iters),
                        "--measure-iters",
                        str(args.measure_iters),
                        "--setup-timeout-s",
                        str(args.setup_timeout_s),
                        *(["--serverlike"] if args.serverlike else []),
                        *(["--redact-case-details"] if args.redact_case_details else []),
                        *(["--emit-json"] if args.emit_json else []),
                    ],
                    repo_root=repo_root,
                )
            except RuntimeError as exc:
                print(f"error: {exc}", file=sys.stderr)
                return 1
            if sandbox_result.stdout:
                print(sandbox_result.stdout, end="")
            if sandbox_result.stderr:
                print(sandbox_result.stderr, end="", file=sys.stderr)
            return sandbox_result.returncode

        return _run_submission(
            submission_dir=args.submission_dir,
            suite_name=suite_name,
            suite_manifest_json=suite_manifest_json,
            device=args.device,
            warmup_iters=args.warmup_iters,
            measure_iters=args.measure_iters,
            setup_timeout_s=args.setup_timeout_s,
            check_correctness=True,
            correctness_only=_local_correctness_only(args.device),
            serverlike=args.serverlike,
            redact_case_details=args.redact_case_details,
            emit_json=args.emit_json,
        )

    if args.command == "_eval-reference-internal":
        suite_name, suite_manifest_json = _resolve_suite_inputs(args)
        return _run_reference(
            suite_name=suite_name,
            suite_manifest_json=suite_manifest_json,
            device=args.device,
            warmup_iters=args.warmup_iters,
            measure_iters=args.measure_iters,
            setup_timeout_s=args.setup_timeout_s,
            check_correctness=True,
            correctness_only=_local_correctness_only(args.device),
            redact_case_details=args.redact_case_details,
            emit_json=args.emit_json,
        )

    if args.command == "_eval-submission-internal":
        suite_name, suite_manifest_json = _resolve_suite_inputs(args)
        return _run_submission(
            submission_dir=args.submission_dir,
            suite_name=suite_name,
            suite_manifest_json=suite_manifest_json,
            device=args.device,
            warmup_iters=args.warmup_iters,
            measure_iters=args.measure_iters,
            setup_timeout_s=args.setup_timeout_s,
            check_correctness=True,
            correctness_only=_local_correctness_only(args.device),
            serverlike=args.serverlike,
            redact_case_details=args.redact_case_details,
            emit_json=args.emit_json,
        )

    raise RuntimeError(f"Unhandled command: {args.command}")


def _handle_backend_command(args, repo_root: Path, config: HarnessConfig | None) -> int:
    if args.backend_command == "status":
        print(f"Config file: {repo_root / '.attention_kernel_challenge' / 'backend.json'}")
        print(f"Configured backend: {config.default_backend if config else 'none'}")
        if config and config.default_backend == "modal":
            print(
                "Configured Modal settings: "
                f"gpu={config.modal.gpu}, timeout_s={config.modal.timeout_s}, "
                f"python_version={config.modal.python_version}"
            )
        print(f"Modal CLI available: {modal_cli_available()}")
        if args.probe_modal:
            ok, profile_text = modal_profile_current()
            print(f"Modal profile ready: {ok}")
            print(f"Modal profile: {profile_text}")
            deploy_ok, deploy_text = modal_deployment_current()
            print(f"Deployed evaluator ready: {deploy_ok}")
            print(f"Deployed evaluator: {deploy_text}")
        else:
            print("Modal profile check: skipped (pass --probe-modal to verify auth/readiness and deployed evaluator)")
        return 0

    if args.backend_command == "use-modal":
        path = save_config(
            repo_root,
            HarnessConfig(
                default_backend="modal",
                modal=ModalBackendConfig(
                    gpu=args.gpu,
                    timeout_s=args.timeout_s,
                    python_version=args.python_version,
                ),
            ),
        )
        print(f"Saved backend config to {path}")
        return 0

    if args.backend_command == "setup-modal":
        modal_config = ModalBackendConfig(
            gpu=args.gpu,
            timeout_s=args.timeout_s,
            python_version=args.python_version,
        )
        try:
            output = deploy_modal_app(args.app_ref, modal_config=modal_config)
        except RuntimeError as exc:
            print(f"error: {exc}", file=sys.stderr)
            print("Install/authenticate Modal in this Python environment first:", file=sys.stderr)
            print("  python -m pip install modal", file=sys.stderr)
            print("  python -m modal setup", file=sys.stderr)
            return 1
        path = save_config(
            repo_root,
            HarnessConfig(
                default_backend="modal",
                modal=modal_config,
            ),
        )
        print(f"Saved backend config to {path}")
        if output:
            print(output)
        print("Modal backend is configured and the evaluator deployment step has completed.")
        print("Next check: python -m attention_kernel_challenge.cli backend status --probe-modal")
        return 0

    if args.backend_command == "deploy-modal":
        modal_config = config.modal if config is not None else ModalBackendConfig()
        try:
            output = deploy_modal_app(args.app_ref, modal_config=modal_config)
        except RuntimeError as exc:
            print(f"error: {exc}", file=sys.stderr)
            print("Install/authenticate Modal in this Python environment first:", file=sys.stderr)
            print("  python -m pip install modal", file=sys.stderr)
            print("  python -m modal setup", file=sys.stderr)
            return 1
        if output:
            print(output)
        return 0

    if args.backend_command == "use-local":
        path = save_config(repo_root, HarnessConfig(default_backend="local"))
        print(f"Saved backend config to {path}")
        return 0

    if args.backend_command == "clear":
        clear_config(repo_root)
        print("Cleared backend config.")
        return 0

    raise RuntimeError(f"Unhandled backend command: {args.backend_command}")


def _run_doctor(config: HarnessConfig | None, repo_root: Path, probe_modal: bool) -> int:
    python_version = sys.version.split()[0]
    print(f"Repo root: {repo_root}")
    print(f"Python: {python_version}")
    if not python_version.startswith("3.11."):
        print("Python note: expected local tooling on 3.11.x; recreate `.venv` with `uv venv --python 3.11 --system-site-packages .venv` if needed.")
    print(f"torch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
    else:
        print("Local note: CPU/local runs are useful for correctness only; H100 latency requires Modal or another remote backend.")

    print(f"Configured backend: {config.default_backend if config else 'none'}")
    print(f"Modal CLI available: {modal_cli_available()}")
    if probe_modal:
        ok, profile_text = modal_profile_current()
        print(f"Modal profile ready: {ok}")
        print(f"Modal profile: {profile_text}")
    else:
        print("Modal profile check: skipped (pass --probe-modal for a live auth/readiness and deploy check)")

    try:
        import modal  # noqa: F401

        print("Python package 'modal': available")
    except Exception:
        print("Python package 'modal': missing")

    print("Recommended fast loop:")
    print("  1. Use local CPU runs for correctness; local GPU runs are useful for unofficial timing.")
    print("  2. Use quick for cheap remote H100 sanity checks.")
    print("  3. Use full for more serious H100 iteration.")
    print("  4. Modal-backed runs always include correctness checking.")
    print("  5. Use broad more sparingly for wider coverage.")
    print("  6. Use local `eval-submission --serverlike` to catch some setup/import surprises before Modal.")
    print("  7. Prefer `backend setup-modal` for the one-command Modal setup path.")
    return 0


def _run_reference(
    suite_name: str,
    suite_manifest_json: str | None,
    device: str,
    warmup_iters: int,
    measure_iters: int,
    setup_timeout_s: float,
    check_correctness: bool,
    correctness_only: bool,
    redact_case_details: bool,
    emit_json: bool,
) -> int:
    config = EvaluationConfig(
        device=device,
        warmup_iters=warmup_iters,
        measure_iters=measure_iters,
        check_correctness=check_correctness,
        correctness_only=correctness_only,
        setup_timeout_s=setup_timeout_s,
        suite_manifest_json=suite_manifest_json,
    )
    try:
        summary = evaluate_reference_suite(suite_name, config)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if redact_case_details:
        summary = summary.redacted()
    if emit_json:
        print(summary.to_json())
    else:
        _print_summary(summary)
    return 0 if summary.overall_valid else 1


def _run_submission(
    submission_dir: str,
    suite_name: str,
    suite_manifest_json: str | None,
    device: str,
    warmup_iters: int,
    measure_iters: int,
    setup_timeout_s: float,
    check_correctness: bool,
    correctness_only: bool,
    serverlike: bool,
    redact_case_details: bool,
    emit_json: bool,
) -> int:
    config = EvaluationConfig(
        device=device,
        setup_device=_local_setup_device(serverlike),
        warmup_iters=warmup_iters,
        measure_iters=measure_iters,
        check_correctness=check_correctness,
        correctness_only=correctness_only,
        setup_timeout_s=setup_timeout_s,
        isolate_submission_process=serverlike,
        suite_manifest_json=suite_manifest_json,
    )
    try:
        summary = evaluate_submission_dir(submission_dir, suite_name, config)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if redact_case_details:
        summary = summary.redacted()
    if emit_json:
        print(summary.to_json())
    else:
        _print_summary(summary)
    return 0 if summary.overall_valid else 1


def _print_summary(summary) -> None:
    print(f"Suite: {summary.suite}")
    print(f"Device: {summary.device}")
    print(f"Overall valid: {summary.overall_valid}")
    if summary.case_results:
        correctness_checked = any(not case.validation.message.startswith("skipped") for case in summary.case_results)
        print(f"Correctness checked: {correctness_checked}")
    if summary.scored_case_count is not None:
        print(f"Scored cases: {summary.scored_case_count}")
    if summary.failure_reason:
        print(f"Failure reason: {summary.failure_reason}")
    if summary.geometric_mean_family_latency_ms is not None:
        print(f"Geometric mean family latency (ms): {summary.geometric_mean_family_latency_ms:.3f}")
    if summary.worst_family_latency_ms is not None:
        print(f"Worst family latency (ms): {summary.worst_family_latency_ms:.3f}")
    print("Cases:")
    for case in summary.case_results:
        parts = [
            f"  - {case.case_id}: family={case.family}",
            f"density={case.density:.4f}",
            f"variant={case.variant or 'n/a'}",
            f"validation={case.validation.message}",
        ]
        if case.latency_ms is not None:
            parts.insert(1, f"latency_ms={case.latency_ms:.3f}")
        print(", ".join(parts))


def _add_suite_arguments(parser: argparse.ArgumentParser, required: bool = False) -> None:
    group = parser.add_mutually_exclusive_group(required=required)
    group.add_argument("--suite", default=None if required else "smoke", metavar="SUITE")
    group.add_argument("--suite-manifest")
    group.add_argument("--suite-manifest-json", help=argparse.SUPPRESS)


def _resolve_suite_inputs(args) -> tuple[str, str | None]:
    if getattr(args, "suite_manifest_json", None):
        return "manifest", args.suite_manifest_json
    if getattr(args, "suite_manifest", None):
        manifest_path = Path(args.suite_manifest)
        return manifest_path.stem, manifest_path.read_text()
    return args.suite, None


def _suite_args_for_subprocess(args) -> list[str]:
    if getattr(args, "suite_manifest", None):
        return ["--suite-manifest-json", Path(args.suite_manifest).read_text()]
    if getattr(args, "suite_manifest_json", None):
        return ["--suite-manifest-json", args.suite_manifest_json]
    return ["--suite", args.suite]


def _local_correctness_only(device: str) -> bool:
    return device == "cpu"


def _local_setup_device(serverlike: bool) -> str | None:
    if serverlike:
        return "cuda"
    return None


if __name__ == "__main__":
    raise SystemExit(main())
