from __future__ import annotations

from dataclasses import dataclass

import torch

from .spec import Tolerances


@dataclass
class ValidationResult:
    passed: bool
    output_max_abs_diff: float
    lse_max_abs_diff: float
    message: str


def validate_outputs(
    candidate_output: torch.Tensor,
    candidate_lse: torch.Tensor,
    reference_output: torch.Tensor,
    reference_lse: torch.Tensor,
    tolerances: Tolerances,
) -> ValidationResult:
    output_passed = torch.allclose(
        candidate_output.to(torch.float32),
        reference_output.to(torch.float32),
        atol=tolerances.output_atol,
        rtol=tolerances.output_rtol,
    )

    lse_passed = _allclose_with_infinities(
        candidate_lse,
        reference_lse,
        atol=tolerances.lse_atol,
        rtol=tolerances.lse_rtol,
    )

    output_diff = _max_abs_diff(candidate_output.to(torch.float32), reference_output.to(torch.float32))
    lse_diff = _max_abs_diff(candidate_lse, reference_lse)

    passed = bool(output_passed and lse_passed)
    if passed:
        message = "ok"
    else:
        message = (
            "validation failed: "
            f"output_max_abs_diff={output_diff:.6g}, "
            f"lse_max_abs_diff={lse_diff:.6g}"
        )

    return ValidationResult(
        passed=passed,
        output_max_abs_diff=output_diff,
        lse_max_abs_diff=lse_diff,
        message=message,
    )


def _allclose_with_infinities(a: torch.Tensor, b: torch.Tensor, atol: float, rtol: float) -> bool:
    same_infinities = torch.eq(torch.isinf(a), torch.isinf(b))
    same_inf_sign = torch.eq(torch.signbit(a), torch.signbit(b)) | ~torch.isinf(a)
    finite_mask = torch.isfinite(a) & torch.isfinite(b)
    finite_close = torch.allclose(a[finite_mask], b[finite_mask], atol=atol, rtol=rtol) if torch.any(finite_mask) else True
    return bool(torch.all(same_infinities) and torch.all(same_inf_sign) and finite_close)


def _max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    finite_mask = torch.isfinite(a) & torch.isfinite(b)
    max_diff = 0.0
    if torch.any(finite_mask):
        max_diff = float(torch.max(torch.abs(a[finite_mask] - b[finite_mask])).item())
    mismatch_mask = torch.logical_xor(torch.isfinite(a), torch.isfinite(b))
    if torch.any(mismatch_mask):
        max_diff = float("inf")
    return max_diff
