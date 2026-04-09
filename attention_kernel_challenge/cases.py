from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch

from .spec import BLOCK_SIZE, BUILTIN_SUITE_NAMES, HEAD_DIM, CaseSpec


@dataclass
class MaterializedCase:
    spec: CaseSpec
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    row_ptr: torch.Tensor
    col_idx: torch.Tensor
    seq_lens: torch.Tensor
    density: float


PUBLISHED_DISTRIBUTION_SUITE_NAMES = (
    "quick",
    "full",
    "broad",
)
PUBLIC_SUITE_ROOT_SEED_ENV = "ATTENTION_KERNEL_CHALLENGE_PUBLIC_SUITE_ROOT_SEED"


def ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def build_suite(name: str) -> List[CaseSpec]:
    if name == "smoke":
        suite = _smoke_suite()
    elif name == "local-dev":
        suite = _local_dev_suite()
    elif name == "quick":
        suite = _sample_public_suite(name, _quick_suite())
    elif name == "full":
        suite = _sample_public_suite(name, _full_suite())
    elif name == "broad":
        suite = _sample_public_suite(name, _broad_suite())
    else:
        raise ValueError(
            f"Unknown suite {name!r}. Expected one of {BUILTIN_SUITE_NAMES}."
        )

    for case in suite:
        validate_case_spec(case)
    return suite


def build_suite_from_manifest_path(path: str | Path) -> List[CaseSpec]:
    return build_suite_from_manifest_json(Path(path).read_text())


def build_suite_from_manifest_json(payload_json: str) -> List[CaseSpec]:
    payload = json.loads(payload_json)
    if not isinstance(payload, list):
        raise ValueError("Suite manifest JSON must be a list of case specs.")

    suite = []
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError("Each suite manifest item must be an object.")
        case = CaseSpec(**item)
        validate_case_spec(case)
        suite.append(case)
    return suite


def validate_case_spec(case: CaseSpec) -> None:
    if case.family not in {"sliding_window", "sliding_window_global", "sliding_window_retrieval"}:
        raise ValueError(f"Unsupported family {case.family!r} in case {case.case_id!r}.")
    if case.batch_size < 1 or case.num_heads < 1:
        raise ValueError(f"Case {case.case_id!r} must have positive batch size and num_heads.")
    if case.t_max < BLOCK_SIZE or case.t_max % BLOCK_SIZE != 0:
        raise ValueError(f"Case {case.case_id!r} must use t_max as a positive multiple of {BLOCK_SIZE}.")
    if case.window_blocks < 1:
        raise ValueError(f"Case {case.case_id!r} must have window_blocks >= 1.")
    if case.global_blocks < 0 or case.retrieval_blocks < 0:
        raise ValueError(f"Case {case.case_id!r} cannot use negative sparse metadata counts.")
    if not 0.0 <= case.retrieval_local_bias <= 1.0:
        raise ValueError(f"Case {case.case_id!r} must have retrieval_local_bias in [0, 1].")
    if not 0.0 < case.seq_len_min_ratio <= 1.0:
        raise ValueError(f"Case {case.case_id!r} must have seq_len_min_ratio in (0, 1].")


def suite_to_manifest_json(suite: Sequence[CaseSpec]) -> str:
    return json.dumps([case.__dict__ for case in suite], indent=2)


def is_public_distribution_suite(name: str) -> bool:
    return name in PUBLISHED_DISTRIBUTION_SUITE_NAMES


def build_public_suite_metadata(name: str) -> List[dict[str, object]]:
    if name == "quick":
        suite = _quick_suite()
    elif name == "full":
        suite = _full_suite()
    elif name == "broad":
        suite = _broad_suite()
    else:
        raise ValueError(
            f"Suite {name!r} does not publish distribution-only metadata."
        )

    payload: List[dict[str, object]] = []
    for case in suite:
        item = dict(case.__dict__)
        item.pop("seed", None)
        item["seed_policy"] = "hidden-site-sampled"
        payload.append(item)
    return payload


def resolve_suite(name: str | None = None, manifest_json: str | None = None) -> List[CaseSpec]:
    if manifest_json is not None:
        return build_suite_from_manifest_json(manifest_json)
    if name is None:
        raise ValueError("Either a built-in suite name or suite manifest JSON must be provided.")
    return build_suite(name)


def materialize_case(spec: CaseSpec, device: str = "cpu") -> MaterializedCase:
    validate_case_spec(spec)
    seq_lens = _sample_seq_lens(spec)
    row_ptr_np, col_idx_np, density = build_csr_metadata(spec, seq_lens.tolist())

    generator = torch.Generator(device="cpu")
    generator.manual_seed(spec.seed)

    shape = (spec.batch_size, spec.num_heads, spec.t_max, HEAD_DIM)
    q = torch.randn(shape, generator=generator, dtype=torch.float32).to(dtype=torch.bfloat16)
    k = torch.randn(shape, generator=generator, dtype=torch.float32).to(dtype=torch.bfloat16)
    v = torch.randn(shape, generator=generator, dtype=torch.float32).to(dtype=torch.bfloat16)

    materialized = MaterializedCase(
        spec=spec,
        q=q.to(device),
        k=k.to(device),
        v=v.to(device),
        row_ptr=torch.from_numpy(row_ptr_np).to(device),
        col_idx=torch.from_numpy(col_idx_np).to(device),
        seq_lens=torch.from_numpy(seq_lens).to(device),
        density=density,
    )
    return materialized


def build_csr_metadata(spec: CaseSpec, seq_lens: Sequence[int]) -> Tuple[np.ndarray, np.ndarray, float]:
    q_blocks = ceil_div(spec.t_max, BLOCK_SIZE)
    row_ptr = np.zeros((spec.batch_size, spec.num_heads, q_blocks + 1), dtype=np.int32)
    per_head_cols: Dict[Tuple[int, int], np.ndarray] = {}
    max_nnz = 0
    density_terms: List[float] = []

    root_rng = np.random.default_rng(spec.seed)

    for batch_index in range(spec.batch_size):
        valid_blocks = ceil_div(int(seq_lens[batch_index]), BLOCK_SIZE)
        for head_index in range(spec.num_heads):
            head_seed = int(root_rng.integers(0, 2**31 - 1))
            head_rng = np.random.default_rng(head_seed)
            cols: List[int] = []
            offsets = [0]

            for q_block in range(q_blocks):
                if q_block >= valid_blocks:
                    offsets.append(offsets[-1])
                    continue
                allowed = _allowed_blocks_for_row(spec, q_block, valid_blocks, head_rng)
                cols.extend(allowed)
                offsets.append(len(cols))

            row_ptr[batch_index, head_index] = np.asarray(offsets, dtype=np.int32)
            col_array = np.asarray(cols, dtype=np.int32) if cols else np.zeros((0,), dtype=np.int32)
            per_head_cols[(batch_index, head_index)] = col_array
            max_nnz = max(max_nnz, int(col_array.size))

            denominator = max(valid_blocks * valid_blocks, 1)
            density_terms.append(float(col_array.size) / float(denominator))

    if max_nnz == 0:
        max_nnz = 1

    col_idx = np.zeros((spec.batch_size, spec.num_heads, max_nnz), dtype=np.int32)
    for (batch_index, head_index), col_array in per_head_cols.items():
        if col_array.size:
            col_idx[batch_index, head_index, : col_array.size] = col_array

    density = float(sum(density_terms) / len(density_terms)) if density_terms else 0.0
    return row_ptr, col_idx, density


def _allowed_blocks_for_row(
    spec: CaseSpec,
    q_block: int,
    valid_blocks: int,
    rng: np.random.Generator,
) -> List[int]:
    allowed = set(_window_blocks(q_block, spec.window_blocks))

    if spec.family == "sliding_window_global":
        for global_block in range(min(spec.global_blocks, valid_blocks)):
            allowed.add(global_block)
    elif spec.family == "sliding_window_retrieval":
        for block in _retrieval_blocks(q_block, valid_blocks, spec.retrieval_blocks, spec.window_blocks, spec.retrieval_local_bias, rng):
            allowed.add(block)

    return sorted(block for block in allowed if 0 <= block <= q_block and block < valid_blocks)


def _window_blocks(q_block: int, window_blocks: int) -> Iterable[int]:
    start = max(0, q_block - window_blocks + 1)
    return range(start, q_block + 1)


def _retrieval_blocks(
    q_block: int,
    valid_blocks: int,
    retrieval_blocks: int,
    window_blocks: int,
    local_bias: float,
    rng: np.random.Generator,
) -> List[int]:
    if retrieval_blocks <= 0 or q_block <= 0:
        return []

    base_window = set(_window_blocks(q_block, window_blocks))
    local_start = max(0, q_block - (4 * window_blocks))

    local_candidates = [
        block
        for block in range(local_start, q_block)
        if block not in base_window
    ]
    far_candidates = [
        block
        for block in range(0, local_start)
        if block not in base_window
    ]

    selected = set()
    all_candidates = local_candidates + far_candidates
    if not all_candidates:
        return []

    while len(selected) < retrieval_blocks and len(selected) < len(all_candidates):
        use_local = bool(local_candidates) and (not far_candidates or rng.random() < local_bias)
        candidate_pool = local_candidates if use_local else far_candidates
        candidate = int(rng.choice(candidate_pool))
        selected.add(candidate)
        if candidate in local_candidates:
            local_candidates.remove(candidate)
        if candidate in far_candidates:
            far_candidates.remove(candidate)

    return sorted(selected)


def _sample_seq_lens(spec: CaseSpec) -> np.ndarray:
    min_len = max(BLOCK_SIZE, int(spec.t_max * spec.seq_len_min_ratio))
    rng = np.random.default_rng(spec.seed ^ 0xABCDEF)
    if min_len >= spec.t_max:
        values = np.full((spec.batch_size,), spec.t_max, dtype=np.int32)
    else:
        values = rng.integers(min_len, spec.t_max + 1, size=spec.batch_size, dtype=np.int32)
    return values


def _smoke_suite() -> List[CaseSpec]:
    return [
        CaseSpec(
            case_id="smoke-window-1",
            family="sliding_window",
            batch_size=1,
            num_heads=2,
            t_max=256,
            window_blocks=2,
            seed=11,
            profile="smoke",
        ),
        CaseSpec(
            case_id="smoke-global-1",
            family="sliding_window_global",
            batch_size=1,
            num_heads=2,
            t_max=384,
            window_blocks=2,
            global_blocks=1,
            seed=22,
            profile="smoke",
        ),
        CaseSpec(
            case_id="smoke-retrieval-1",
            family="sliding_window_retrieval",
            batch_size=1,
            num_heads=2,
            t_max=512,
            window_blocks=2,
            retrieval_blocks=2,
            seed=33,
            profile="smoke",
        ),
    ]


def _sample_public_suite(name: str, template_suite: Sequence[CaseSpec]) -> List[CaseSpec]:
    root_seed = _resolve_public_suite_root_seed(name)
    return [
        replace(
            case,
            seed=_derive_public_case_seed(root_seed, name, case.case_id),
        )
        for case in template_suite
    ]


def _resolve_public_suite_root_seed(name: str) -> int:
    seed_payload = os.environ.get(PUBLIC_SUITE_ROOT_SEED_ENV)
    if seed_payload is None:
        seed_payload = f"local-public-sample:{name}"
    return _seed_payload_to_int(seed_payload)


def _seed_payload_to_int(payload: str) -> int:
    try:
        return int(payload)
    except ValueError:
        digest = hashlib.sha256(payload.encode()).digest()
        return int.from_bytes(digest[:8], "big") % (2**31 - 1)


def _derive_public_case_seed(root_seed: int, suite_name: str, case_id: str) -> int:
    digest = hashlib.sha256(f"{root_seed}:{suite_name}:{case_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**31 - 1)


def _local_dev_suite() -> List[CaseSpec]:
    return [
        CaseSpec(
            case_id="local-window-1",
            family="sliding_window",
            batch_size=1,
            num_heads=4,
            t_max=512,
            window_blocks=4,
            seed=101,
            profile="local-dev",
        ),
        CaseSpec(
            case_id="local-window-2",
            family="sliding_window",
            batch_size=2,
            num_heads=4,
            t_max=768,
            window_blocks=3,
            seed=102,
            profile="local-dev",
        ),
        CaseSpec(
            case_id="local-global-1",
            family="sliding_window_global",
            batch_size=1,
            num_heads=4,
            t_max=640,
            window_blocks=4,
            global_blocks=2,
            seed=201,
            profile="local-dev",
        ),
        CaseSpec(
            case_id="local-global-2",
            family="sliding_window_global",
            batch_size=2,
            num_heads=4,
            t_max=896,
            window_blocks=3,
            global_blocks=1,
            seed=202,
            profile="local-dev",
        ),
        CaseSpec(
            case_id="local-retrieval-1",
            family="sliding_window_retrieval",
            batch_size=1,
            num_heads=4,
            t_max=768,
            window_blocks=3,
            retrieval_blocks=2,
            seed=301,
            profile="local-dev",
        ),
        CaseSpec(
            case_id="local-retrieval-2",
            family="sliding_window_retrieval",
            batch_size=2,
            num_heads=4,
            t_max=1024,
            window_blocks=4,
            retrieval_blocks=3,
            seed=302,
            profile="local-dev",
        ),
    ]


def _broad_suite() -> List[CaseSpec]:
    return [
        CaseSpec("broad-window-1", "sliding_window", 1, 16, 4096, window_blocks=4, seed=0, profile="broad"),
        CaseSpec("broad-window-2", "sliding_window", 2, 16, 8192, window_blocks=6, seed=0, profile="broad"),
        CaseSpec("broad-window-3", "sliding_window", 4, 32, 16384, window_blocks=8, seed=0, profile="broad"),
        CaseSpec("broad-global-1", "sliding_window_global", 1, 16, 4096, window_blocks=4, global_blocks=1, seed=0, profile="broad"),
        CaseSpec("broad-global-2", "sliding_window_global", 2, 32, 8192, window_blocks=5, global_blocks=2, seed=0, profile="broad"),
        CaseSpec("broad-global-3", "sliding_window_global", 4, 16, 16384, window_blocks=7, global_blocks=2, seed=0, profile="broad"),
        CaseSpec("broad-retrieval-1", "sliding_window_retrieval", 1, 16, 4096, window_blocks=4, retrieval_blocks=2, retrieval_local_bias=0.75, seed=0, profile="broad"),
        CaseSpec("broad-retrieval-2", "sliding_window_retrieval", 2, 16, 8192, window_blocks=5, retrieval_blocks=3, retrieval_local_bias=0.65, seed=0, profile="broad"),
        CaseSpec("broad-retrieval-3", "sliding_window_retrieval", 2, 32, 16384, window_blocks=7, retrieval_blocks=4, retrieval_local_bias=0.55, seed=0, profile="broad"),
    ]


def _quick_suite() -> List[CaseSpec]:
    return [
        CaseSpec("quick-window-1", "sliding_window", 1, 16, 4096, window_blocks=4, seed=0, profile="quick"),
        CaseSpec("quick-global-1", "sliding_window_global", 1, 16, 4096, window_blocks=4, global_blocks=1, seed=0, profile="quick"),
        CaseSpec("quick-retrieval-1", "sliding_window_retrieval", 1, 16, 4096, window_blocks=4, retrieval_blocks=2, retrieval_local_bias=0.75, seed=0, profile="quick"),
    ]


def _full_suite() -> List[CaseSpec]:
    return [
        CaseSpec(
            "full-window-1",
            "sliding_window",
            2,
            16,
            8192,
            window_blocks=4,
            seq_len_min_ratio=0.625,
            seed=0,
            profile="full",
        ),
        CaseSpec(
            "full-window-2",
            "sliding_window",
            2,
            16,
            8192,
            window_blocks=6,
            seq_len_min_ratio=0.75,
            seed=0,
            profile="full",
        ),
        CaseSpec(
            "full-window-3",
            "sliding_window",
            2,
            16,
            8192,
            window_blocks=8,
            seq_len_min_ratio=1.0,
            seed=0,
            profile="full",
        ),
        CaseSpec(
            "full-global-1",
            "sliding_window_global",
            2,
            16,
            8192,
            window_blocks=4,
            global_blocks=1,
            seq_len_min_ratio=0.625,
            seed=0,
            profile="full",
        ),
        CaseSpec(
            "full-global-2",
            "sliding_window_global",
            2,
            16,
            8192,
            window_blocks=5,
            global_blocks=2,
            seq_len_min_ratio=0.75,
            seed=0,
            profile="full",
        ),
        CaseSpec(
            "full-global-3",
            "sliding_window_global",
            2,
            16,
            8192,
            window_blocks=7,
            global_blocks=4,
            seq_len_min_ratio=1.0,
            seed=0,
            profile="full",
        ),
        CaseSpec(
            "full-retrieval-1",
            "sliding_window_retrieval",
            2,
            16,
            8192,
            window_blocks=4,
            retrieval_blocks=2,
            retrieval_local_bias=0.8,
            seq_len_min_ratio=0.625,
            seed=0,
            profile="full",
        ),
        CaseSpec(
            "full-retrieval-2",
            "sliding_window_retrieval",
            2,
            16,
            8192,
            window_blocks=5,
            retrieval_blocks=3,
            retrieval_local_bias=0.5,
            seq_len_min_ratio=0.75,
            seed=0,
            profile="full",
        ),
        CaseSpec(
            "full-retrieval-3",
            "sliding_window_retrieval",
            2,
            16,
            8192,
            window_blocks=7,
            retrieval_blocks=4,
            retrieval_local_bias=0.2,
            seq_len_min_ratio=1.0,
            seed=0,
            profile="full",
        ),
    ]
