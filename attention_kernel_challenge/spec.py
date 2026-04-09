from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

BLOCK_SIZE = 128
HEAD_DIM = 128
SCORE_SCALE = 1.0 / math.sqrt(HEAD_DIM)
MAX_VARIANT_COUNT = 16
BUILTIN_SUITE_NAMES = (
    "smoke",
    "local-dev",
    "quick",
    "full",
    "broad",
)

FamilyName = Literal[
    "sliding_window",
    "sliding_window_global",
    "sliding_window_retrieval",
]


@dataclass(frozen=True)
class CaseSpec:
    case_id: str
    family: FamilyName
    batch_size: int
    num_heads: int
    t_max: int
    window_blocks: int
    global_blocks: int = 0
    retrieval_blocks: int = 0
    retrieval_local_bias: float = 0.7
    seq_len_min_ratio: float = 0.625
    seed: int = 0
    profile: str = "smoke"


@dataclass(frozen=True)
class VariantSpec:
    name: str
    families: tuple[FamilyName, ...] = ()
    min_t_max: int = 0
    max_t_max: int = 1 << 30
    min_batch_heads: int = 0
    max_batch_heads: int = 1 << 30
    min_batch_size: int = 0
    max_batch_size: int = 1 << 30
    min_num_heads: int = 0
    max_num_heads: int = 1 << 30
    min_window_blocks: int = 0
    max_window_blocks: int = 1 << 30
    min_global_blocks: int = 0
    max_global_blocks: int = 1 << 30
    min_retrieval_blocks: int = 0
    max_retrieval_blocks: int = 1 << 30
    min_retrieval_local_bias: float = 0.0
    max_retrieval_local_bias: float = 1.0

    def matches(self, case: CaseSpec) -> bool:
        if self.families and case.family not in self.families:
            return False

        batch_heads = case.batch_size * case.num_heads
        return (
            self.min_t_max <= case.t_max <= self.max_t_max
            and self.min_batch_heads <= batch_heads <= self.max_batch_heads
            and self.min_batch_size <= case.batch_size <= self.max_batch_size
            and self.min_num_heads <= case.num_heads <= self.max_num_heads
            and self.min_window_blocks <= case.window_blocks <= self.max_window_blocks
            and self.min_global_blocks <= case.global_blocks <= self.max_global_blocks
            and self.min_retrieval_blocks <= case.retrieval_blocks <= self.max_retrieval_blocks
            and self.min_retrieval_local_bias
            <= case.retrieval_local_bias
            <= self.max_retrieval_local_bias
        )


@dataclass(frozen=True)
class Tolerances:
    output_atol: float = 1e-3
    output_rtol: float = 1e-2
    lse_atol: float = 1e-5
    lse_rtol: float = 1e-5


@dataclass(frozen=True)
class EvaluationConfig:
    device: str = "cpu"
    setup_device: str | None = None
    warmup_iters: int = 1
    measure_iters: int = 3
    check_correctness: bool = False
    correctness_only: bool = False
    setup_timeout_s: float = 30.0
    setup_warmup_iters: int = 1
    enforce_post_setup_cache_stability: bool = True
    isolate_submission_process: bool = False
    tolerances: Tolerances = Tolerances()
    suite_manifest_json: str | None = None
