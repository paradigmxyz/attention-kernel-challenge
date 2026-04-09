from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from attention_kernel_challenge.cases import (
    PUBLIC_SUITE_ROOT_SEED_ENV,
    build_public_suite_metadata,
    build_csr_metadata,
    build_suite,
    build_suite_from_manifest_json,
    materialize_case,
    suite_to_manifest_json,
)
from attention_kernel_challenge.evaluator import (
    _build_public_warmup_descriptors,
    _synchronize as evaluator_synchronize,
    evaluate_callable,
    evaluate_reference_suite,
)
from attention_kernel_challenge.reference import reference_block_sparse_attn_fwd
from attention_kernel_challenge.isolated_submission import _synchronize as isolated_synchronize
from attention_kernel_challenge.spec import CaseSpec, EvaluationConfig


class HarnessTests(unittest.TestCase):
    def test_suite_is_deterministic(self) -> None:
        first = build_suite("local-dev")
        second = build_suite("local-dev")
        self.assertEqual(first, second)

    def test_generated_metadata_is_sorted_and_causal(self) -> None:
        case = build_suite("smoke")[2]
        materialized = materialize_case(case)
        row_ptr = materialized.row_ptr.cpu().numpy()
        col_idx = materialized.col_idx.cpu().numpy()

        for batch_index in range(case.batch_size):
            for head_index in range(case.num_heads):
                for q_block in range(row_ptr.shape[-1] - 1):
                    start = int(row_ptr[batch_index, head_index, q_block])
                    end = int(row_ptr[batch_index, head_index, q_block + 1])
                    values = list(col_idx[batch_index, head_index, start:end])
                    self.assertEqual(values, sorted(set(values)))
                    self.assertTrue(all(value <= q_block for value in values))

    def test_reference_evaluator_smoke_suite_passes(self) -> None:
        summary = evaluate_reference_suite("smoke", EvaluationConfig(device="cpu", warmup_iters=0, measure_iters=1))
        self.assertTrue(summary.overall_valid)
        self.assertEqual(len(summary.case_results), len(build_suite("smoke")))

    def test_public_suite_manifest_roundtrip(self) -> None:
        suite = build_suite("broad")
        manifest_json = suite_to_manifest_json(suite)
        self.assertEqual(suite, build_suite_from_manifest_json(manifest_json))

    def test_full_stays_in_single_bucket_and_varies_descriptors(self) -> None:
        suite = build_suite("full")
        self.assertEqual(len(suite), 9)
        self.assertEqual({case.batch_size for case in suite}, {2})
        self.assertEqual({case.num_heads for case in suite}, {16})
        self.assertEqual({case.t_max for case in suite}, {8192})
        self.assertEqual(len({case.window_blocks for case in suite}), 5)
        self.assertEqual({case.seq_len_min_ratio for case in suite}, {0.625, 0.75, 1.0})
        self.assertEqual(
            {case.global_blocks for case in suite if case.family == "sliding_window_global"},
            {1, 2, 4},
        )
        self.assertEqual(
            {case.retrieval_blocks for case in suite if case.family == "sliding_window_retrieval"},
            {2, 3, 4},
        )
        self.assertEqual(
            {case.retrieval_local_bias for case in suite if case.family == "sliding_window_retrieval"},
            {0.8, 0.5, 0.2},
        )

    def test_public_suite_metadata_hides_exact_seeds(self) -> None:
        metadata = build_public_suite_metadata("full")
        self.assertEqual(len(metadata), 9)
        self.assertTrue(all("seed" not in item for item in metadata))
        self.assertEqual({item["seed_policy"] for item in metadata}, {"hidden-site-sampled"})

    def test_public_suite_sampling_uses_hidden_root_seed(self) -> None:
        with patch.dict("os.environ", {PUBLIC_SUITE_ROOT_SEED_ENV: "111"}):
            first = build_suite("full")
            second = build_suite("full")
        with patch.dict("os.environ", {PUBLIC_SUITE_ROOT_SEED_ENV: "222"}):
            third = build_suite("full")

        self.assertEqual(first, second)
        self.assertNotEqual([case.seed for case in first], [case.seed for case in third])
        self.assertEqual(
            [{**case.__dict__, "seed": 0} for case in first],
            [{**case.__dict__, "seed": 0} for case in third],
        )

    def test_stale_public_suite_aliases_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            build_suite("public-quick")
        with self.assertRaises(ValueError):
            build_suite("public-queue")
        with self.assertRaises(ValueError):
            build_suite("public-dev")

    def test_hidden_scoring_suite_is_not_public_builtin(self) -> None:
        with self.assertRaises(ValueError):
            build_suite("public-score")

    def test_setup_suite_redaction_removes_hidden_details_and_duplicates(self) -> None:
        suite = [
            CaseSpec(
                case_id="hidden-a",
                family="sliding_window_retrieval",
                batch_size=2,
                num_heads=4,
                t_max=512,
                window_blocks=2,
                retrieval_blocks=3,
                retrieval_local_bias=0.2,
                seq_len_min_ratio=0.5,
                seed=11,
                profile="hidden",
            ),
            CaseSpec(
                case_id="hidden-b",
                family="sliding_window_retrieval",
                batch_size=2,
                num_heads=4,
                t_max=512,
                window_blocks=2,
                retrieval_blocks=3,
                retrieval_local_bias=0.9,
                seq_len_min_ratio=0.75,
                seed=22,
                profile="hidden",
            ),
        ]
        redacted = _build_public_warmup_descriptors(suite)
        self.assertEqual(len(redacted), 1)
        case = redacted[0]
        self.assertEqual(case.case_id, "warmup-key-1")
        self.assertEqual(case.profile, "setup")
        self.assertEqual(case.retrieval_local_bias, 0.5)
        self.assertEqual(case.seq_len_min_ratio, 1.0)
        self.assertEqual(case.seed, 0)
        self.assertEqual(case.retrieval_blocks, 3)
        self.assertEqual(case.t_max, 512)

    def test_redacted_summary_hides_failure_details(self) -> None:
        summary = evaluate_reference_suite(
            "smoke",
            EvaluationConfig(device="cpu", warmup_iters=0, measure_iters=1),
        )
        summary.failure_reason = "Case 'smoke-window-1' failed: seed=11 boom"
        redacted = summary.redacted()
        self.assertEqual(redacted.case_results, [])
        self.assertEqual(redacted.scored_case_count, len(summary.case_results))
        self.assertEqual(redacted.failure_reason, "Evaluation failed on a hidden case.")

    def test_reference_evaluator_core_api_skips_correctness_by_default(self) -> None:
        summary = evaluate_reference_suite(
            "smoke",
            EvaluationConfig(device="cpu", warmup_iters=0, measure_iters=1),
        )
        self.assertTrue(summary.overall_valid)
        self.assertTrue(all(case.validation.message.startswith("skipped") for case in summary.case_results))

    def test_synchronize_supports_mps(self) -> None:
        synchronize_mock = Mock()
        with patch.object(
            torch,
            "mps",
            SimpleNamespace(synchronize=synchronize_mock),
            create=True,
        ):
            evaluator_synchronize(torch.device("mps"))
            isolated_synchronize(torch.device("mps"))

        self.assertEqual(synchronize_mock.call_count, 2)

    def test_setup_can_use_separate_setup_device(self) -> None:
        captured = []

        def setup(suite_specs, device):
            captured.append(device)

        summary = evaluate_callable(
            reference_block_sparse_attn_fwd,
            "smoke",
            EvaluationConfig(
                device="cpu",
                setup_device="cuda",
                warmup_iters=0,
                measure_iters=1,
            ),
            setup=setup,
        )

        self.assertTrue(summary.overall_valid)
        self.assertEqual(captured, ["cuda"])


if __name__ == "__main__":
    unittest.main()
