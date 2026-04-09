import unittest

from attention_kernel_challenge.cases import build_suite, materialize_case
from attention_kernel_challenge.reference import (
    dense_reference_block_sparse_attn_fwd,
    reference_block_sparse_attn_fwd,
)
from attention_kernel_challenge.spec import Tolerances
from attention_kernel_challenge.validation import validate_outputs


class ReferenceTests(unittest.TestCase):
    def test_streaming_reference_matches_dense_reference(self) -> None:
        for case_spec in build_suite("smoke"):
            case = materialize_case(case_spec, device="cpu")
            reference_out, reference_lse = reference_block_sparse_attn_fwd(
                case.q, case.k, case.v, case.row_ptr, case.col_idx, case.seq_lens
            )
            dense_out, dense_lse = dense_reference_block_sparse_attn_fwd(
                case.q, case.k, case.v, case.row_ptr, case.col_idx, case.seq_lens
            )

            validation = validate_outputs(
                reference_out,
                reference_lse,
                dense_out,
                dense_lse,
                Tolerances(),
            )
            self.assertTrue(validation.passed, f"{case_spec.case_id}: {validation.message}")


if __name__ == "__main__":
    unittest.main()
