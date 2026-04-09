import sys
import tempfile
import unittest
from pathlib import Path

import torch._inductor

from attention_kernel_challenge.cases import suite_to_manifest_json
from attention_kernel_challenge.evaluator import evaluate_submission_dir
from attention_kernel_challenge.spec import CaseSpec, EvaluationConfig
from attention_kernel_challenge.submission_loader import (
    _submission_import_guard,
    find_matching_variant,
    load_submission,
    pack_submission_dir,
    unpack_submission_archive,
)


class SubmissionLoaderTests(unittest.TestCase):
    def test_example_submission_loads_and_runs(self) -> None:
        loaded = load_submission("example_submission")
        summary = evaluate_submission_dir(
            "example_submission",
            "smoke",
            EvaluationConfig(device="cpu", warmup_iters=0, measure_iters=1),
        )
        self.assertTrue(summary.overall_valid)
        self.assertEqual(loaded.variants[0].name, "default")

    def test_example_submission_loads_and_runs_isolated(self) -> None:
        summary = evaluate_submission_dir(
            "example_submission",
            "smoke",
            EvaluationConfig(
                device="cpu",
                warmup_iters=0,
                measure_iters=1,
                isolate_submission_process=True,
            ),
        )
        self.assertTrue(summary.overall_valid, summary.failure_reason)

    def test_pack_and_unpack_submission_archive(self) -> None:
        archive = pack_submission_dir("example_submission")
        unpacked = unpack_submission_archive(archive)
        loaded = load_submission(unpacked)
        self.assertTrue(callable(loaded.entrypoint))

    def test_disallowed_imports_fail(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "submission.py").write_text(
                "VARIANT_MANIFEST = ['default']\n"
                "import pandas\n\n"
                "def block_sparse_attn_fwd(*args):\n"
                "    return args[0], args[0]\n"
            )
            with self.assertRaises(ImportError):
                load_submission(root)

    def test_runtime_guard_allows_torch_transitive_imports_but_not_direct_ones(self) -> None:
        with _submission_import_guard(set()):
            exec(
                compile("import typing_extensions", torch._inductor.__file__, "exec"),
                torch._inductor.__dict__,
            )

        with _submission_import_guard(set()):
            with self.assertRaises(ImportError):
                exec("import typing_extensions", {"__name__": "submission"})

    def test_runtime_guard_allows_compiler_owned_subimports(self) -> None:
        with _submission_import_guard(set()):
            exec(
                compile("import sympy.core.relational", torch._inductor.__file__, "exec"),
                torch._inductor.__dict__,
            )
            sympy_relational = sys.modules["sympy.core.relational"]
            exec(
                compile(
                    "import sympy.assumptions.wrapper",
                    sympy_relational.__file__,
                    "exec",
                ),
                sympy_relational.__dict__,
            )

    def test_runtime_guard_rejects_spoofed_compiler_importer_name(self) -> None:
        with _submission_import_guard(set()):
            with self.assertRaises(ImportError):
                exec("import typing_extensions", {"__name__": "torch._inductor.fake"})

    def test_runtime_guard_does_not_expand_to_arbitrary_third_party_imports(self) -> None:
        with _submission_import_guard(set()):
            with self.assertRaises(ImportError):
                exec(
                    compile("import pandas", torch._inductor.__file__, "exec"),
                    torch._inductor.__dict__,
                )

    def test_variant_manifest_is_required(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "submission.py").write_text(
                "def block_sparse_attn_fwd(*args):\n"
                "    return args[0], args[0]\n"
            )
            with self.assertRaises(AttributeError):
                load_submission(root)

    def test_variant_manifest_supports_structural_dispatch_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "submission.py").write_text(
                "VARIANT_MANIFEST = [\n"
                "    {\n"
                "        'name': 'window_exact',\n"
                "        'families': ['sliding_window'],\n"
                "        'window_blocks': 4,\n"
                "        'batch_size': 1,\n"
                "        'num_heads': 16,\n"
                "    },\n"
                "    {\n"
                "        'name': 'global_range',\n"
                "        'families': ['sliding_window_global'],\n"
                "        'min_window_blocks': 5,\n"
                "        'max_window_blocks': 7,\n"
                "        'min_global_blocks': 2,\n"
                "        'max_global_blocks': 4,\n"
                "    },\n"
                "    {\n"
                "        'name': 'retrieval_exact',\n"
                "        'families': ['sliding_window_retrieval'],\n"
                "        'window_blocks': 5,\n"
                "        'retrieval_blocks': 3,\n"
                "        'retrieval_local_bias': 0.5,\n"
                "    },\n"
                "    {\n"
                "        'name': 'retrieval_fallback',\n"
                "        'families': ['sliding_window_retrieval'],\n"
                "        'min_window_blocks': 6,\n"
                "    },\n"
                "]\n"
                "def block_sparse_attn_fwd(*args):\n"
                "    return args[0], args[0]\n"
            )
            loaded = load_submission(root)
            self.assertEqual(
                find_matching_variant(
                    CaseSpec(
                        case_id="window-case",
                        family="sliding_window",
                        batch_size=1,
                        num_heads=16,
                        t_max=4096,
                        window_blocks=4,
                    ),
                    loaded.variants,
                ).name,
                "window_exact",
            )
            self.assertEqual(
                find_matching_variant(
                    CaseSpec(
                        case_id="global-case",
                        family="sliding_window_global",
                        batch_size=2,
                        num_heads=16,
                        t_max=8192,
                        window_blocks=5,
                        global_blocks=2,
                    ),
                    loaded.variants,
                ).name,
                "global_range",
            )
            self.assertEqual(
                find_matching_variant(
                    CaseSpec(
                        case_id="retrieval-case",
                        family="sliding_window_retrieval",
                        batch_size=2,
                        num_heads=16,
                        t_max=8192,
                        window_blocks=5,
                        retrieval_blocks=3,
                        retrieval_local_bias=0.5,
                    ),
                    loaded.variants,
                ).name,
                "retrieval_exact",
            )
            self.assertEqual(
                find_matching_variant(
                    CaseSpec(
                        case_id="fallback-case",
                        family="sliding_window_retrieval",
                        batch_size=2,
                        num_heads=16,
                        t_max=8192,
                        window_blocks=7,
                        retrieval_blocks=4,
                        retrieval_local_bias=0.2,
                    ),
                    loaded.variants,
                ).name,
                "retrieval_fallback",
            )

    def test_variant_manifest_rejects_mixed_exact_and_range_for_same_field(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "submission.py").write_text(
                "VARIANT_MANIFEST = [\n"
                "    {'name': 'bad', 'window_blocks': 4, 'min_window_blocks': 4},\n"
                "]\n"
                "def block_sparse_attn_fwd(*args):\n"
                "    return args[0], args[0]\n"
            )
            with self.assertRaises(ValueError):
                load_submission(root)

    def test_variant_manifest_structural_rules_must_be_unambiguous(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "submission.py").write_text(
                "VARIANT_MANIFEST = [\n"
                "    {'name': 'narrow', 'families': ['sliding_window'], 'window_blocks': 4},\n"
                "    {'name': 'wide', 'families': ['sliding_window'], 'min_window_blocks': 3, 'max_window_blocks': 5},\n"
                "]\n"
                "def block_sparse_attn_fwd(*args):\n"
                "    return args[0], args[0]\n"
            )
            loaded = load_submission(root)
            with self.assertRaises(ValueError):
                find_matching_variant(
                    CaseSpec(
                        case_id="overlap-case",
                        family="sliding_window",
                        batch_size=1,
                        num_heads=16,
                        t_max=4096,
                        window_blocks=4,
                    ),
                    loaded.variants,
                )

    def test_runtime_policy_violation_fails_submission(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "submission.py").write_text(
                "VARIANT_MANIFEST = ['default']\n"
                "def block_sparse_attn_fwd(q, k, v, row_ptr, col_idx, seq_lens):\n"
                "    import subprocess\n"
                "    subprocess.run(['echo', 'hi'])\n"
                "    return q, q.sum(dim=-1, dtype=q.dtype)\n"
            )
            summary = evaluate_submission_dir(
                str(root),
                "smoke",
                EvaluationConfig(device="cpu", warmup_iters=0, measure_iters=1),
            )
            self.assertFalse(summary.overall_valid)
            self.assertIn("subprocess.run", summary.failure_reason)

    def test_isolated_runtime_policy_violation_fails_submission(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "submission.py").write_text(
                "VARIANT_MANIFEST = ['default']\n"
                "def block_sparse_attn_fwd(q, k, v, row_ptr, col_idx, seq_lens):\n"
                "    import subprocess\n"
                "    subprocess.run(['echo', 'hi'])\n"
                "    return q, q.sum(dim=-1, dtype=q.dtype)\n"
            )
            summary = evaluate_submission_dir(
                str(root),
                "smoke",
                EvaluationConfig(
                    device="cpu",
                    warmup_iters=0,
                    measure_iters=1,
                    isolate_submission_process=True,
                ),
            )
            self.assertFalse(summary.overall_valid)
            self.assertIn("subprocess.run", summary.failure_reason)

    def test_isolated_import_time_policy_violation_fails_submission(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "submission.py").write_text(
                "import subprocess\n"
                "subprocess.run(['echo', 'hi'])\n"
                "VARIANT_MANIFEST = ['default']\n"
                "def block_sparse_attn_fwd(q, k, v, row_ptr, col_idx, seq_lens):\n"
                "    return q, q.sum(dim=-1, dtype=q.dtype)\n"
            )
            summary = evaluate_submission_dir(
                str(root),
                "smoke",
                EvaluationConfig(
                    device="cpu",
                    warmup_iters=0,
                    measure_iters=1,
                    isolate_submission_process=True,
                ),
            )
            self.assertFalse(summary.overall_valid)
            self.assertIn("subprocess.run", summary.failure_reason)

    def test_setup_device_can_simulate_server_only_setup_branch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "submission.py").write_text(
                "VARIANT_MANIFEST = ['default']\n"
                "def setup(suite_specs, device, variants):\n"
                "    if device == 'cuda':\n"
                "        import typing_extensions\n"
                "def block_sparse_attn_fwd(q, k, v, row_ptr, col_idx, seq_lens):\n"
                "    return q, q.sum(dim=-1, dtype=q.dtype)\n"
            )
            summary = evaluate_submission_dir(
                str(root),
                "smoke",
                EvaluationConfig(
                    device="cpu",
                    setup_device="cuda",
                    warmup_iters=0,
                    measure_iters=1,
                    isolate_submission_process=True,
                ),
            )
            self.assertFalse(summary.overall_valid)
            self.assertIn("typing_extensions", summary.failure_reason)

    def test_posix_spawn_policy_violation_fails_submission(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "submission.py").write_text(
                "VARIANT_MANIFEST = ['default']\n"
                "def block_sparse_attn_fwd(q, k, v, row_ptr, col_idx, seq_lens):\n"
                "    import os\n"
                "    pid = os.posix_spawn('/usr/bin/true', ['/usr/bin/true'], {})\n"
                "    os.waitpid(pid, 0)\n"
                "    return q, q.sum(dim=-1, dtype=q.dtype)\n"
            )
            summary = evaluate_submission_dir(
                str(root),
                "smoke",
                EvaluationConfig(device="cpu", warmup_iters=0, measure_iters=1),
            )
            self.assertFalse(summary.overall_valid)
            self.assertIn("os.posix_spawn", summary.failure_reason)

    def test_dynamic_import_of_harness_module_fails_submission(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "submission.py").write_text(
                "VARIANT_MANIFEST = ['default']\n"
                "import importlib\n"
                "reference = importlib.import_module('attention_kernel_challenge.reference')\n"
                "def block_sparse_attn_fwd(*args):\n"
                "    return reference.reference_block_sparse_attn_fwd(*args)\n"
            )
            with self.assertRaises(ImportError):
                load_submission(root)

    def test_post_setup_cache_mutation_fails_submission(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            example_source = Path("example_submission/submission.py").read_text()
            marker = (
                "\n_write_once_counter = 0\n"
                "def _touch_cache():\n"
                "    import os\n"
                "    from pathlib import Path\n"
                "    global _write_once_counter\n"
                "    _write_once_counter += 1\n"
                "    if _write_once_counter >= 2:\n"
                "        Path(os.environ['TRITON_CACHE_DIR'], 'late_compile.txt').write_text(str(_write_once_counter))\n"
            )
            mutated = example_source.replace(
                "def block_sparse_attn_fwd(q, k, v, row_ptr, col_idx, seq_lens):\n",
                marker + "\n\ndef block_sparse_attn_fwd(q, k, v, row_ptr, col_idx, seq_lens):\n    _touch_cache()\n",
            )
            (root / "submission.py").write_text(mutated)
            summary = evaluate_submission_dir(
                str(root),
                "smoke",
                EvaluationConfig(device="cpu", warmup_iters=0, measure_iters=1),
            )
            self.assertFalse(summary.overall_valid)
            self.assertIn("cache mutation", summary.failure_reason.lower())

    def test_timed_outputs_are_validated(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            example_source = Path("example_submission/submission.py").read_text()
            marker = (
                "\n_call_count = 0\n"
                "def _return_wrong_output(q):\n"
                "    return q, q.sum(dim=-1, dtype=torch.float32)\n"
            )
            mutated = example_source.replace(
                "def block_sparse_attn_fwd(q, k, v, row_ptr, col_idx, seq_lens):\n",
                marker
                + "\n\ndef block_sparse_attn_fwd(q, k, v, row_ptr, col_idx, seq_lens):\n"
                + "    global _call_count\n"
                + "    _call_count += 1\n"
                + "    if _call_count > 1:\n"
                + "        return _return_wrong_output(q)\n",
            )
            (root / "submission.py").write_text(mutated)
            summary = evaluate_submission_dir(
                str(root),
                "smoke",
                EvaluationConfig(
                    device="cpu",
                    warmup_iters=1,
                    measure_iters=1,
                    check_correctness=True,
                    setup_warmup_iters=0,
                ),
            )
            self.assertFalse(summary.overall_valid)
            self.assertIn("benchmark validation failed", summary.failure_reason.lower())

    def test_submission_eval_skips_correctness_by_default(self) -> None:
        summary = evaluate_submission_dir(
            "example_submission",
            "smoke",
            EvaluationConfig(device="cpu", warmup_iters=0, measure_iters=1),
        )
        self.assertTrue(summary.overall_valid)
        self.assertTrue(all(case.validation.message.startswith("skipped") for case in summary.case_results))

    def test_submission_eval_correctness_only_omits_timings(self) -> None:
        summary = evaluate_submission_dir(
            "example_submission",
            "smoke",
            EvaluationConfig(
                device="cpu",
                warmup_iters=0,
                measure_iters=1,
                check_correctness=True,
                correctness_only=True,
            ),
        )
        self.assertTrue(summary.overall_valid)
        self.assertIsNone(summary.geometric_mean_family_latency_ms)
        self.assertIsNone(summary.worst_family_latency_ms)
        self.assertTrue(all(case.latency_ms is None for case in summary.case_results))
        self.assertTrue(all(case.validation.message == "ok (local correctness check)" for case in summary.case_results))

    def test_setup_receives_unique_public_warmup_descriptors(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            example_source = Path("example_submission/submission.py").read_text()
            source = example_source.replace(
                "def setup(suite_specs, device, variants):\n    return None\n",
                "def setup(suite_specs, device, variants):\n"
                "    if len(suite_specs) != 1:\n"
                "        raise RuntimeError(f'expected 1 public descriptor, got {len(suite_specs)}')\n"
                "    spec = suite_specs[0]\n"
                "    if spec.case_id != 'warmup-key-1':\n"
                "        raise RuntimeError(f'unexpected descriptor id: {spec.case_id}')\n"
                "    if spec.seed != 0 or spec.seq_len_min_ratio != 1.0:\n"
                "        raise RuntimeError('hidden details leaked into setup descriptor')\n"
                "    if spec.retrieval_local_bias != 0.5:\n"
                "        raise RuntimeError('hidden retrieval bias leaked into setup descriptor')\n",
            )
            (root / "submission.py").write_text(source)
            manifest_json = suite_to_manifest_json(
                [
                    CaseSpec(
                        case_id="hidden-case-1",
                        family="sliding_window_retrieval",
                        batch_size=8,
                        num_heads=1,
                        t_max=512,
                        window_blocks=2,
                        retrieval_blocks=2,
                        retrieval_local_bias=0.1,
                        seq_len_min_ratio=0.625,
                        seed=123,
                        profile="hidden",
                    ),
                    CaseSpec(
                        case_id="hidden-case-2",
                        family="sliding_window_retrieval",
                        batch_size=8,
                        num_heads=1,
                        t_max=512,
                        window_blocks=2,
                        retrieval_blocks=2,
                        retrieval_local_bias=0.9,
                        seq_len_min_ratio=0.75,
                        seed=456,
                        profile="hidden",
                    ),
                ]
            )
            summary = evaluate_submission_dir(
                str(root),
                "manifest",
                EvaluationConfig(
                    device="cpu",
                    warmup_iters=0,
                    measure_iters=1,
                    setup_warmup_iters=0,
                    suite_manifest_json=manifest_json,
                ),
            )
            self.assertTrue(summary.overall_valid, summary.failure_reason)

    def test_runtime_warmups_use_public_descriptor_once_per_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            example_source = Path("example_submission/submission.py").read_text()
            marker = (
                "\n_public_full_seq_calls = 0\n"
                "def _assert_public_runtime_warmup_budget(q, seq_lens):\n"
                "    global _public_full_seq_calls\n"
                "    if bool(torch.all(seq_lens == q.shape[2])):\n"
                "        _public_full_seq_calls += 1\n"
                "        if _public_full_seq_calls > 2:\n"
                "            raise RuntimeError('runtime warmup ran more than once for the same public descriptor')\n"
            )
            mutated = example_source.replace(
                "def block_sparse_attn_fwd(q, k, v, row_ptr, col_idx, seq_lens):\n",
                marker + "\n\ndef block_sparse_attn_fwd(q, k, v, row_ptr, col_idx, seq_lens):\n    _assert_public_runtime_warmup_budget(q, seq_lens)\n",
            )
            (root / "submission.py").write_text(mutated)
            manifest_json = suite_to_manifest_json(
                [
                    CaseSpec(
                        case_id="hidden-r1",
                        family="sliding_window_retrieval",
                        batch_size=4,
                        num_heads=2,
                        t_max=256,
                        window_blocks=2,
                        retrieval_blocks=2,
                        retrieval_local_bias=0.2,
                        seq_len_min_ratio=0.5,
                        seed=1,
                        profile="hidden",
                    ),
                    CaseSpec(
                        case_id="hidden-r2",
                        family="sliding_window_retrieval",
                        batch_size=4,
                        num_heads=2,
                        t_max=256,
                        window_blocks=2,
                        retrieval_blocks=2,
                        retrieval_local_bias=0.9,
                        seq_len_min_ratio=0.5,
                        seed=2,
                        profile="hidden",
                    ),
                ]
            )
            summary = evaluate_submission_dir(
                str(root),
                "manifest",
                EvaluationConfig(
                    device="cpu",
                    warmup_iters=1,
                    measure_iters=1,
                    setup_warmup_iters=1,
                    suite_manifest_json=manifest_json,
                ),
            )
            self.assertTrue(summary.overall_valid, summary.failure_reason)

    def test_isolated_submission_blocks_setup_frame_leak(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            example_source = Path("example_submission/submission.py").read_text()
            source = example_source.replace(
                "def setup(suite_specs, device, variants):\n    return None\n",
                "def setup(suite_specs, device, variants):\n"
                "    import sys\n"
                "    frame = sys._getframe()\n"
                "    while frame is not None:\n"
                "        suite = frame.f_locals.get('suite')\n"
                "        if isinstance(suite, list) and suite and hasattr(suite[0], 'seed'):\n"
                "            seeds = [item.seed for item in suite]\n"
                "            if any(seed != 0 for seed in seeds):\n"
                "                raise RuntimeError(f'leaked hidden suite seeds in setup: {seeds}')\n"
                "        frame = frame.f_back\n"
                "    return None\n",
            )
            (root / "submission.py").write_text(source)
            manifest_json = suite_to_manifest_json(
                [
                    CaseSpec(
                        case_id="hidden-setup-1",
                        family="sliding_window_retrieval",
                        batch_size=1,
                        num_heads=2,
                        t_max=256,
                        window_blocks=2,
                        retrieval_blocks=2,
                        seed=123,
                        profile="hidden",
                    )
                ]
            )
            summary = evaluate_submission_dir(
                str(root),
                "manifest",
                EvaluationConfig(
                    device="cpu",
                    warmup_iters=0,
                    measure_iters=1,
                    isolate_submission_process=True,
                    suite_manifest_json=manifest_json,
                ),
            )
            self.assertTrue(summary.overall_valid, summary.failure_reason)

    def test_isolated_submission_blocks_runtime_frame_leak(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            example_source = Path("example_submission/submission.py").read_text()
            mutated = example_source.replace(
                "def block_sparse_attn_fwd(q, k, v, row_ptr, col_idx, seq_lens):\n",
                "def block_sparse_attn_fwd(q, k, v, row_ptr, col_idx, seq_lens):\n"
                "    import sys\n"
                "    frame = sys._getframe()\n"
                "    while frame is not None:\n"
                "        case = frame.f_locals.get('case')\n"
                "        if case is not None and hasattr(case, 'spec'):\n"
                "            raise RuntimeError(f'leaked runtime seed: {case.spec.seed}')\n"
                "        suite = frame.f_locals.get('suite')\n"
                "        if isinstance(suite, list) and suite and hasattr(suite[0], 'seed'):\n"
                "            seeds = [item.seed for item in suite]\n"
                "            raise RuntimeError(f'leaked suite in runtime: {seeds}')\n"
                "        frame = frame.f_back\n",
            )
            (root / "submission.py").write_text(mutated)
            manifest_json = suite_to_manifest_json(
                [
                    CaseSpec(
                        case_id="hidden-runtime-1",
                        family="sliding_window_retrieval",
                        batch_size=1,
                        num_heads=2,
                        t_max=256,
                        window_blocks=2,
                        retrieval_blocks=2,
                        seed=123,
                        profile="hidden",
                    )
                ]
            )
            summary = evaluate_submission_dir(
                str(root),
                "manifest",
                EvaluationConfig(
                    device="cpu",
                    warmup_iters=0,
                    measure_iters=1,
                    setup_warmup_iters=0,
                    isolate_submission_process=True,
                    suite_manifest_json=manifest_json,
                ),
            )
            self.assertTrue(summary.overall_valid, summary.failure_reason)


if __name__ == "__main__":
    unittest.main()
