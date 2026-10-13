# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import importlib.metadata
import importlib.util
import sys
import types
import unittest
from unittest import mock

import torch

from faiss.contrib.e_means import Emeans


def _cute_sm90_ready():
    if not torch.cuda.is_available():
        return False
    if getattr(torch.version, "hip", None) is not None:
        return False
    if torch.cuda.get_device_capability() != (9, 0):
        return False
    try:
        return (
            importlib.metadata.version("quack-kernels") == "0.6.5"
            and importlib.metadata.version("nvidia-cutlass-dsl") == "4.7.1"
            and importlib.metadata.version("cuda-python").startswith("13.")
            and importlib.util.find_spec("cutlass") is not None
        )
    except importlib.metadata.PackageNotFoundError:
        return False


_CUTE_SM90 = _cute_sm90_ready()
_CUTE_REASON = "the pinned CuTe stack on SM90 is required"


def _two_sm90_ready():
    return (
        _CUTE_SM90
        and torch.cuda.device_count() >= 2
        and torch.cuda.get_device_capability(1) == (9, 0)
    )


_TWO_SM90 = _two_sm90_ready()


class TestEmeansCute(unittest.TestCase):
    def test_backend_is_lazy_and_narrow(self):
        import faiss.contrib as contrib

        name = "faiss.contrib.e_means_cute"
        saved_module = sys.modules.pop(name, None)
        saved_attribute = getattr(contrib, "e_means_cute", None)
        if hasattr(contrib, "e_means_cute"):
            delattr(contrib, "e_means_cute")
        try:
            with self.assertRaisesRegex(ValueError, "BF16 or E4M3"):
                Emeans(8, 4, assignment_backend="cute")
            model = Emeans(
                8,
                4,
                assignment_backend="cute",
                assignment_precision="bfloat16",
            )
            self.assertEqual(model.assignment_backend, "cute")
            fp8 = Emeans(
                8,
                4,
                assignment_backend="cute",
                assignment_precision="float8_e4m3fn",
            )
            self.assertEqual(fp8.assignment_precision, "float8_e4m3fn")
            self.assertNotIn(name, sys.modules)
        finally:
            if saved_module is not None:
                sys.modules[name] = saved_module
            if saved_attribute is not None:
                contrib.e_means_cute = saved_attribute

    def test_compatibility_error_precedes_device_work(self):
        from faiss.contrib import e_means_cute

        def lane_helper(tiled_copy, reference_src, tidx):
            return tiled_copy, reference_src, tidx

        modules = {
            "cuda.bindings.driver": types.SimpleNamespace(
                CUstream=lambda: None
            ),
            "quack.epilogue.ops": types.SimpleNamespace(
                _lane_warp_info_n=lane_helper
            ),
            "quack.gemm_sm90": types.SimpleNamespace(GemmSm90=object),
            "quack.tile_scheduler": types.SimpleNamespace(
                TileScheduler=object
            ),
            "quack.gemm_tvm_ffi_utils": types.SimpleNamespace(
                compile_gemm_kernel=lambda: None
            ),
        }
        versions = {
            "quack-kernels": "0.6.5",
            "nvidia-cutlass-dsl": "4.7.1",
            "cuda-python": "13.0.0",
        }
        for cuda_version in ("13.0.0", "13.99.1"):
            with self.subTest(cuda_python=cuda_version):
                versions["cuda-python"] = cuda_version
                e_means_cute._stack_status.cache_clear()
                with mock.patch.object(
                    importlib.metadata,
                    "version",
                    side_effect=versions.__getitem__,
                ), mock.patch.object(
                    importlib.util, "find_spec", return_value=object()
                ), mock.patch.object(
                    importlib, "import_module", side_effect=modules.__getitem__
                ):
                    self.assertEqual(
                        e_means_cute._stack_status(), (True, "ready")
                    )

        with mock.patch.object(
            e_means_cute,
            "_stack_status",
            return_value=(False, "test mismatch"),
        ), mock.patch.object(
            torch.cuda, "device", return_value=contextlib.nullcontext()
        ), self.assertRaisesRegex(RuntimeError, "test mismatch"):
            e_means_cute._check_device(torch.device("cuda"))

        e_means_cute._stack_status.cache_clear()
        with mock.patch.object(
            importlib.metadata,
            "version",
            side_effect=lambda name: (
                "0" if name == "quack-kernels" else "1"
            ),
        ):
            self.assertEqual(
                e_means_cute._stack_status(),
                (False, "quack-kernels version mismatch"),
            )
        e_means_cute._stack_status.cache_clear()
        versions = {
            "quack-kernels": "0.6.5",
            "nvidia-cutlass-dsl": "4.7.1",
            "cuda-python": "12.9.0",
        }
        with mock.patch.object(
            importlib.metadata, "version", side_effect=versions.__getitem__
        ):
            self.assertEqual(
                e_means_cute._stack_status(),
                (False, "cuda-python requires major version 13"),
            )
        e_means_cute._stack_status.cache_clear()
        with torch.inference_mode():
            untracked = torch.empty(1)
        self.assertIsNone(e_means_cute._tensor_version(untracked))

    def test_rejects_non_sm90_and_insufficient_shared_memory(self):
        from faiss.contrib import e_means_cute

        properties = mock.Mock(shared_memory_per_block_optin=232447)
        with mock.patch.object(
            torch.cuda, "device", return_value=contextlib.nullcontext()
        ):
            with mock.patch.object(
                e_means_cute, "_require_stack"
            ), mock.patch.object(
                torch.cuda, "get_device_capability", return_value=(8, 0)
            ), self.assertRaisesRegex(RuntimeError, "exactly SM90"):
                e_means_cute._check_device(torch.device("cuda"))
            with mock.patch.object(
                e_means_cute, "_require_stack"
            ), mock.patch.object(
                torch.cuda, "get_device_capability", return_value=(9, 0)
            ), mock.patch.object(
                torch.cuda, "get_device_properties", return_value=properties
            ), self.assertRaisesRegex(RuntimeError, "opt-in shared memory"):
                e_means_cute._check_device(torch.device("cuda"))
        self.assertEqual(e_means_cute._check_padded_span("row", 1, 9), 16)
        with self.assertRaisesRegex(ValueError, "padded row storage"):
            e_means_cute._check_padded_span("row", 2**28, 1)

    @unittest.skipUnless(_CUTE_SM90, _CUTE_REASON)
    def test_fp8_train_and_assign_keep_fp32_state(self):
        device = torch.device("cuda")
        low = torch.full((5, 65), 0.1, device=device)
        high = torch.full((5, 65), 2.3, device=device)
        x = torch.cat((low, high))
        init = torch.stack((low[0], high[0]))
        model = Emeans(
            65,
            2,
            niter=1,
            batch_size=10,
            alpha0=1.0,
            assignment_chunk_size=10,
            assignment_precision="float8_e4m3fn",
            assignment_backend="cute",
        )
        model.train(x, init_centroids=init)
        distances, labels = model.assign(x)

        self.assertEqual(model.centroids.dtype, torch.float32)
        self.assertEqual(distances.dtype, torch.float32)
        self.assertEqual(labels.dtype, torch.int64)
        self.assertEqual(labels.tolist(), [0] * 5 + [1] * 5)
        torch.testing.assert_close(model.centroids[0], low[0])
        torch.testing.assert_close(model.centroids[1], high[0])
        torch.testing.assert_close(distances, torch.zeros_like(distances))
        self.assertTrue(model.recipe["assignment_transform_centered"])

        query = low[:1].clone()
        query[0, 0] += 0.125
        query_distance, query_label = model.assign(query)
        self.assertEqual(query_label.item(), 0)
        torch.testing.assert_close(
            query_distance, query_distance.new_tensor([0.015625])
        )

    @unittest.skipUnless(_CUTE_SM90, _CUTE_REASON)
    def test_fp8_single_chunk_fusion_matches_multi_chunk(self):
        from faiss.contrib.e_means_fp8 import fit, validate_rows

        # d=128 exposes differing FP32 row-norm reduction orders; the former
        # 31x65 fixture did not detect the multi-chunk inconsistency.
        generator = torch.Generator(device="cuda").manual_seed(31)
        rows = torch.randn(
            (256, 128), generator=generator, device="cuda"
        ).mul_(0.25)
        centroids = torch.randn(
            (512, 128), generator=generator, device="cuda"
        ).mul_(0.25)
        model = Emeans(
            128,
            512,
            assignment_chunk_size=len(rows),
            assignment_precision="float8_e4m3fn",
            assignment_backend="cute",
        )
        model.centroids = centroids
        model._assignment_transform = fit(rows)

        with mock.patch(
            "faiss.contrib.e_means_fp8.validate_rows", wraps=validate_rows
        ) as separate_validation:
            observed = model.assign(rows)
        separate_validation.assert_not_called()

        model.assignment_chunk_size = 64
        with mock.patch(
            "faiss.contrib.e_means_fp8.validate_rows", wraps=validate_rows
        ) as separate_validation:
            expected = model.assign(rows)
        separate_validation.assert_called_once()

        self.assertTrue(torch.equal(observed[0], expected[0]))
        self.assertTrue(torch.equal(observed[1], expected[1]))

    @unittest.skipUnless(_CUTE_SM90, _CUTE_REASON)
    def test_fp8_range_validation_precedes_assignment(self):
        device = torch.device("cuda")
        x = torch.arange(8, dtype=torch.float32, device=device)[:, None]
        model = Emeans(
            1, 2, niter=1, batch_size=8, alpha0=0.5,
            assignment_chunk_size=2,
            assignment_precision="float8_e4m3fn",
            assignment_backend="cute",
        )
        model.train(x, init_centroids=x[[0, -1]])

        cases = ((0, 1000.0), (3, 1000.0), (5, 1000.0),
                 (2, float("nan")), (4, float("inf")),
                 (1, float("-inf")))
        for position, value in cases:
            with self.subTest(position=position, value=value):
                query = torch.zeros((6, 1), device=device)
                query[position] = value
                with mock.patch(
                    "faiss.contrib.e_means_fp8.validate_rows"
                ) as validate_rows, mock.patch(
                    "faiss.contrib.e_means_cute_kernel.launch_tiled_gemm"
                ) as launch_gemm, self.assertRaisesRegex(
                    ValueError, "exceeds the E4M3 representable range"
                ):
                    model.assignment_chunk_size = len(query)
                    model.assign(query)
                validate_rows.assert_not_called()
                launch_gemm.assert_not_called()

    @unittest.skipUnless(_CUTE_SM90, _CUTE_REASON)
    def test_ragged_oracle_ties_and_determinism(self):
        device = torch.device("cuda")
        centroids = torch.full((193, 17), 10.0, device=device)
        centroids[0].zero_()
        centroids[5].fill_(3.0)
        centroids[10].zero_()
        centroids[10, 0] = 0.99609375
        centroids[11].zero_()
        centroids[11, 0] = 1.00390625
        centroids[127].zero_()
        centroids[192].fill_(-3.0)
        rows = torch.stack(
            (
                centroids[5],
                centroids[192],
                torch.nn.functional.pad(
                    torch.ones(1, device=device), (0, 16)
                ),
                torch.full((17,), -0.0, device=device),
            )
        )
        model = Emeans(
            17,
            193,
            assignment_precision="bfloat16",
            assignment_backend="cute",
        )
        model.centroids = centroids
        before_rows = rows.clone()
        before_centroids = centroids.clone()

        observed = [model.assign(rows) for _ in range(4)]
        for distances, labels in observed[1:]:
            torch.testing.assert_close(
                distances, observed[0][0], rtol=0, atol=0
            )
            torch.testing.assert_close(
                labels, observed[0][1], rtol=0, atol=0
            )
        distances, labels = observed[0]
        self.assertEqual(labels[:2].tolist(), [5, 192])
        self.assertIn(labels[2].item(), (10, 11))
        self.assertEqual(labels[3].item(), 0)

        rounded_rows = rows.to(torch.bfloat16).double()
        rounded_centroids = centroids.to(torch.bfloat16).double()
        oracle = (
            rounded_rows[:, None] - rounded_centroids
        ).square().sum(dim=2)
        selected = oracle.gather(1, labels[:, None]).squeeze(1)
        torch.testing.assert_close(
            distances.double(), selected, rtol=0, atol=1e-5
        )
        torch.testing.assert_close(rows, before_rows, rtol=0, atol=0)
        torch.testing.assert_close(
            centroids, before_centroids, rtol=0, atol=0
        )

    @unittest.skipUnless(_CUTE_SM90, _CUTE_REASON)
    def test_large_batch_float_ties_and_planted_winners(self):
        from faiss.contrib.e_means_fp8 import fit

        device = torch.device("cuda")
        batch, clusters, dim = 8192, 16384, 96
        centroids = torch.full((clusters, dim), 4.0, device=device)
        centroids[0].zero_()
        centroids[128].zero_()
        centroids[777].fill_(1)
        centroids[1024].fill_(1)
        centroids[-1].fill_(-1)
        rows = torch.zeros((batch, dim), device=device)
        rows[1].fill_(1)
        rows[2].fill_(-1)
        rows[3].fill_(0.25)
        for precision in ("bfloat16", "float8_e4m3fn"):
            with self.subTest(precision=precision):
                model = Emeans(
                    dim,
                    clusters,
                    assignment_chunk_size=batch,
                    assignment_precision=precision,
                    assignment_backend="cute",
                )
                model.centroids = centroids
                if precision == "float8_e4m3fn":
                    model._assignment_transform = fit(rows[:4])

                distances, labels = model.assign(rows)

                self.assertEqual(
                    labels[:4].tolist(), [0, 777, clusters - 1, 0]
                )
                torch.testing.assert_close(
                    distances[:4],
                    torch.tensor([0.0, 0.0, 0.0, 6.0], device=device),
                    rtol=0,
                    atol=1e-5,
                )
                self.assertTrue(bool(torch.all(labels[4:] == 0)))
                torch.testing.assert_close(
                    distances[4:], torch.zeros_like(distances[4:]),
                    rtol=0, atol=1e-5,
                )

    @unittest.skipUnless(_CUTE_SM90, _CUTE_REASON)
    def test_prepared_refresh_reuses_compatible_storage(self):
        from faiss.contrib import e_means_cute

        device = torch.device("cuda")
        source = torch.zeros((2, 17), device=device)
        source[1].fill_(2)
        rows = torch.zeros((4, 17), device=device)
        prepared = e_means_cute._prepare_centroids(source)
        e_means_cute._assign_rows(rows, prepared)
        binding = prepared.bindings[4]
        storage = [
            value.data_ptr() for value in (
                prepared.centroids, prepared.norms, prepared.padded_norms
            )
        ]

        refreshed_source = source.flip(0).contiguous()
        expected_norms = refreshed_source.to(
            torch.bfloat16
        ).float().square().sum(dim=1)
        refreshed = e_means_cute._prepare_centroids(
            refreshed_source, prepared
        )
        self.assertIs(refreshed, prepared)
        self.assertEqual(
            storage,
            [value.data_ptr() for value in (
                refreshed.centroids, refreshed.norms, refreshed.padded_norms
            )],
        )
        self.assertIs(refreshed.bindings[4], binding)
        torch.testing.assert_close(
            refreshed.norms, expected_norms, rtol=0, atol=0
        )
        _, refreshed_labels = e_means_cute._assign_rows(rows, refreshed)
        self.assertEqual(refreshed_labels.tolist(), [1, 1, 1, 1])

        with mock.patch.object(
            e_means_cute, "_refresh_prepared", side_effect=AssertionError
        ):
            self.assertIs(
                e_means_cute._prepare_centroids(refreshed_source, refreshed),
                refreshed,
            )
        refreshed_source.add_(1)
        self.assertIs(
            e_means_cute._prepare_centroids(refreshed_source, refreshed),
            refreshed,
        )
        torch.testing.assert_close(
            refreshed.centroids, refreshed_source.to(torch.bfloat16)
        )

        rebuilt = e_means_cute._prepare_centroids(
            torch.zeros((3, 17), device=device), refreshed
        )
        self.assertIsNot(rebuilt, refreshed)
        self.assertNotEqual(rebuilt.centroids.data_ptr(), storage[0])

    @unittest.skipUnless(_CUTE_SM90, _CUTE_REASON)
    def test_workspace_lifetime_reprepare_and_stream_ordering(self):
        from faiss.contrib.e_means_cute import (
            _assign_rows,
            _prepare_centroids,
        )

        device = torch.device("cuda")
        centroids = torch.stack(
            (torch.zeros(17, device=device), torch.ones(17, device=device))
        )
        prepared = _prepare_centroids(centroids.contiguous())
        with self.assertRaisesRegex(ValueError, "contiguous CUDA FP32"):
            _prepare_centroids(centroids.t())
        stream = torch.cuda.Stream()
        rows = torch.zeros((4, 17), device=device)
        with self.assertRaisesRegex(ValueError, "contiguous CUDA FP32"):
            _assign_rows(rows.to(torch.float64), prepared)
        with torch.cuda.stream(stream):
            first_distance, first_labels = _assign_rows(rows, prepared)
            saved_distance = first_distance.clone()
            saved_labels = first_labels.clone()
            _assign_rows(torch.ones_like(rows), prepared)
            _assign_rows(rows[:3].contiguous(), prepared)
            _assign_rows(rows[:2].contiguous(), prepared)
        stream.synchronize()
        torch.testing.assert_close(
            first_distance, saved_distance, rtol=0, atol=0
        )
        torch.testing.assert_close(first_labels, saved_labels, rtol=0, atol=0)
        refresh_source = centroids.clone()
        with torch.cuda.stream(stream):
            torch.cuda._sleep(500_000_000)
            refresh_source[0].fill_(3)
            refresh_source[1].zero_()
            changed = _prepare_centroids(refresh_source, prepared)
            _, changed_labels = _assign_rows(rows, changed)
        stream.synchronize()
        self.assertIs(changed, prepared)
        self.assertEqual(changed_labels.tolist(), [1, 1, 1, 1])
        torch.testing.assert_close(
            first_distance, saved_distance, rtol=0, atol=0
        )
        torch.testing.assert_close(first_labels, saved_labels, rtol=0, atol=0)

    @unittest.skipUnless(_CUTE_SM90, _CUTE_REASON)
    def test_gemm_and_decode_use_the_current_stream(self):
        from faiss.contrib.e_means_cute import (
            _assign_rows,
            _prepare_centroids,
        )

        device = torch.device("cuda")
        centroids = torch.stack(
            (
                torch.zeros(17, device=device),
                torch.full((17,), 2.0, device=device),
            )
        )
        prepared = _prepare_centroids(centroids.contiguous())
        rows = torch.zeros((257, 17), device=device)
        _assign_rows(rows, prepared)
        torch.cuda.synchronize()

        side = torch.cuda.Stream()
        with torch.cuda.stream(side):
            # The delay must outlast the tiny misplaced default-stream launch:
            # such a launch reads the old zero workspace before this side's
            # copy, while a correct current-stream launch waits behind it.
            torch.cuda._sleep(500_000_000)
            distances, labels = _assign_rows(
                torch.full_like(rows, 2.0), prepared
            )
        side.synchronize()
        self.assertEqual(labels.tolist(), [1] * len(rows))
        torch.testing.assert_close(distances, torch.zeros_like(distances))

    @unittest.skipUnless(_TWO_SM90, "two SM90 GPUs are required")
    def test_noncurrent_device_is_scoped(self):
        from faiss.contrib.e_means_cute import _prepare_centroids

        torch.cuda.set_device(0)
        torch.cuda.synchronize(0)
        cached0 = _prepare_centroids(
            torch.zeros((2, 17), device="cuda:0")
        )
        baseline0 = torch.cuda.memory_allocated(0)
        device = torch.device("cuda:1")
        centroids = torch.stack(
            (
                torch.zeros(17, device=device),
                torch.full((17,), 2.0, device=device),
            )
        )
        rows = torch.full((257, 17), 2.0, device=device)
        model = Emeans(
            17,
            2,
            assignment_precision="bfloat16",
            assignment_backend="cute",
            assignment_chunk_size=257,
        )
        model.centroids = centroids
        self.assertEqual(torch.cuda.current_device(), 0)
        distances, labels = model.assign(rows)
        torch.cuda.synchronize(device)

        self.assertEqual(torch.cuda.current_device(), 0)
        self.assertEqual((distances.device, labels.device), (device, device))
        self.assertEqual(labels.tolist(), [1] * len(rows))
        torch.testing.assert_close(distances, torch.zeros_like(distances))
        self.assertEqual(torch.cuda.memory_allocated(0), baseline0)
        rebuilt1 = _prepare_centroids(centroids.contiguous(), cached0)
        self.assertIsNot(rebuilt1, cached0)
        self.assertEqual(rebuilt1.centroids.device, device)
        self.assertEqual(torch.cuda.current_device(), 0)

    @unittest.skipUnless(_CUTE_SM90, _CUTE_REASON)
    def test_two_pass_revival_is_deterministic(self):
        from faiss.contrib import e_means_cute

        x = torch.zeros((10, 17), device="cuda")
        x[:, 0] = torch.tensor(
            [0, 1, 2, 3, 4, 10, 11, 12, 13, 14],
            dtype=torch.float32,
            device="cuda",
        )
        init = torch.zeros((3, 17), device="cuda")
        init[:, 0] = torch.tensor([0, 10, 100], device="cuda")
        args = dict(
            d=17,
            k=3,
            niter=2,
            batch_size=2,
            alpha0=0.5,
            assignment_chunk_size=2,
            assignment_precision="bfloat16",
            assignment_backend="cute",
        )
        first = Emeans(**args)
        with mock.patch.object(
            e_means_cute,
            "_new_binding",
            wraps=e_means_cute._new_binding,
        ) as new_binding:
            first.train(x, init_centroids=init)
            distances, labels = first.assign(x)
            self.assertEqual(new_binding.call_count, 1)

        self.assertGreater(first.n_splits, 0)
        self.assertTrue(bool(torch.isfinite(first.centroids).all()))
        self.assertTrue(bool(torch.isfinite(first.obj).all()))
        for cluster in labels.unique():
            selected = labels == cluster
            torch.testing.assert_close(
                first.centroids[cluster],
                x[selected].mean(dim=0),
                rtol=0,
                atol=0,
            )
        closing_objective = distances.double().sum().item()
        self.assertGreater(abs(first.obj[-1].item() - closing_objective), 1e-3)
        self.assertEqual(first.recipe["assignment_backend"], "cute")

        first.centroids[0].fill_(100)
        first.centroids[1].zero_()
        _, relabeled = first.assign(x[:1])
        self.assertEqual(relabeled.item(), 1)


if __name__ == "__main__":
    unittest.main()
