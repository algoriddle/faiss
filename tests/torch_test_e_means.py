# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import importlib.util
import sys
import unittest
from unittest import mock

import torch

from faiss.contrib.e_means import Emeans


_REDUCED_PRECISION_CUDA = (
    torch.cuda.is_available()
    and getattr(torch.version, "hip", None) is None
    and torch.cuda.get_device_capability() >= (8, 0)
)
_TRITON_CUDA = (
    _REDUCED_PRECISION_CUDA
    and importlib.util.find_spec("triton") is not None
)
_FP8_TRITON_CUDA = (
    _TRITON_CUDA
    and hasattr(torch, "float8_e4m3fn")
    and torch.cuda.get_device_capability() in ((9, 0), (10, 0), (10, 3))
)
_INT8_TRITON_CUDA = (
    _TRITON_CUDA
    and torch.cuda.get_device_capability() in ((8, 0), (9, 0), (10, 0))
)


class TestTorchEmeans(unittest.TestCase):
    def test_assignment_backend_contract(self):
        with self.assertRaisesRegex(ValueError, "assignment_backend"):
            Emeans(2, 2, assignment_backend="other")

        sys.modules.pop("faiss.contrib.e_means_triton", None)
        model = Emeans(2, 2, niter=1, batch_size=2, alpha0=0.5)
        model.train(torch.zeros(2, 2))
        self.assertNotIn("faiss.contrib.e_means_triton", sys.modules)
        with mock.patch.dict(
            sys.modules, {"faiss.contrib.e_means_triton": None}
        ):
            for precision in ("bfloat16", "int8"):
                with self.subTest(
                    missing_triton=precision
                ), self.assertRaisesRegex(
                    ImportError, "requires the triton package"
                ):
                    x = torch.zeros(2, 2)
                    if precision == "int8":
                        x = x.to(torch.int8)
                    Emeans(
                        2,
                        2,
                        assignment_precision=precision,
                        assignment_backend="triton",
                    ).train(x)
        with self.assertRaisesRegex(ValueError, "requires Triton"):
            Emeans(2, 2, assignment_precision="float8_e4m3fn")
        with self.assertRaisesRegex(ValueError, "requires Triton"):
            Emeans(2, 2, assignment_precision="int8")

    def test_int8_support_contract(self):
        from faiss.contrib.e_means_int8 import _check_int8_support

        device = torch.device("cuda")
        for capability in ((8, 9), (10, 3)):
            with self.subTest(capability=capability), mock.patch.object(
                torch.cuda, "get_device_capability", return_value=capability
            ), self.assertRaisesRegex(RuntimeError, "SM80, SM90, or SM100"):
                _check_int8_support(device)
        with mock.patch.object(
            torch.version, "hip", "6.0"
        ), mock.patch.object(
            torch.cuda, "get_device_capability", return_value=(9, 0)
        ), self.assertRaisesRegex(RuntimeError, "NVIDIA"):
            _check_int8_support(device)
        with self.assertRaisesRegex(ValueError, "d <= 43919"):
            Emeans(
                43920,
                2,
                assignment_precision="int8",
                assignment_backend="triton",
            )
        with self.assertRaisesRegex(TypeError, "INT8 or UINT8 input"):
            Emeans(
                2,
                2,
                assignment_precision="int8",
                assignment_backend="triton",
            ).train(torch.zeros(2, 2))

    def test_int8_quantizer(self):
        from faiss.contrib.e_means_int8 import Int8Quantizer

        x = torch.tensor(
            [[10, 100], [12, 104], [14, 108], [16, 112]],
            dtype=torch.float32,
        )
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            with self.subTest(train_dtype=dtype):
                trained = Int8Quantizer(2)
                trained.train(x.to(dtype))
                self.assertEqual((trained.d, trained.code_size), (2, 2))
                self.assertTrue(trained.is_trained)
                self.assertEqual(trained.scale, 16.0)
                torch.testing.assert_close(
                    trained.shift, torch.tensor([13.0, 106.0])
                )

        quantizer = Int8Quantizer(2)
        with self.assertRaisesRegex(RuntimeError, "must be trained"):
            quantizer.compute_codes(x)
        with self.assertRaisesRegex(ValueError, "wrong dimension"):
            Int8Quantizer(3).train(x)
        quantizer.train(x)
        expected = torch.tensor(
            [[-48, -96], [-16, -32], [16, 32], [48, 96]],
            dtype=torch.int8,
        )
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            with self.subTest(compute_codes_dtype=dtype):
                encoded = quantizer.compute_codes(x.to(dtype))
                torch.testing.assert_close(encoded, expected)
                torch.testing.assert_close(quantizer.decode(encoded), x)

        state = expected.float().mean(dim=0, keepdim=True)
        torch.testing.assert_close(
            quantizer.decode(state), torch.tensor([[13.0, 106.0]])
        )
        with self.assertRaisesRegex(ValueError, "wrong dimension"):
            quantizer.compute_codes(torch.zeros(1, 3))
        with self.assertRaisesRegex(ValueError, "wrong dimension"):
            quantizer.decode(torch.zeros(1, 3, dtype=torch.int8))
        with self.assertRaisesRegex(ValueError, "must be finite"):
            quantizer.decode(torch.tensor([[float("nan"), 0.0]]))
        if torch.cuda.is_available():
            with self.assertRaisesRegex(ValueError, "same device"):
                quantizer.compute_codes(x.cuda())

        failed = Int8Quantizer(2)
        failed.train(x)
        invalid = x.clone()
        invalid[0, 0] = torch.nan
        with self.assertRaisesRegex(ValueError, "sample must be finite"):
            failed.train(invalid)
        self.assertFalse(failed.is_trained)
        self.assertEqual(
            (failed.shift, failed.scale, failed._bias), (None,) * 3
        )
        with self.assertRaisesRegex(RuntimeError, "must be trained"):
            failed.compute_codes(x)
        with self.assertRaisesRegex(RuntimeError, "must be trained"):
            failed.decode(torch.zeros(1, 2, dtype=torch.int8))

    def test_int8_quantizer_codes_contract(self):
        from faiss.contrib.e_means_int8 import Int8Quantizer

        sample = torch.tensor(
            [[10.0, 100.0], [12.0, 104.0], [14.0, 108.0], [16.0, 112.0]]
        )
        quantizer = Int8Quantizer(2)
        quantizer.train(sample)

        clipped = quantizer.compute_codes(torch.tensor([[1e6, -1e6]]))
        torch.testing.assert_close(
            clipped, torch.tensor([[127, -128]], dtype=torch.int8)
        )
        with self.assertRaisesRegex(ValueError, "finite"):
            quantizer.compute_codes(torch.tensor([[float("nan"), 0.0]]))

        points = torch.tensor([[13.02, 106.02], [13.98, 107.98]])
        codes = quantizer.compute_codes(points)
        byte_distance = (codes[0].int() - codes[1].int()).square().sum()
        decoded = quantizer.decode(codes)
        decoded_distance = (decoded[0] - decoded[1]).square().sum()
        torch.testing.assert_close(
            byte_distance / quantizer.scale**2, decoded_distance
        )
        original_distance = (points[0] - points[1]).square().sum()
        self.assertFalse(torch.isclose(decoded_distance, original_distance))

    def test_fp8_support_contract(self):
        from faiss.contrib.e_means_fp8 import _check_fp8_support

        device = torch.device("cuda")
        for capability in ((8, 0), (8, 9)):
            with self.subTest(capability=capability), mock.patch.object(
                torch.cuda, "get_device_capability", return_value=capability
            ), self.assertRaisesRegex(RuntimeError, "SM90, SM100, or SM103"):
                _check_fp8_support(device)
        with mock.patch.object(
            torch.version, "hip", "6.0"
        ), mock.patch.object(
            torch.cuda, "get_device_capability", return_value=(9, 0)
        ), self.assertRaisesRegex(RuntimeError, "NVIDIA"):
            _check_fp8_support(device)
        with mock.patch.object(
            torch, "float8_e4m3fn", None
        ), self.assertRaisesRegex(RuntimeError, "does not provide"):
            _check_fp8_support(device)

    def test_closing_mean_and_assign(self):
        x = torch.tensor(
            [[0.0], [2.0], [10.0], [12.0]], requires_grad=True
        )
        init = torch.tensor([[0.0], [10.0]], requires_grad=True)
        km = Emeans(1, 2, niter=1, batch_size=4, alpha0=0.5)

        objective = km.train(x, init_centroids=init)

        torch.testing.assert_close(km.centroids, torch.tensor([[1.0], [11.0]]))
        self.assertEqual(objective, 8.0)
        self.assertEqual(km.recipe["assignment_chunk_size"], 8192)
        self.assertFalse(km.centroids.requires_grad)
        km.centroids.requires_grad_()
        distance, labels = km.assign(x)
        torch.testing.assert_close(distance, torch.ones(4))
        torch.testing.assert_close(labels, torch.tensor([0, 0, 1, 1]))
        self.assertFalse(distance.requires_grad)

    def test_alpha_one_preserves_an_empty_cluster(self):
        x = torch.tensor([[99.0], [0.0], [1.0], [100.0]])
        init = torch.tensor([[0.0], [100.0]])
        km = Emeans(1, 2, niter=1, batch_size=2, alpha0=1.0)

        km.train(x, init_centroids=init)

        torch.testing.assert_close(
            km.centroids, torch.tensor([[0.5], [99.5]])
        )

    def test_closing_mean_uses_only_the_final_pass(self):
        x = torch.tensor([[3.0], [4.0], [4.0], [6.0], [12.0]])
        init = torch.tensor([[0.0], [10.0]])
        km = Emeans(1, 2, niter=2, batch_size=5, alpha0=1.0)

        km.train(x, init_centroids=init)

        torch.testing.assert_close(
            km.centroids, torch.tensor([[4.25], [12.0]])
        )

    def test_blocked_assignment_matches_cdist(self):
        generator = torch.Generator().manual_seed(123)
        x = torch.randn(23, 7, generator=generator)
        centroids = torch.randn(11, 7, generator=generator)
        km = Emeans(7, 11, centroid_chunk_size=3)
        km.centroids = centroids

        distance, labels = km.assign(x)
        reference = torch.cdist(x, centroids).square()
        expected_distance, expected_labels = reference.min(dim=1)

        torch.testing.assert_close(
            distance, expected_distance, atol=2e-5, rtol=1e-5
        )
        torch.testing.assert_close(labels, expected_labels)

    def test_blocked_assignment_breaks_ties_by_index(self):
        x = torch.zeros((1, 2))
        centroids = torch.ones((5, 2))
        km = Emeans(2, 5, centroid_chunk_size=2)
        km.centroids = centroids

        _, labels = km.assign(x)

        self.assertEqual(labels.item(), 0)

    def test_revival_occurs_during_training(self):
        x = torch.zeros((10, 1))
        init = torch.tensor([[0.0], [10.0]])
        km = Emeans(1, 2, niter=5, batch_size=10, alpha0=0.5)

        km.train(x, init_centroids=init)

        self.assertEqual(km.n_splits, 1)

    def test_chunking_does_not_change_updates(self):
        generator = torch.Generator().manual_seed(123)
        x = torch.randn(41, 5, generator=generator)
        init = x[:4].clone()
        args = {"niter": 3, "batch_size": 13, "alpha0": 0.2, "seed": 7}
        a = Emeans(
            5,
            4,
            assignment_chunk_size=4,
            centroid_chunk_size=2,
            **args,
        )
        b = Emeans(
            5,
            4,
            assignment_chunk_size=64,
            centroid_chunk_size=8,
            **args,
        )

        a.train(x, init_centroids=init)
        b.train(x, init_centroids=init)

        torch.testing.assert_close(
            a.centroids, b.centroids, atol=2e-6, rtol=1e-6
        )
        torch.testing.assert_close(a.obj, b.obj, atol=2e-5, rtol=1e-6)

    def test_training_does_not_advance_global_rng(self):
        x = torch.arange(80, dtype=torch.float32).reshape(20, 4)
        state = torch.random.get_rng_state()
        km = Emeans(4, 3, niter=1, batch_size=7, alpha0=0.2)

        km.train(x)

        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))

    def test_input_types_match(self):
        x = torch.randn(20, 4)
        init = x[:3].clone()
        km = Emeans(4, 3, niter=1, batch_size=7, alpha0=0.2)

        with self.assertRaises(TypeError):
            km.train(x.numpy(), init_centroids=init)
        with self.assertRaises(TypeError):
            km.train(x, init_centroids=init.numpy())

    @unittest.skipUnless(
        _REDUCED_PRECISION_CUDA, "reduced-precision CUDA not available"
    )
    def test_float16_range_contract(self):
        device = torch.device("cuda")
        valid = torch.zeros((2, 1), device=device)
        out_of_range = torch.tensor([[65505.0]], device=device)
        km = Emeans(1, 1, assignment_precision="float16")

        with self.subTest(value="train x"):
            with self.assertRaisesRegex(ValueError, "65504"):
                km.train(torch.cat((valid[:1], out_of_range)))

        with self.subTest(value="NaN train x"):
            nan = torch.tensor([[0.0], [float("nan")]], device=device)
            with self.assertRaisesRegex(ValueError, "65504"):
                km.train(nan)

        with self.subTest(value="initial centroid"):
            with self.assertRaisesRegex(ValueError, "65504"):
                km.train(valid, init_centroids=out_of_range)

        with self.subTest(value="assign x"):
            km.centroids = valid[:1]
            with self.assertRaisesRegex(ValueError, "65504"):
                km.assign(out_of_range)

        with self.subTest(value="installed centroid"):
            km.centroids = out_of_range
            with self.assertRaisesRegex(ValueError, "65504"):
                km.assign(valid[:1])

        with self.subTest(value="float16 limit"):
            limit = torch.full((2, 1), 65504.0, device=device)
            km = Emeans(
                1, 1, niter=1, batch_size=2, alpha0=0.5,
                assignment_precision="float16",
            )
            km.train(limit, init_centroids=limit[:1])
            km.assign(limit[:1])
            torch.testing.assert_close(km.centroids, limit[:1])

        with self.subTest(value="bfloat16 above float16 limit"):
            large = torch.full((2, 1), 70000.0, device=device)
            km = Emeans(
                1, 1, niter=1, batch_size=2, alpha0=0.5,
                assignment_precision="bfloat16",
            )
            km.train(large, init_centroids=large[:1])
            km.assign(large[:1])

    @unittest.skipUnless(
        _REDUCED_PRECISION_CUDA, "reduced-precision CUDA not available"
    )
    def test_float16_revival_checks_range_before_commit(self):
        x = torch.full((10, 1), 65504.0, device="cuda")
        init = x[:2].clone()
        km = Emeans(
            1, 2, niter=1, batch_size=2, alpha0=0.5,
            assignment_precision="float16",
        )

        with self.assertRaisesRegex(ValueError, "65504"):
            km.train(x, init_centroids=init)

        self.assertEqual(km.n_splits, 0)

    @unittest.skipUnless(
        _REDUCED_PRECISION_CUDA, "reduced-precision CUDA not available"
    )
    def test_reduced_precision_assignment(self):
        x = torch.tensor(
            [[1.001], [1.002], [3.001], [3.002]], device="cuda"
        )
        init = torch.tensor([[1.0], [3.0]], device="cuda")
        for precision in ("float16", "bfloat16"):
            with self.subTest(precision=precision):
                km = Emeans(
                    1,
                    2,
                    niter=1,
                    batch_size=len(x),
                    alpha0=0.5,
                    assignment_precision=precision,
                )

                km.train(x, init_centroids=init)
                distance, labels = km.assign(x)

                torch.testing.assert_close(
                    km.centroids,
                    torch.tensor([[1.0015], [3.0015]], device="cuda"),
                )
                self.assertEqual(
                    km.recipe["assignment_precision"], precision
                )
                self.assertEqual(
                    (km.centroids.dtype, distance.dtype, labels.dtype),
                    (torch.float32, torch.float32, torch.int64),
                )

    @unittest.skipUnless(_TRITON_CUDA, "Triton CUDA not available")
    def test_triton_split_k_breaks_ties_by_index(self):
        km = Emeans(
            1, 8065, assignment_precision="bfloat16",
            assignment_backend="triton",
        )
        km.centroids = torch.ones(8065, 1, device="cuda")
        km.centroids[126:128] = 0
        distance, labels = km.assign(torch.zeros(1, 1, device="cuda"))
        self.assertEqual((distance.item(), labels.item()), (0.0, 126))

    @unittest.skipUnless(_TRITON_CUDA, "Triton CUDA not available")
    def test_triton_assignment(self):
        x = torch.tensor(
            [1.001, 1.002, 3.001, 3.002], device="cuda"
        )[:, None]
        init = x[[0, 2]]
        for precision in ("float32", "float16", "bfloat16"):
            with self.subTest(precision=precision):
                km = Emeans(
                    1, 2, niter=1, batch_size=4, alpha0=0.5,
                    assignment_precision=precision,
                    assignment_backend="triton",
                )
                km.train(x, init_centroids=init)
                distance, labels = km.assign(x)

                self.assertEqual(km.recipe["assignment_backend"], "triton")
                self.assertEqual(labels.tolist(), [0, 0, 1, 1])
                self.assertEqual(
                    (km.centroids.dtype, distance.dtype, labels.dtype),
                    (torch.float32, torch.float32, torch.int64),
                )

    @unittest.skipUnless(_FP8_TRITON_CUDA, "Triton FP8 not available")
    def test_triton_fp8_assignment(self):
        x = torch.tensor([[0.0], [2.0], [10.0], [12.0]], device="cuda")
        init = x[[0, 2]]
        km = Emeans(
            1,
            2,
            niter=1,
            batch_size=4,
            alpha0=0.5,
            assignment_precision="float8_e4m3fn",
            assignment_backend="triton",
        )

        objective = km.train(x, init_centroids=init)
        distance, labels = km.assign(x)

        self.assertEqual(objective, 8.0)
        torch.testing.assert_close(
            km.centroids, torch.tensor([[1.0], [11.0]], device="cuda")
        )
        torch.testing.assert_close(distance, torch.ones_like(distance))
        torch.testing.assert_close(
            labels, torch.tensor([0, 0, 1, 1], device="cuda")
        )
        self.assertEqual(km.recipe["assignment_transform_scale"], 8.0)
        self.assertTrue(km.recipe["assignment_transform_centered"])
        self.assertEqual(
            km.recipe["assignment_transform_decision"],
            "small_sample_centered",
        )
        self.assertEqual(
            (km.centroids.dtype, distance.dtype, labels.dtype),
            (torch.float32, torch.float32, torch.int64),
        )

        statistics_x = torch.tensor(
            [[0.1], [2.2], [10.3], [12.4]], device="cuda"
        )
        km.train(statistics_x, init_centroids=statistics_x[[0, 2]])
        transformed = km._assignment_transform.apply(statistics_x)
        rounded = transformed.to(torch.float8_e4m3fn).float()
        self.assertFalse(torch.equal(transformed, rounded))
        torch.testing.assert_close(
            km.centroids, torch.tensor([[1.15], [11.35]], device="cuda")
        )

        with self.assertRaisesRegex(ValueError, "representable range"):
            km.assign(torch.tensor([[100.0]], device="cuda"))
        km.centroids.fill_(100.0)
        with self.assertRaisesRegex(ValueError, "representable range"):
            km.assign(statistics_x[:1])

        clamped_x = torch.full((2, 1), 1000.0, device="cuda")
        clamped = Emeans(
            1, 2, niter=3, batch_size=2, alpha0=0.5,
            assignment_precision="float8_e4m3fn",
            assignment_backend="triton",
        )
        # The untouched centroid crosses the E4M3 limit on pass 3.
        with self.assertRaisesRegex(ValueError, "representable range"):
            clamped.train(clamped_x, init_centroids=clamped_x.clone())

        # Only the fifth-batch split exceeds E4M3, isolating pass validation.
        revival_x = torch.full((10, 1), 1e7, device="cuda")
        revival_init = torch.tensor(
            [[1e7], [1e7 + 400]], device="cuda"
        )
        revival = Emeans(
            1, 2, niter=1, batch_size=2, alpha0=0.05,
            assignment_precision="float8_e4m3fn",
            assignment_backend="triton",
        )
        with self.assertRaisesRegex(ValueError, "representable range"):
            revival.train(revival_x, init_centroids=revival_init)
        self.assertEqual(revival.n_splits, 1)

        with self.assertRaisesRegex(ValueError, "CUDA tensor"):
            km.train(x.cpu())
        self.assertIsNone(km.centroids)
        with self.assertRaisesRegex(RuntimeError, "train must be called"):
            km.assign(x)

    @unittest.skipUnless(_FP8_TRITON_CUDA, "Triton FP8 not available")
    def test_triton_fp8_uses_matched_rounded_norms(self):
        from faiss.contrib.e_means_triton import assign_rows
        from faiss.contrib.e_means_triton import prepare_fp8_centroids

        scale = 8.0
        rows = torch.tensor(
            [[-8.5485334, 0.5754005, 5.9468751]], device="cuda"
        ) / scale
        centroids = torch.tensor(
            [
                [1.3439447, -6.8812032, 5.9468989],
                [1.6238794, -0.7164297, -0.9630429],
            ],
            device="cuda",
        ) / scale
        prepared, norm = prepare_fp8_centroids(centroids, None, scale)

        distance, label = assign_rows(
            rows, prepared, norm, scale=scale
        )
        mixed_norm = (centroids * scale).square().sum(dim=1)
        _, mixed_label = assign_rows(
            rows, prepared, mixed_norm, scale=scale
        )

        expected = (
            (rows * scale).to(torch.float8_e4m3fn).float()
            - prepared[label, : rows.shape[1]].float()
        ).square().sum() / scale**2
        torch.testing.assert_close(distance[0], expected)
        self.assertEqual((label.item(), mixed_label.item()), (1, 0))

    @unittest.skipUnless(_FP8_TRITON_CUDA, "Triton FP8 not available")
    def test_triton_fp8_routes_and_ties(self):
        full_x = torch.cat(
            (
                torch.zeros(512, 65, device="cuda"),
                torch.ones(512, 65, device="cuda"),
            )
        )
        full = Emeans(
            65,
            2,
            niter=1,
            batch_size=1024,
            alpha0=0.5,
            assignment_precision="float8_e4m3fn",
            assignment_backend="triton",
        )
        full.train(x=full_x, init_centroids=full_x[[0, -1]])
        full_distance, full_labels = full.assign(full_x)
        torch.testing.assert_close(
            full_distance, torch.zeros_like(full_distance)
        )
        torch.testing.assert_close(
            full_labels,
            torch.cat(
                (
                    torch.zeros(512, dtype=torch.int64, device="cuda"),
                    torch.ones(512, dtype=torch.int64, device="cuda"),
                )
            ),
        )
        full.centroids.fill_(5.0)
        tie_distance, tie_label = full.assign(
            torch.full((1, 65), 5.0, device="cuda")
        )
        self.assertEqual((tie_distance.item(), tie_label.item()), (0.0, 0))

        split_x = torch.zeros(8065, 1, device="cuda")
        split = Emeans(
            1,
            8065,
            niter=1,
            batch_size=8065,
            alpha0=0.5,
            assignment_precision="float8_e4m3fn",
            assignment_backend="triton",
        )
        split.train(split_x)
        split_distance, split_label = split.assign(split_x[:1])
        self.assertEqual(
            (split_distance.item(), split_label.item()), (0.0, 0)
        )
        self.assertEqual(
            split.recipe["assignment_transform_decision"],
            "uncentered_near_parity",
        )

    @unittest.skipUnless(_INT8_TRITON_CUDA, "Triton INT8 not available")
    def test_triton_int8_assignment(self):
        base = torch.tensor([-4, -2, 4, 6], device="cuda")[:, None]
        cases = (
            (torch.int8, 0, "int8"),
            (torch.uint8, 128, "uint8"),
        )
        for dtype, offset, representation in cases:
            with self.subTest(dtype=dtype):
                x = (base + offset).to(dtype)
                km = Emeans(
                    1,
                    2,
                    niter=1,
                    batch_size=4,
                    alpha0=0.5,
                    assignment_precision="int8",
                    assignment_backend="triton",
                )

                objective = km.train(x, init_centroids=x[[0, 2]])
                distance, labels = km.assign(x)

                self.assertEqual(objective, 8.0)
                torch.testing.assert_close(
                    km.centroids,
                    torch.tensor(
                        [[-3 + offset], [5 + offset]],
                        dtype=torch.float32,
                        device="cuda",
                    ),
                )
                torch.testing.assert_close(
                    distance, torch.ones(4, device="cuda")
                )
                torch.testing.assert_close(
                    labels, torch.tensor([0, 0, 1, 1], device="cuda")
                )
                self.assertEqual(
                    km.recipe["assignment_int8_representation"],
                    representation,
                )
                self.assertEqual(
                    (km.centroids.dtype, distance.dtype, labels.dtype),
                    (torch.float32, torch.float32, torch.int64),
                )
                if dtype == torch.int8:
                    with self.assertRaisesRegex(
                        TypeError, "does not match"
                    ):
                        km.assign((base + 128).to(torch.uint8))
                    km.centroids[0, 0] = torch.nan
                    with self.assertRaisesRegex(ValueError, "must be finite"):
                        km.assign(x)
                    with self.assertRaisesRegex(
                        TypeError, "INT8 or UINT8 input"
                    ):
                        km.train(base.float())
                    self.assertIsNone(km.centroids)
                    with self.assertRaisesRegex(
                        RuntimeError, "train must be called"
                    ):
                        km.assign(x)

        split = Emeans(
            1,
            2,
            niter=1,
            batch_size=2,
            alpha0=0.5,
            assignment_precision="int8",
            assignment_backend="triton",
        )
        split.train(torch.zeros(10, 1, dtype=torch.int8, device="cuda"))
        split_rows = split.centroids.round().clamp(-128, 127).to(torch.int8)
        _, split_labels = split.assign(split_rows)
        self.assertEqual(split.n_splits, 1)
        torch.testing.assert_close(
            split_labels, torch.tensor([0, 1], device="cuda")
        )

        from faiss.contrib.e_means_int8 import Int8Quantizer

        source = base.float()
        quantizer = Int8Quantizer(1)
        quantizer.train(source)
        encoded = quantizer.compute_codes(source)
        preprocessed = Emeans(
            1,
            2,
            niter=1,
            batch_size=4,
            alpha0=0.5,
            assignment_precision="int8",
            assignment_backend="triton",
        )
        preprocessed.train(encoded, init_centroids=encoded[[0, 2]])
        torch.testing.assert_close(
            quantizer.decode(preprocessed.centroids),
            torch.tensor([[-3.0], [5.0]], device="cuda"),
        )

    @unittest.skipUnless(_INT8_TRITON_CUDA, "Triton INT8 not available")
    def test_triton_int8_integer_oracle(self):
        from faiss.contrib.e_means_triton import assign_int8_rows
        from faiss.contrib.e_means_triton import prepare_int8_centroids

        rows = torch.tensor(
            [[3] * 17, [-2] * 17], dtype=torch.int8, device="cuda"
        )
        centroids = torch.tensor(
            [[0] * 17, [4] * 17, [4] * 17],
            dtype=torch.float32,
            device="cuda",
        )
        prepared, norm = prepare_int8_centroids(centroids, "int8")
        self.assertEqual(prepared.shape, (3, 32))
        self.assertFalse(bool(prepared[:, 17:].any()))

        distance, labels = assign_int8_rows(
            rows, prepared, norm, "int8"
        )

        reference = (
            rows.int()[:, None] - prepared[:, :17].int()
        ).square().sum(2)
        expected_distance, expected_labels = reference.min(dim=1)
        torch.testing.assert_close(distance, expected_distance.float())
        torch.testing.assert_close(labels, expected_labels)

        short_storage = torch.empty(81, dtype=torch.int8, device="cuda")
        short_backed = short_storage.as_strided((3, 17), (32, 1))
        with self.assertRaisesRegex(ValueError, "shapes do not match"):
            assign_int8_rows(
                rows, short_backed, norm, "int8"
            )

        unsigned_rows = (rows.int() + 128).to(torch.uint8)
        unsigned_distance, unsigned_labels = assign_int8_rows(
            unsigned_rows, prepared, norm, "uint8"
        )
        torch.testing.assert_close(unsigned_distance, distance)
        torch.testing.assert_close(unsigned_labels, labels)

        rounded = torch.zeros(1, 257, dtype=torch.int8, device="cuda")
        rounded[0, :6] = torch.tensor(
            [-2, 0, 0, 2, 127, -128], dtype=torch.int8, device="cuda"
        )
        centroid = torch.zeros(1, 257, device="cuda")
        centroid[0, :6] = torch.tensor(
            [-1.5, -0.5, 0.5, 1.5, 200.0, -200.0], device="cuda"
        )
        for source, source_rows, source_centroid in (
            ("int8", rounded, centroid),
            ("uint8", (rounded.int() + 128).to(torch.uint8), centroid + 128),
        ):
            with self.subTest(ragged_staging=source):
                prepared, norm = prepare_int8_centroids(
                    source_centroid, source
                )
                self.assertEqual(prepared.shape, (1, 272))
                self.assertFalse(bool(prepared[:, 257:].any()))
                distance, labels = assign_int8_rows(
                    source_rows, prepared, norm, source
                )
                self.assertEqual(
                    (distance.item(), labels.item()), (0.0, 0)
                )

        with self.assertRaisesRegex(ValueError, "must be finite"):
            prepare_int8_centroids(
                torch.tensor([[float("inf")]], device="cuda"),
                "int8",
            )
        if torch.cuda.device_count() > 1:
            other = torch.device("cuda:1")
            with torch.cuda.device(0):
                prepared, norm = prepare_int8_centroids(
                    torch.zeros(2, 1, device=other),
                    "int8",
                )
            self.assertEqual((prepared.device, norm.device), (other, other))

    @unittest.skipUnless(_TRITON_CUDA, "Triton CUDA not available")
    def test_triton_full_dimension_assignment(self):
        x = torch.ones(1024, 65, device="cuda")
        centroids = torch.arange(
            3, dtype=torch.float32, device="cuda"
        )[:, None].expand(3, 65)
        km = Emeans(
            65,
            3,
            assignment_precision="bfloat16",
            assignment_backend="triton",
        )
        km.centroids = centroids

        distance, labels = km.assign(x)

        torch.testing.assert_close(distance, torch.zeros_like(distance))
        torch.testing.assert_close(labels, torch.ones_like(labels))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_cuda_smoke(self):
        generator = torch.Generator(device="cuda").manual_seed(123)
        x = torch.randn(100, 16, generator=generator, device="cuda")
        km = Emeans(
            16,
            8,
            niter=2,
            batch_size=32,
            alpha0=0.2,
            assignment_chunk_size=16,
            centroid_chunk_size=4,
        )
        rng_state = torch.cuda.get_rng_state()

        km.train(x)
        distance, labels = km.assign(x[:10])

        self.assertEqual(km.centroids.device.type, "cuda")
        self.assertEqual(distance.shape, (10,))
        self.assertEqual(labels.shape, (10,))
        self.assertTrue(torch.isfinite(km.centroids).all())
        self.assertTrue(torch.isfinite(distance).all())
        self.assertTrue(((labels >= 0) & (labels < km.k)).all())
        self.assertTrue(torch.equal(rng_state, torch.cuda.get_rng_state()))


if __name__ == "__main__":
    unittest.main()
