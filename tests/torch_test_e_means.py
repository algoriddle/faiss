# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

from faiss.contrib.e_means import Emeans


_REDUCED_PRECISION_CUDA = (
    torch.cuda.is_available()
    and getattr(torch.version, "hip", None) is None
    and torch.cuda.get_device_capability() >= (8, 0)
)


class TestTorchEmeans(unittest.TestCase):
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
