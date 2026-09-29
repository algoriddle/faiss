# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import numpy as np
import torch

from faiss.contrib.e_means import Emeans


class TestTorchEmeans(unittest.TestCase):
    def test_closing_mean_and_assign(self):
        x = torch.tensor([[0.0], [2.0], [10.0], [12.0]])
        init = torch.tensor([[0.0], [10.0]])
        km = Emeans(1, 2, niter=1, batch_size=4, alpha0=0.5)

        objective = km.train(x, init_centroids=init)

        torch.testing.assert_close(km.centroids, torch.tensor([[1.0], [11.0]]))
        self.assertEqual(objective, 8.0)
        self.assertEqual(km.recipe["assignment_chunk_size"], 8192)
        distance, labels = km.assign(x)
        torch.testing.assert_close(distance, torch.ones(4))
        torch.testing.assert_close(labels, torch.tensor([0, 0, 1, 1]))

    def test_blocked_assignment_matches_cdist(self):
        generator = torch.Generator().manual_seed(123)
        x = torch.randn(23, 7, generator=generator)
        centroids = torch.randn(11, 7, generator=generator)
        km = Emeans(7, 11, centroid_chunk_size=3)

        distance, labels = km._assign_rows(x, centroids)
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

    def test_empty_cluster_is_preserved_at_alpha_one(self):
        x = torch.arange(4, dtype=torch.float32)[:, None]
        init = torch.tensor([[0.0], [100.0]])
        km = Emeans(1, 2, niter=1, batch_size=4, alpha0=1.0)

        km.train(x, init_centroids=init)

        torch.testing.assert_close(
            km.centroids, torch.tensor([[1.5], [100.0]])
        )

    def test_revival_occurs_during_training(self):
        x = torch.zeros((10, 1))
        init = torch.tensor([[0.0], [10.0]])
        km = Emeans(1, 2, niter=5, batch_size=10, alpha0=0.5)

        km.train(x, init_centroids=init)

        self.assertEqual(km.n_splits, 1)

    def test_revival_threshold_matches_canonical_association(self):
        km = Emeans(1, 3)
        km._usage = torch.tensor([27.715767, 0.38280505, 803.37445])
        initial = torch.tensor([[10.0], [20.0], [30.0]])
        km._sums = initial * km._usage[:, None]
        expected_usage = km._usage.sum() - km._usage[1]
        expected_sum = km._sums.sum(dim=0) - km._sums[1]

        nsplit = km._revive_unused(torch.Generator().manual_seed(123))

        self.assertEqual(nsplit, 1)
        torch.testing.assert_close(
            km._read_centroids()[0], torch.tensor([10.0])
        )
        self.assertAlmostEqual(
            float(km._read_centroids()[1, 0]), 30.003, places=5
        )
        torch.testing.assert_close(km._usage.sum(), expected_usage)
        torch.testing.assert_close(km._sums.sum(dim=0), expected_sum)

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

    def test_training_detaches_input(self):
        x = torch.randn(20, 4, requires_grad=True)
        init = x[:3].detach().clone().requires_grad_()
        km = Emeans(4, 3, niter=1, batch_size=7, alpha0=0.2)

        km.train(x, init_centroids=init)
        self.assertFalse(km.centroids.requires_grad)
        km.centroids.requires_grad_()
        distance, _ = km.assign(x)

        self.assertFalse(distance.requires_grad)

    def test_input_types_match(self):
        x = torch.randn(20, 4)
        init = x[:3].clone()
        km = Emeans(4, 3, niter=1, batch_size=7, alpha0=0.2)

        with self.assertRaises(TypeError):
            km.train(x.numpy(), init_centroids=init)
        with self.assertRaises(TypeError):
            km.train(x, init_centroids=init.numpy())

    def test_numpy_and_torch_agree(self):
        rng = np.random.RandomState(123)
        centers = (10 * rng.randn(4, 6)).astype("float32")
        labels = np.arange(80) % 4
        x = centers[labels] + 0.1 * rng.randn(80, 6).astype("float32")
        args = {
            "niter": 3,
            "batch_size": len(x),
            "alpha0": 0.2,
            "seed": 7,
        }
        numpy_km = Emeans(6, 4, **args)
        torch_km = Emeans(6, 4, **args)

        numpy_km.train(x, init_centroids=centers)
        torch_km.train(
            torch.from_numpy(x), init_centroids=torch.from_numpy(centers)
        )

        np.testing.assert_allclose(
            numpy_km.centroids,
            torch_km.centroids.numpy(),
            atol=2e-5,
            rtol=1e-6,
        )
        np.testing.assert_allclose(
            numpy_km.obj, torch_km.obj.numpy(), atol=3e-3, rtol=1e-3
        )
        self.assertEqual(
            numpy_km.iteration_stats[0].keys(),
            torch_km.iteration_stats[0].keys(),
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
