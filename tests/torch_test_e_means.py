# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
import tempfile

import numpy as np
import torch

from faiss.contrib.e_means import Emeans, DistributedEmeans
from faiss.contrib.e_means.e_means import _epoch_seed


def _cpu_gloo_replica(rank, rendezvous):
    import torch.distributed as dist

    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    try:
        distributed = DistributedEmeans(1, 2, group=dist.group.WORLD)
        with unittest.TestCase().assertRaises(ValueError):
            distributed.begin(10, torch.zeros((2, 1)))
        with unittest.TestCase().assertRaises(RuntimeError):
            distributed.accumulate(torch.zeros((1, 1)))
        with unittest.TestCase().assertRaises(NotImplementedError):
            distributed.train(torch.zeros((10, 1)))
        initial = np.array([[0.], [100.]], dtype="float32")
        if rank:
            initial.fill(0)
        dist.broadcast(torch.from_numpy(initial), src=0)
        rows = (np.arange(10) % 3).astype("float32")[:, None]
        args = dict(niter=2, batch_size=4, alpha0=0.5, seed=7)
        model, reference = Emeans(1, 2, **args), Emeans(1, 2, **args)
        model.begin(len(rows), initial)
        reference.begin(len(rows), initial)
        for _ in range(2):
            for begin in range(0, len(rows), 4):
                n = min(4, len(rows) - begin)
                local_begin = begin + rank * (n // 2)
                stats = model.accumulate(rows[local_begin:local_begin + n // 2])
                # Communication is caller-owned; local update consumes the
                # combined global statistics and global prescribed row count.
                combined = {name: np.array(stats[name], copy=True, order="C")
                            for name in ("sums", "counts", "objective")}
                for value in combined.values():
                    dist.all_reduce(torch.from_numpy(value))
                model.update(dict(combined, n=n))
                reference.update(reference.accumulate(rows[begin:begin + n]))
            model.end_epoch()
            reference.end_epoch()
        np.testing.assert_allclose(model.finalize(), reference.finalize(), rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(model.obj, reference.obj, rtol=1e-6, atol=1e-6)
        assert model.n_splits == reference.n_splits and model.n_splits > 0
    finally:
        dist.destroy_process_group()


class TestTorchEmeans(unittest.TestCase):
    @unittest.skipUnless(torch.distributed.is_available() and
                         torch.distributed.is_gloo_available(), "Gloo unavailable")
    def test_cpu_gloo_caller_owned_statistics(self):
        with tempfile.TemporaryDirectory() as directory:
            torch.multiprocessing.start_processes(
                _cpu_gloo_replica, args=("file://" + directory + "/group",),
                nprocs=2, join=True, start_method="fork"
            )

    def test_local_update_has_no_group_argument(self):
        model = Emeans(1, 1, niter=1, batch_size=2, alpha0=0.5)
        model.begin(4, torch.zeros((1, 1)))
        stats = model.accumulate(torch.zeros((2, 1)))
        with self.assertRaises(TypeError):
            model.update(stats, group=object())
        model.update(stats)
        stats = model.accumulate(torch.ones((2, 1)))
        with self.assertRaises(TypeError):
            model.update(stats, group=object())
        model.update(stats)
        model.end_epoch()
        torch.testing.assert_close(model.finalize(), torch.tensor([[0.5]]))

    def test_caller_batches_match_matrix_training(self):
        for device in ("cpu", "cuda") if torch.cuda.is_available() else ("cpu",):
            with self.subTest(device=device):
                # Binary values keep this exact fixture's FP32 sums independent
                # of CUDA atomic addition order.
                x = torch.arange(51, device=device, dtype=torch.float32)
                x = x.reshape(17, 3) / 8
                init = x[[0, 5, 10, 16]].clone()
                args = dict(niter=3, batch_size=4, alpha0=0.2, seed=7,
                            assignment_chunk_size=2, centroid_chunk_size=2)
                matrix, caller = Emeans(3, 4, **args), Emeans(3, 4, **args)
                matrix.train(x, init_centroids=init)
                caller.begin(len(x), init)
                for epoch in range(caller.niter):
                    generator = torch.Generator(device=device).manual_seed(
                        _epoch_seed(caller.seed, epoch)
                    )
                    order = torch.randperm(
                        len(x), generator=generator, device=device
                    )
                    for begin in range(0, len(x), 4):
                        stats = caller.accumulate(x[order[begin:begin + 4]])
                        caller.update(stats)
                    caller.end_epoch()
                torch.testing.assert_close(
                    caller.finalize(), matrix.centroids, rtol=0, atol=0
                )
                torch.testing.assert_close(caller.obj, matrix.obj, rtol=0, atol=0)
                self.assertEqual(caller.n_splits, matrix.n_splits)

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
