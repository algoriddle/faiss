# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
import tempfile
import importlib.util

import numpy as np
import torch

from faiss.contrib.e_means import Emeans, DistributedEmeans
from faiss.contrib.e_means.e_means import _epoch_seed
from faiss.contrib.e_means.e_means_quantizer import CastQuantizer
from faiss.contrib.e_means.e_means_fp8 import Fp8Quantizer
from faiss.contrib.e_means.e_means_int8 import Int8Quantizer


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

    def test_explicit_int8_quantizer_and_scope(self):
        x = torch.tensor([[10., 100.], [12., 104.], [14., 108.], [16., 112.]])
        q = Int8Quantizer(2)
        with self.assertRaises(RuntimeError):
            q.compute_codes(x)
        q.train(x)
        self.assertEqual((q.code_size, q.scale), (2, 16.))
        self.assertIs(q.to("cpu"), q)
        expected = torch.tensor([[-48, -96], [-16, -32], [16, 32], [48, 96]], dtype=torch.int8)
        torch.testing.assert_close(q.compute_codes(x), expected)
        torch.testing.assert_close(q.decode(expected), x)
        torch.testing.assert_close(q.compute_codes(torch.tensor([[1e6, -1e6]])),
                                   torch.tensor([[127, -128]], dtype=torch.int8))
        q.validate(x)
        with self.assertRaises(ValueError):
            q.validate(torch.full((1, 2), float("nan")))
        self.assertEqual(q.compute_codes(x[:0]).shape, (0, 2))
        with self.assertRaisesRegex(ValueError, "requires Triton"):
            Emeans(2, 2, quantizer=q)
        with self.assertRaisesRegex(TypeError, "no JIT INT8"):
            Emeans(2, 2, quantizer=q, assignment_backend="triton").train(x)
        with self.assertRaisesRegex(ValueError, "d <= 43919"):
            Emeans(43920, 2, quantizer=Int8Quantizer(43920), assignment_backend="triton")

    @unittest.skipUnless(
        torch.cuda.is_available() and getattr(torch.version, "hip", None) is None
        and torch.cuda.get_device_capability() in ((8, 0), (9, 0), (10, 0))
        and importlib.util.find_spec("triton") is not None, "qualified INT8 CUDA required"
    )
    def test_native_int8_uint8_and_decoded_training(self):
        base = torch.tensor([[-4.], [-2.], [4.], [6.]], device="cuda")
        q = Int8Quantizer(1)
        q.train(base)
        for dtype, offset, quantizer in ((torch.int8, 0, None),
                                         (torch.uint8, 128, None), (torch.int8, 0, q)):
            with self.subTest(dtype=dtype, calibrated=quantizer is not None):
                codes = (base + offset).to(dtype) if quantizer is None else q.compute_codes(base)
                initial = base[[0, 2]] + offset
                km = Emeans(1, 2, niter=1, batch_size=4, alpha0=.5, quantizer=quantizer,
                            assignment_backend="triton", assignment_chunk_size=1)
                objective = km.train(codes, initial)
                self.assertEqual(objective, 8.)
                torch.testing.assert_close(km.centroids,
                    torch.tensor([[-3. + offset], [5. + offset]], device="cuda"))
                distance, labels = km.assign(codes)
                torch.testing.assert_close(distance, torch.ones(4, device="cuda"))
                torch.testing.assert_close(labels, torch.tensor([0, 0, 1, 1], device="cuda"))
                self.assertEqual((km.centroids.dtype, distance.dtype, labels.dtype),
                                 (torch.float32, torch.float32, torch.int64))
                self.assertEqual(km.recipe["representation"], str(dtype).removeprefix("torch."))
        with self.assertRaisesRegex(TypeError, "no JIT INT8"):
            Emeans(1, 2, quantizer=q, assignment_backend="triton").train(base)

    def test_explicit_fp8_calibration_and_scope(self):
        q = Fp8Quantizer(1)
        with self.assertRaises(RuntimeError):
            q.compute_codes(torch.zeros((1, 1)))
        sample = torch.tensor([[0.], [2.], [10.], [12.]])
        q.train(sample)
        self.assertTrue(q.is_trained)
        self.assertEqual(q.code_size, 1)
        self.assertEqual((q.scale, q.decision), (8., "small_sample_centered"))
        self.assertIs(q.to("cpu"), q)
        torch.testing.assert_close(q.decode(q.compute_codes(sample)), sample)
        self.assertEqual(q.compute_codes(sample[:0]).shape, (0, 1))
        with self.assertRaises(ValueError):
            q.validate(torch.tensor([[float("nan")]]))
        with self.assertRaisesRegex(ValueError, "FP8 requires Triton"):
            Emeans(1, 2, quantizer=q)
        with self.assertRaises(ValueError):
            Emeans(1, 2, quantizer=q, assignment_backend="triton").train(sample)

    @unittest.skipUnless(
        torch.cuda.is_available() and getattr(torch.version, "hip", None) is None
        and torch.cuda.get_device_capability() in ((9, 0), (10, 0), (10, 3))
        and importlib.util.find_spec("triton") is not None, "qualified FP8 CUDA required"
    )
    def test_fp8_native_jit_decoded_lifecycle(self):
        x = torch.tensor([[0.1], [2.2], [10.3], [12.4]], device="cuda")
        q = Fp8Quantizer(1)
        q.train(x)
        codes = q.compute_codes(x)
        represented = q.decode(codes)
        self.assertFalse(torch.equal(x, represented))
        init = torch.tensor([[0.], [10.]], device="cuda")
        args = dict(niter=1, batch_size=4, alpha0=0.5, quantizer=q,
                    assignment_backend="triton", assignment_chunk_size=2)
        jit, native, steps = (Emeans(1, 2, **args) for _ in range(3))
        jit.train(x, init)
        native.train(codes, init)
        steps.begin(len(x), init)
        stats = steps.accumulate(codes)
        steps.update(stats)
        steps.end_epoch()
        steps.finalize()
        expected = represented.reshape(2, 2, 1).mean(dim=1)
        for model in (jit, native, steps):
            torch.testing.assert_close(model.centroids, expected)
            torch.testing.assert_close(model.obj, jit.obj, atol=0, rtol=0)
            distance, labels = model.assign(codes)
            operands = q.decode(q.compute_codes(expected))
            oracle = (represented[:, None] - operands[None, :]).square().sum(2)
            torch.testing.assert_close(distance, oracle.min(1).values, atol=2e-5, rtol=2e-5)
            torch.testing.assert_close(labels, oracle.argmin(1))
        self.assertEqual((jit.centroids.dtype, distance.dtype, labels.dtype),
                         (torch.float32, torch.float32, torch.int64))
        raw = Emeans(1, 2, niter=1, batch_size=4, alpha0=0.5,
                     assignment_backend="triton")
        raw.train(codes, codes[[0, 2]].float())
        torch.testing.assert_close(raw.centroids, codes.float().reshape(2, 2, 1).mean(1))

    @unittest.skipUnless(
        torch.cuda.is_available() and getattr(torch.version, "hip", None) is None
        and torch.cuda.get_device_capability() in ((9, 0), (10, 0), (10, 3))
        and importlib.util.find_spec("triton") is not None, "qualified FP8 CUDA required"
    )
    def test_fp8_long_positive_selected_distances(self):
        d = 4096
        base = (torch.arange(d, device='cuda') % 251).float()
        x = torch.stack((base, base + 32, base + 40))
        q = Fp8Quantizer(d)
        q.train(torch.stack((torch.full_like(base, -128), torch.full_like(base, 128))))
        codes = q.compute_codes(x)
        for k in (2, 1024):
            initial = (base + 96).repeat(k, 1)
            initial[:2] = x[:2]
            operands = q.decode(q.compute_codes(initial)).double()
            oracle = (q.decode(codes).double()[:, None] - operands[None]).square().sum(2)
            for rows in (codes, x):
                with self.subTest(k=k, native=rows is codes):
                    model = Emeans(d, k, quantizer=q, assignment_backend='triton')
                    model.centroids = initial.clone()
                    distance, labels = model.assign(rows)
                    torch.testing.assert_close(labels, oracle.argmin(1))
                    chosen = oracle.gather(1, labels[:, None]).flatten()
                    torch.testing.assert_close(distance.double(), chosen, rtol=2e-6, atol=1e-3)
                    self.assertEqual(distance[:2].tolist(), [0., 0.])

    def test_backend_selection_and_cpu_rejection(self):
        self.assertEqual(Emeans(1, 1).assignment_backend, "native")
        with self.assertRaises(ValueError):
            Emeans(1, 1, assignment_backend="unknown")
        with self.assertRaisesRegex(ValueError, "Triton assignment"):
            Emeans(1, 1, assignment_backend="triton").train(torch.zeros((2, 1)))

    @unittest.skipUnless(
        torch.cuda.is_available() and getattr(torch.version, "hip", None) is None
        and torch.cuda.get_device_capability() >= (8, 0)
        and importlib.util.find_spec("triton") is not None, "Triton CUDA required"
    )
    def test_triton_matches_represented_training(self):
        initial = torch.tensor([[0.], [3.]], device="cuda")
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                x = torch.tensor([[0.125], [0.25], [3.125], [3.25], [3.5]],
                                 dtype=dtype, device="cuda")
                args = dict(niter=2, batch_size=2, alpha0=0.5, seed=7)
                native = Emeans(1, 2, **args)
                tiled = Emeans(1, 2, assignment_backend="triton", **args)
                native.train(x, initial)
                tiled.train(x, initial)
                torch.testing.assert_close(tiled.centroids, native.centroids, atol=2e-6, rtol=2e-6)
                torch.testing.assert_close(tiled.obj, native.obj, atol=1e-5, rtol=2e-6)
                torch.testing.assert_close(tiled.assign(x)[1], native.assign(x)[1])
                self.assertEqual(tiled.recipe["assignment_backend"], "triton")

    def test_stateless_cast_and_cpu_scope(self):
        q = CastQuantizer(1, torch.bfloat16)
        q.train()
        self.assertTrue(q.is_trained)
        self.assertEqual(q.code_size, 2)
        self.assertIs(q.to("cpu"), q)
        x = torch.tensor([[1.001], [3.001]])
        codes = q.compute_codes(x)
        torch.testing.assert_close(q.decode(codes), x.bfloat16().float())
        q.validate(x)
        with self.assertRaises(ValueError):
            q.validate(torch.tensor([[float("inf")]]))
        with self.assertRaises(ValueError):
            Emeans(1, 1, quantizer=q).train(x)
        model = Emeans(1, 1, niter=1, batch_size=2, alpha0=0.5)
        model.train(codes)
        torch.testing.assert_close(model.centroids, codes.float().mean(0, keepdim=True))

    @unittest.skipUnless(
        torch.cuda.is_available() and getattr(torch.version, "hip", None) is None
        and torch.cuda.get_device_capability() >= (8, 0), "NVIDIA SM80 required"
    )
    def test_native_and_cast_represented_means(self):
        x = torch.tensor([[1.001], [1.002], [3.001], [3.002]], device="cuda")
        initial = torch.tensor([[1.0], [3.0]], device="cuda")
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                q = CastQuantizer(1, dtype)
                codes = q.compute_codes(x)
                native = Emeans(1, 2, niter=1, batch_size=4, alpha0=0.5)
                cast = Emeans(1, 2, niter=1, batch_size=4, alpha0=0.5, quantizer=q)
                native.train(codes, initial)
                cast.train(x, initial)
                expected = codes.float().reshape(2, 2, 1).mean(dim=1)
                torch.testing.assert_close(native.centroids, expected, atol=0, rtol=0)
                torch.testing.assert_close(cast.centroids, expected, atol=0, rtol=0)
                torch.testing.assert_close(native.obj, cast.obj, atol=0, rtol=0)
                dn, ln = native.assign(codes)
                dc, lc = cast.assign(x)
                torch.testing.assert_close(dn, dc, atol=0, rtol=0)
                torch.testing.assert_close(ln, lc)
                self.assertEqual((native.centroids.dtype, dn.dtype),
                                 (torch.float32, torch.float32))

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

    def test_capture_assignments_match_closing_means(self):
        cases = [("cpu", "native", None)]
        if torch.cuda.is_available():
            cases.append(("cuda", "native", None))
            if (getattr(torch.version, "hip", None) is None
                    and torch.cuda.get_device_capability() >= (8, 0)
                    and importlib.util.find_spec("triton") is not None):
                cases += [("cuda", "triton", None), ("cuda", "triton", torch.bfloat16),
                          ("cuda", "triton", torch.int8)]
        for device, backend, represented in cases:
            with self.subTest(device=device, backend=backend, represented=represented):
                generator = torch.Generator(device=device).manual_seed(3)
                x = torch.randn(301, 8, generator=generator, device=device)
                quantizer, rows = None, x
                if represented is torch.bfloat16:
                    quantizer = CastQuantizer(8, torch.bfloat16)
                elif represented is torch.int8:
                    quantizer = Int8Quantizer(8)
                    quantizer.train(x)
                    quantizer.to(device)
                    rows = quantizer.compute_codes(x)
                args = dict(niter=2, seed=1, assignment_backend=backend,
                            quantizer=quantizer, assignment_chunk_size=64)
                km = Emeans(8, 9, capture_assignments=True, **args)
                km.train(rows)
                labels = km.assignments
                self.assertEqual((labels.dtype, labels.device.type, labels.shape),
                                 (torch.int64, device, (301,)))
                self.assertTrue(((labels >= 0) & (labels < 9)).all())
                # Closing means are built from the represented rows the final pass labelled.
                values = (quantizer.decode(quantizer.compute_codes(x))
                          if quantizer is not None else x)
                for j in labels.unique().tolist():
                    torch.testing.assert_close(km.centroids[j], values[labels == j].mean(0),
                                               rtol=1e-5, atol=1e-5)
                plain = Emeans(8, 9, **args)
                plain.train(rows)
                self.assertIsNone(plain.assignments)
                if device == "cpu":
                    torch.testing.assert_close(plain.centroids, km.centroids, rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_capture_assignments_memory_lifetimes(self):
        # The n-long label buffer must be the only per-row cost of capture: each
        # epoch's permutation is dropped before the next is drawn, and a final-pass
        # batch's labels are freed before the next batch allocates its own.
        import faiss.contrib.e_means.e_means as module

        n, d, k = 1 << 20, 4, 64
        x = torch.randn(n, d, generator=torch.Generator(device="cuda").manual_seed(0),
                        device="cuda")
        draws, batches = [], []
        randperm, accumulate = module._randperm, module.Emeans._accumulate

        def spy_randperm(rows, seed, is_torch, device=None):
            if is_torch and rows == n:
                torch.cuda.synchronize()
                draws.append(torch.cuda.memory_allocated())
            return randperm(rows, seed, is_torch, device)

        def spy_accumulate(model, rows, index=None, **kwargs):
            if model._pass_sums is not None:
                torch.cuda.synchronize()
                batches.append(torch.cuda.memory_allocated())
            return accumulate(model, rows, index, **kwargs)

        module._randperm, module.Emeans._accumulate = spy_randperm, spy_accumulate
        try:
            Emeans(d, k, niter=3, seed=1, batch_size=n // 4, alpha0=0.5,
                   capture_assignments=True).train(x)
        finally:
            module._randperm, module.Emeans._accumulate = randperm, accumulate
        epochs = draws[1:]  # draws[0] picks the initial centroids
        self.assertEqual((len(epochs), len(batches)), (3, 4))
        self.assertLess(max(epochs) - epochs[0], 4 * n)  # one int64 permutation = 8n bytes
        self.assertLess(max(batches) - batches[0], n)  # one batch of int64 labels = 2n bytes

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
