# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import numpy as np

from faiss.contrib.e_means import Emeans
from faiss.contrib.e_means.e_means import (
    _epoch_seed,
    _resolve_recipe,
    _scheduled_alpha,
)


class TestEmeansRecipe(unittest.TestCase):
    def test_reference_values(self):
        cases = [
            (1_000_000, 16_384, 25, "throughput", 262_144, 1.0),
            (
                100_000_000,
                262_144,
                25,
                "quality",
                4_194_304,
                0.42718555053293517,
            ),
        ]
        for case in cases:
            n, k, niter, operating_point, expected_batch, expected_alpha = case
            with self.subTest(n=n, k=k, operating_point=operating_point):
                batch, alpha = _resolve_recipe(
                    n, k, niter, operating_point, None, None
                )
                self.assertEqual(batch, expected_batch)
                self.assertAlmostEqual(alpha, expected_alpha)

    def test_invalid_overrides(self):
        with self.assertRaises(ValueError):
            Emeans(4, 3, batch_size=16)
        with self.assertRaises(ValueError):
            Emeans(4, 3, alpha0=0.2)
        with self.assertRaises(ValueError):
            Emeans(4, 3, batch_size=0, alpha0=0.2)
        with self.assertRaises(ValueError):
            Emeans(4, 3, batch_size=16, alpha0=0.0)

    def test_derived_recipe_metadata(self):
        x = np.arange(124, dtype="float32").reshape(31, 4)
        km = Emeans(4, 3, niter=3)

        km.train(x)

        self.assertEqual(km.recipe["name"], "canonical_v1")
        batch_size = km.recipe["batch_size"]
        steps_per_pass = (len(x) + batch_size - 1) // batch_size
        self.assertEqual(km.recipe["steps_per_pass"], steps_per_pass)
        self.assertEqual(km.recipe["total_steps"], 3 * steps_per_pass)
        self.assertEqual(km.recipe["tail_policy"], "keep")
        self.assertEqual(km.recipe["assignment_chunk_size"], 65536)

    def test_schedule_uses_logical_steps(self):
        alpha0 = 0.4
        self.assertEqual(_scheduled_alpha(alpha0, 0, 10), alpha0)
        self.assertAlmostEqual(_scheduled_alpha(alpha0, 10, 10), 0.04)
        last = _scheduled_alpha(alpha0, 9, 10)
        self.assertGreater(last, 0.04)
        self.assertAlmostEqual(last, 0.04880982706687237)


class TestEmeans(unittest.TestCase):
    def test_closing_mean_and_assign(self):
        x = np.array([[0.0], [2.0], [10.0], [12.0]], dtype="float32")
        init = np.array([[0.0], [10.0]], dtype="float32")
        km = Emeans(1, 2, niter=1, batch_size=4, alpha0=0.5)

        objective = km.train(x, init_centroids=init)

        np.testing.assert_array_equal(km.centroids, [[1.0], [11.0]])
        self.assertEqual(objective, 8.0)
        self.assertEqual(km.recipe["name"], "manual")
        self.assertIsNone(km.recipe["operating_point"])
        distance, labels = km.assign(x)
        np.testing.assert_array_equal(labels, [0, 0, 1, 1])
        np.testing.assert_array_equal(distance, [1.0, 1.0, 1.0, 1.0])

    def test_untouched_centroid_retains_its_position(self):
        x = np.arange(4, dtype="float32")[:, None]
        init = np.array([[0.0], [100.0]], dtype="float32")
        first = Emeans(1, 2, niter=1, batch_size=4, alpha0=1.0)
        first.train(x, init_centroids=init)
        np.testing.assert_array_equal(first.centroids, [[1.5], [100.0]])

        second = Emeans(1, 2, niter=2, batch_size=4, alpha0=1.0)
        second.train(x, init_centroids=init)
        # Explicit C/u replaces the old readout floor's contraction to 68.66455.
        np.testing.assert_array_equal(second.centroids, [[1.5], [100.0]])

    def test_caller_batches_match_matrix_training(self):
        x = np.random.RandomState(123).randn(17, 3).astype("float32")
        init = x[:4].copy()
        args = dict(niter=3, batch_size=4, alpha0=0.2, seed=7,
                    assignment_chunk_size=2)
        matrix, caller = Emeans(3, 4, **args), Emeans(3, 4, **args)
        matrix.train(x, init_centroids=init)
        caller.begin(len(x), init)
        for epoch in range(caller.niter):
            order = np.random.RandomState(_epoch_seed(caller.seed, epoch)).permutation(len(x))
            for begin in range(0, len(x), caller.recipe["batch_size"]):
                caller.update(caller.accumulate(x[order[begin:begin + 4]]))
            caller.end_epoch()
        result = caller.finalize()
        self.assertIs(result, caller.centroids)
        np.testing.assert_array_equal(result, matrix.centroids)
        np.testing.assert_array_equal(caller.obj, matrix.obj)
        self.assertEqual(caller.n_splits, matrix.n_splits)

    def test_capture_assignments_match_closing_means(self):
        x = np.random.RandomState(5).randn(203, 3).astype("float32")
        args = dict(niter=3, seed=1, assignment_chunk_size=16)
        km = Emeans(3, 7, capture_assignments=True, **args)
        km.train(x)
        labels = km.assignments
        self.assertEqual((labels.dtype, labels.shape), (np.int64, (len(x),)))
        self.assertTrue(((labels >= 0) & (labels < 7)).all())
        # Each closing centroid is the mean of the rows the final pass gave it.
        for j in np.unique(labels):
            np.testing.assert_allclose(
                km.centroids[j], x[labels == j].mean(axis=0), rtol=1e-5, atol=1e-6
            )
        plain = Emeans(3, 7, **args)
        plain.train(x)
        self.assertIsNone(plain.assignments)
        np.testing.assert_array_equal(plain.centroids, km.centroids)
        np.testing.assert_array_equal(plain.obj, km.obj)

    def test_caller_batches_return_final_pass_labels(self):
        x = np.random.RandomState(123).randn(17, 3).astype("float32")
        init = x[:4].copy()
        args = dict(niter=3, batch_size=4, alpha0=0.2, seed=7,
                    assignment_chunk_size=2, capture_assignments=True)
        matrix, caller = Emeans(3, 4, **args), Emeans(3, 4, **args)
        matrix.train(x, init_centroids=init)
        caller.begin(len(x), init)
        labels = np.full(len(x), -1)
        for epoch in range(caller.niter):
            order = np.random.RandomState(_epoch_seed(caller.seed, epoch)).permutation(len(x))
            for begin in range(0, len(x), caller.recipe["batch_size"]):
                rows = order[begin:begin + 4]
                stats = caller.accumulate(x[rows])
                self.assertEqual("labels" in stats, epoch == caller.niter - 1)
                if "labels" in stats:
                    labels[rows] = stats["labels"]
                caller.update(stats)
            caller.end_epoch()
        np.testing.assert_array_equal(caller.finalize(), matrix.centroids)
        self.assertIsNone(caller.assignments)
        np.testing.assert_array_equal(labels, matrix.assignments)

    def test_capture_memory_is_one_int64_per_row(self):
        # Even when one batch holds every row, capture adds one int64 per row:
        # train() writes the final pass's labels straight to their rows. And an
        # epoch boundary holds one permutation, not two.
        import tracemalloc

        def peak(**args):
            tracemalloc.start()
            base = tracemalloc.get_traced_memory()[0]
            Emeans(d, k, alpha0=0.5, seed=3, **args).train(x)
            used = tracemalloc.get_traced_memory()[1] - base
            tracemalloc.stop()
            return used

        n, d, k = 400_000, 2, 16
        x = np.random.RandomState(0).randn(n, d).astype("float32")
        Emeans(d, k, niter=1, batch_size=1000, alpha0=0.5).train(x[:1000])  # warm caches
        off = peak(niter=1, batch_size=n)
        on = peak(niter=1, batch_size=n, capture_assignments=True)
        self.assertTrue(6.0 < (on - off) / n < 10.0, (on - off) / n)
        self.assertLess((peak(niter=2, batch_size=n) - off) / n, 4.0)

    def test_lifecycle_rejects_incomplete_batches(self):
        km = Emeans(1, 1, niter=1, batch_size=3, alpha0=0.5)
        km.begin(5, np.zeros((1, 1), dtype="float32"))
        with self.assertRaises(RuntimeError):
            km.finalize()
        empty = km.accumulate(np.empty((0, 1), dtype="float32"))
        with self.assertRaises(ValueError):
            km.update(empty)
        km.reset()
        km.begin(5, np.zeros((1, 1), dtype="float32"))
        stats = km.accumulate(np.zeros((3, 1), dtype="float32"))
        with self.assertRaises(RuntimeError):
            km.accumulate(np.zeros((3, 1), dtype="float32"))
        km.update(stats)
        with self.assertRaises(RuntimeError):
            km.end_epoch()
        km.update(km.accumulate(np.ones((2, 1), dtype="float32")))
        km.end_epoch()
        np.testing.assert_array_equal(
            km.finalize(), np.array([[0.4]], dtype="float32")
        )

    def test_revival_starts_on_fifth_update(self):
        x = np.zeros((10, 1), dtype="float32")
        init = np.array([[0.0], [10.0]], dtype="float32")

        before = Emeans(1, 2, niter=4, batch_size=10, alpha0=0.5)
        before.train(x, init_centroids=init)
        self.assertEqual(before.n_splits, 0)

        on_fifth = Emeans(1, 2, niter=5, batch_size=10, alpha0=0.5)
        on_fifth.train(x, init_centroids=init)
        self.assertEqual(on_fifth.n_splits, 1)

    def test_revival_follows_current_batch_update(self):
        seed = 7
        order = np.random.RandomState(_epoch_seed(seed, 0)).permutation(5)
        x = np.zeros((5, 1), dtype="float32")
        x[order[-1]] = 10.0
        init = np.array([[0.0], [10.0]], dtype="float32")
        km = Emeans(
            1, 2, niter=1, batch_size=1, alpha0=0.5, seed=seed
        )

        km.train(x, init_centroids=init)

        # The fifth batch makes cluster 1 live before revival checks usage.
        self.assertEqual(km.n_splits, 0)

    def test_each_pass_uses_tail(self):
        x = np.array([[0.0], [0.0], [0.0], [0.0], [10.0]], dtype="float32")
        km = Emeans(1, 1, niter=2, batch_size=3, alpha0=0.5)

        km.train(x, init_centroids=np.array([[0.0]], dtype="float32"))

        np.testing.assert_array_equal(km.centroids, [[2.0]])
        self.assertEqual(km.recipe["steps_per_pass"], 2)
        self.assertEqual(km.recipe["total_steps"], 4)
        self.assertEqual(km.recipe["tail_policy"], "keep")

    def test_assignment_chunking_preserves_logical_updates(self):
        rng = np.random.RandomState(123)
        x = rng.randn(41, 5).astype("float32")
        init = x[:4].copy()
        args = {"niter": 3, "batch_size": 13, "alpha0": 0.2, "seed": 7}
        a = Emeans(5, 4, assignment_chunk_size=4, **args)
        b = Emeans(5, 4, assignment_chunk_size=64, **args)

        a.train(x, init_centroids=init)
        b.train(x, init_centroids=init)

        # This fixture remains away from assignment boundaries. In general,
        # Faiss may choose different exact-search kernels for different query
        # sizes, so chunking is not promised to be bitwise invariant.
        np.testing.assert_allclose(a.centroids, b.centroids)
        np.testing.assert_allclose(a.obj, b.obj)
        self.assertEqual(a.recipe["steps_per_pass"], 4)
        self.assertEqual(a.recipe["total_steps"], 12)

    def test_retrain_is_deterministic_and_reset(self):
        x = np.random.RandomState(123).randn(40, 4).astype("float32")
        km = Emeans(4, 3, niter=2, batch_size=11, alpha0=0.2)
        km.train(x)
        centroids = km.centroids.copy()
        objective = km.obj.copy()

        km.train(x)

        np.testing.assert_array_equal(km.centroids, centroids)
        np.testing.assert_array_equal(km.obj, objective)
        km.reset()
        with self.assertRaises(RuntimeError):
            km.assign(x)

    def test_training_does_not_advance_global_rng(self):
        x = np.arange(80, dtype="float32").reshape(20, 4)
        state = np.random.get_state()
        km = Emeans(4, 3, niter=1, batch_size=7, alpha0=0.2)

        km.train(x)

        after = np.random.get_state()
        self.assertEqual(state[0], after[0])
        np.testing.assert_array_equal(state[1], after[1])
        self.assertEqual(state[2:], after[2:])

    def test_invalid_input(self):
        km = Emeans(3, 2, niter=1, batch_size=2, alpha0=0.5)
        with self.assertRaises(ValueError):
            km.train(np.zeros((4, 2), dtype="float32"))
        with self.assertRaises(ValueError):
            km.train(np.full((4, 3), np.nan, dtype="float32"))
        with self.assertRaises(RuntimeError):
            km.assign(np.zeros((1, 3), dtype="float32"))


if __name__ == "__main__":
    unittest.main()
