# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import numpy as np

from faiss.contrib.e_means import (
    Emeans,
    _epoch_seed,
    _numpy_unused_threshold,
    _resolve_recipe,
    _scheduled_alpha,
)


class TestEmeansRecipe(unittest.TestCase):
    def test_reference_values(self):
        cases = [
            (1_000_000, 16_384, 25, "throughput", 262_144, 1.0),
            (
                1_000_000,
                16_384,
                25,
                "quality",
                32_768,
                0.1545388281304627,
            ),
            (100_000_000, 262_144, 25, "throughput", 33_554_432, 1.0),
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

    def test_empty_cluster_is_preserved_at_alpha_one(self):
        x = np.arange(4, dtype="float32")[:, None]
        init = np.array([[0.0], [100.0]], dtype="float32")
        km = Emeans(1, 2, niter=1, batch_size=4, alpha0=1.0)

        km.train(x, init_centroids=init)

        np.testing.assert_array_equal(km.centroids, [[1.5], [100.0]])

    def test_readout_clamp_is_part_of_canonical_profile(self):
        x = np.arange(4, dtype="float32")[:, None]
        init = np.array([[0.0], [100.0]], dtype="float32")
        km = Emeans(1, 2, niter=2, batch_size=4, alpha0=1.0)

        km.train(x, init_centroids=init)

        self.assertAlmostEqual(float(km.centroids[1, 0]), 68.66455, places=4)

    def test_revival_starts_on_fifth_update(self):
        x = np.zeros((10, 1), dtype="float32")
        init = np.array([[0.0], [10.0]], dtype="float32")

        before = Emeans(1, 2, niter=4, batch_size=10, alpha0=0.5)
        before.train(x, init_centroids=init)
        self.assertEqual(before.n_splits, 0)

        on_fifth = Emeans(1, 2, niter=5, batch_size=10, alpha0=0.5)
        on_fifth.train(x, init_centroids=init)
        self.assertEqual(on_fifth.n_splits, 1)

    def test_revival_precedes_current_batch_update(self):
        seed = 7
        order = np.random.RandomState(_epoch_seed(seed, 0)).permutation(5)
        x = np.zeros((5, 1), dtype="float32")
        x[order[-1]] = 10.0
        init = np.array([[0.0], [10.0]], dtype="float32")
        km = Emeans(
            1, 2, niter=1, batch_size=1, alpha0=0.5, seed=seed
        )

        km.train(x, init_centroids=init)

        self.assertEqual(km.n_splits, 1)

    def test_revival_threshold_and_split_state(self):
        km = Emeans(1, 3)
        km._usage = np.array(
            [27.715767, 0.38280505, 803.37445], dtype="float32"
        )
        initial = np.array([[10.0], [20.0], [30.0]], dtype="float32")
        km._sums = initial * km._usage[:, None]
        expected_usage = km._usage.sum() - km._usage[1]
        expected_sum = km._sums.sum(axis=0) - km._sums[1]

        nsplit = km._revive_unused(np.random.RandomState(123))

        self.assertEqual(nsplit, 1)
        np.testing.assert_allclose(km._read_centroids()[0], [10.0])
        self.assertAlmostEqual(
            float(km._read_centroids()[1, 0]), 30.003, places=5
        )
        self.assertEqual(km._usage.sum(), expected_usage)
        np.testing.assert_allclose(km._sums.sum(axis=0), expected_sum)

    def test_revival_threshold_is_float32(self):
        usage = np.array([69938690.0], dtype="float32")

        threshold = _numpy_unused_threshold(usage, 52359)

        self.assertEqual(threshold.dtype, np.dtype("float32"))
        self.assertEqual(threshold.view("uint32"), 1124438855)

    def test_tail_batch_is_used(self):
        x = np.array([[0.0], [0.0], [0.0], [0.0], [10.0]], dtype="float32")
        km = Emeans(1, 1, niter=1, batch_size=3, alpha0=0.5)

        km.train(x, init_centroids=np.array([[0.0]], dtype="float32"))

        np.testing.assert_array_equal(km.centroids, [[2.0]])
        self.assertEqual(km.recipe["steps_per_pass"], 2)
        self.assertEqual(km.recipe["total_steps"], 2)
        self.assertEqual(km.recipe["tail_policy"], "keep")

    def test_each_pass_uses_tail(self):
        class RecordingEmeans(Emeans):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.assignment_sizes = []

            def _assign_rows(self, x, centroids, centroid_norm=None):
                self.assignment_sizes.append(len(x))
                return super()._assign_rows(x, centroids, centroid_norm)

        x = np.arange(5, dtype="float32")[:, None]
        km = RecordingEmeans(
            1, 1, niter=2, batch_size=3, alpha0=0.5
        )

        km.train(x, init_centroids=np.array([[0.0]], dtype="float32"))

        self.assertEqual(km.assignment_sizes, [3, 2, 3, 2])

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

    def test_training_is_deterministic(self):
        x = np.random.RandomState(123).randn(100, 8).astype("float32")
        args = {"niter": 3, "batch_size": 17, "alpha0": 0.2, "seed": 9}
        a = Emeans(8, 7, **args)
        b = Emeans(8, 7, **args)

        a.train(x)
        b.train(x)

        np.testing.assert_array_equal(a.centroids, b.centroids)
        np.testing.assert_array_equal(a.obj, b.obj)

    def test_retrain_and_reset_release_training_state(self):
        x = np.random.RandomState(123).randn(40, 4).astype("float32")
        km = Emeans(4, 3, niter=2, batch_size=11, alpha0=0.2)
        km.train(x)
        centroids = km.centroids.copy()

        km.train(x)

        np.testing.assert_array_equal(km.centroids, centroids)
        self.assertIsNone(km._usage)
        self.assertIsNone(km._sums)
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

    def test_assignment_breaks_ties_by_index(self):
        km = Emeans(2, 5)
        km.centroids = np.ones((5, 2), dtype="float32")

        _, labels = km.assign(np.zeros((1, 2), dtype="float32"))

        self.assertEqual(labels.item(), 0)

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
