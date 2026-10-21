# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
import warnings

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


def _unit(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def _three_directions(seed=0, per=40, d=6):
    """Rows scattered around three well-separated unit directions."""
    rng = np.random.RandomState(seed)
    axes = np.eye(d, dtype="float32")[:3]
    rows = np.concatenate([a + 0.15 * rng.randn(per, d) for a in axes]).astype("float32")
    return _unit(rows).astype("float32"), axes


class TestSphericalEmeans(unittest.TestCase):
    def test_one_step_is_dhillon_modha(self):
        x, axes = _three_directions()
        init = _unit(axes + 0.3).astype("float32")
        km = Emeans(6, 3, niter=1, batch_size=len(x), alpha0=1.0, spherical=True)
        km.train(x, init_centroids=init)
        # Dhillon and Modha (2001), steps 2-3: assign by highest cosine, then
        # each concept vector is the normalized sum of its cluster's rows.
        labels = np.argmax(x @ _unit(init).T, axis=1)
        expected = _unit(np.stack([x[labels == j].sum(0) for j in range(3)]))
        np.testing.assert_allclose(km.centroids, expected, rtol=0, atol=1e-6)

    def test_lloyd_steps_match_faiss_spherical(self):
        import faiss
        x, axes = _three_directions(seed=1)
        init = _unit(axes + 0.3).astype("float32")
        for niter in (1, 3):
            with self.subTest(niter=niter):
                ours = Emeans(6, 3, niter=niter, batch_size=len(x), alpha0=1.0,
                              spherical=True)
                ours.train(x, init_centroids=init)
                reference = faiss.Kmeans(6, 3, niter=niter, spherical=True)
                reference.train(x, init_centroids=init)
                # A full-batch pass at alpha0=1 is a spherical Lloyd step; at three
                # passes both settle on the same concept vectors of these directions.
                np.testing.assert_allclose(ours.centroids, reference.centroids,
                                           rtol=0, atol=1e-5)

    def test_rows_are_normalized_and_reported_once(self):
        x, _ = _three_directions(seed=2)
        scaled = x * np.random.RandomState(3).uniform(0.2, 9.0, (len(x), 1)).astype("float32")
        init = x[[0, 40, 80]].copy()
        args = dict(niter=2, batch_size=16, alpha0=0.3, seed=5, spherical=True)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            unit = Emeans(6, 3, **args)
            unit.train(x, init_centroids=init)
            self.assertEqual(len(caught), 0)
            raw = Emeans(6, 3, **args)
            raw.train(scaled, init_centroids=init * 7.0)
        notices = [w for w in caught if "not unit-norm" in str(w.message)]
        self.assertEqual(len(notices), 1)
        np.testing.assert_allclose(raw.centroids, unit.centroids, rtol=0, atol=1e-6)
        np.testing.assert_array_equal(raw.assign(scaled)[1], unit.assign(x)[1])

    def test_centroids_stay_on_the_sphere(self):
        x = _unit(np.random.RandomState(4).randn(500, 8)).astype("float32")
        km = Emeans(8, 20, niter=3, spherical=True)
        km.train(x)
        np.testing.assert_allclose(np.linalg.norm(km.centroids, axis=1), 1.0,
                                   rtol=0, atol=1e-6)
        self.assertTrue(km.recipe["spherical"])
        self.assertEqual(km.recipe["assign_returns"], "cosine_similarity")

    def test_assign_returns_cosine_similarity(self):
        x = _unit(np.random.RandomState(6).randn(300, 5)).astype("float32")
        km = Emeans(5, 7, niter=2, spherical=True)
        km.train(x)
        similarity, labels = km.assign(x * 3.0)
        cosines = x @ km.centroids.T
        np.testing.assert_array_equal(labels, np.argmax(cosines, axis=1))
        np.testing.assert_allclose(similarity, cosines.max(axis=1), rtol=0, atol=1e-6)
        self.assertEqual(km.assign_returns, "cosine_similarity")

    def test_scaling_spans_match_one_chunk(self):
        # 70,000 rows in 8,192-row chunks are scaled in two spans (65,536 + 4,464).
        rng = np.random.RandomState(7)
        x = (rng.randn(70000, 4) * rng.uniform(0.5, 3.0, (70000, 1))).astype("float32")
        init = x[:5].copy()
        out = []
        for chunk in (8192, 70000):
            km = Emeans(4, 5, niter=1, batch_size=len(x), alpha0=1.0, spherical=True,
                        assignment_chunk_size=chunk)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                km.train(x, init_centroids=init)
            out.append((km.centroids, km.assign(x)))
        np.testing.assert_allclose(out[0][0], out[1][0], rtol=0, atol=1e-6)
        np.testing.assert_array_equal(out[0][1][1], out[1][1][1])
        np.testing.assert_allclose(out[0][1][0], out[1][1][0], rtol=0, atol=1e-6)

    def test_default_is_euclidean(self):
        km = Emeans(4, 3)
        self.assertFalse(km.spherical)
        self.assertEqual(km.assign_returns, "squared_l2_distance")


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
