# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""E-means clustering with NumPy arrays or PyTorch tensors."""

import math
import time

import numpy as np

import faiss


_ALPHA_MIN_FACTOR = 0.1
_CHECK_UNUSED_EVERY = 5
_FLOAT16_MAX = 65504.0
_INIT_USAGE = 2.0**-16
_READOUT_EPSILON = 1e-5
_SPLIT_EPSILON = 1e-4
_UNUSED_THRESHOLD = 0.1

# The default profile couples fitted batch-size and initial-step laws. The
# batch laws were fitted on full passes with 100K to 10M training vectors; the
# step law also includes 100M. That law was fitted with a zero-ending cosine;
# the reference algorithm uses a 0.1 floor, which raises the mean step.
_BATCH_LAWS = {
    "throughput": (0.035134, 0.134749, 0.876473, 0.813344),
    "quality": (0.00439175, 0.134749, 0.876473, 0.813344),
}

_ALPHA_LAW = (0.8244, -0.4617, 0.9003, -0.4498, -0.1057)


def _validate_params(d, k, niter, operating_point, batch_size, alpha0, seed):
    """Validate constructor arguments."""
    if d < 1:
        raise ValueError("d must be positive")
    if k < 1:
        raise ValueError("k must be positive")
    if niter < 1:
        raise ValueError("niter must be positive")
    if operating_point not in _BATCH_LAWS:
        raise ValueError("operating_point must be 'throughput' or 'quality'")
    if (batch_size is None) != (alpha0 is None):
        raise ValueError("batch_size and alpha0 must be set together")
    if batch_size is not None:
        if int(batch_size) < 1:
            raise ValueError("batch_size must be positive")
        if not 0.0 < float(alpha0) <= 1.0:
            raise ValueError("alpha0 must be in (0, 1]")
    if seed < 0 or seed >= 2**32:
        raise ValueError("seed must be in [0, 2**32)")


def _round_power_of_two(value):
    """Round a positive value to the nearest power of two."""
    if value <= 1:
        return 1
    low = 2 ** int(math.floor(math.log2(value)))
    high = 2 * low
    return low if value / low < high / value else high


def _resolve_recipe(n, k, niter, operating_point, batch_size, alpha0):
    """Return the coupled statistical batch size and initial EMA step."""
    if batch_size is None:
        coefficient, k_exp, n_exp, iter_exp = _BATCH_LAWS[operating_point]
        raw_batch = coefficient * k**k_exp * n**n_exp * niter**iter_exp
        batch_size = _round_power_of_two(raw_batch)
        # These intentionally differ: they reproduce the two fitted recipes.
        if operating_point == "throughput":
            batch_size = min(batch_size, _round_power_of_two(n / 2))
        else:
            batch_size = min(batch_size, n // 2)
        batch_size = max(1, min(batch_size, n))

        coefficient, k_exp, batch_exp, n_exp, iter_exp = _ALPHA_LAW
        alpha0 = (
            coefficient
            * k**k_exp
            * batch_size**batch_exp
            * n**n_exp
            * niter**iter_exp
        )
        alpha0 = min(max(alpha0, 1e-4), 1.0)
    else:
        batch_size = min(int(batch_size), n)
        alpha0 = float(alpha0)

    return batch_size, alpha0


def _epoch_seed(seed, iteration):
    """Derive one reproducible permutation seed without ambient RNG state."""
    return (seed * 1000003 + iteration) % 2**32


def _scheduled_alpha(alpha0, step, total_steps):
    """Cosine-anneal an EMA step towards one tenth of its initial value."""
    fraction = min(step, total_steps) / max(1, total_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * fraction))
    return alpha0 * (_ALPHA_MIN_FACTOR + (1.0 - _ALPHA_MIN_FACTOR) * cosine)


def _is_torch_tensor(x):
    """Detect tensors; other inputs remain NumPy-compatible array-likes."""
    if isinstance(x, np.ndarray):
        return False
    try:
        import torch
    except ImportError:
        return False
    return isinstance(x, torch.Tensor)


def _as_float32_matrix(x, name, is_torch, abs_bound=None):
    if is_torch:
        import torch

        if not isinstance(x, torch.Tensor):
            raise TypeError("%s must be a torch.Tensor" % name)
        if x.layout != torch.strided:
            raise ValueError("%s must be a dense strided tensor" % name)
        if x.ndim != 2:
            raise ValueError("%s must be a 2-D tensor" % name)
        if x.shape[0] == 0 or x.shape[1] == 0:
            raise ValueError("%s must not be empty" % name)
        if x.device.type not in ("cpu", "cuda"):
            raise ValueError("%s must be on a CPU or CUDA device" % name)
        x = x.detach().to(dtype=torch.float32).contiguous()
        if abs_bound is None:
            if not bool(torch.isfinite(x).all()):
                raise ValueError("%s contains NaN or infinity" % name)
        else:
            _check_torch_abs_bound(name, abs_bound, x)
        return x

    x = np.asarray(x)
    if x.ndim != 2:
        raise ValueError("%s must be a 2-D array" % name)
    if x.shape[0] == 0 or x.shape[1] == 0:
        raise ValueError("%s must not be empty" % name)
    x = np.ascontiguousarray(x, dtype="float32")
    if not np.isfinite(x).all():
        raise ValueError("%s contains NaN or infinity" % name)
    return x


def _check_torch_abs_bound(name, abs_bound, *tensors):
    import torch

    extrema = torch.stack(
        [value for tensor in tensors for value in torch.aminmax(tensor)]
    )
    if not bool((extrema.abs() <= abs_bound).all()):
        raise ValueError(
            "%s must be finite with abs(value) <= %g" % (name, abs_bound)
        )


def _numpy_unused_threshold(usage, k):
    """Compute the threshold consistently across NumPy scalar rules."""
    total = usage.sum(dtype="float32")
    scaled = np.multiply(
        total, np.float32(_UNUSED_THRESHOLD), dtype="float32"
    )
    return np.divide(scaled, np.float32(k), dtype="float32")


def _torch_generator(device, seed):
    import torch

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator


def _randperm(n, seed, is_torch, device=None):
    if is_torch:
        import torch

        return torch.randperm(
            n,
            generator=_torch_generator(device, seed),
            device=device,
        )
    return np.random.RandomState(seed).permutation(n)


def _clone(x):
    return x.clone() if _is_torch_tensor(x) else x.copy()


def _synchronize(device):
    if device.type == "cuda":
        import torch

        torch.cuda.synchronize(device)


def _get_triton_assign_rows():
    try:
        from faiss.contrib.e_means_triton import assign_rows
    except ImportError as error:
        raise ImportError(
            "Triton assignment requires the triton package"
        ) from error
    return assign_rows


class Emeans:
    """Cluster vectors with mini-batch exponential moving averages.

    Parameters
    ----------
    d : int
        Dimension of the vectors to cluster.
    k : int
        Number of clusters.
    niter : int, optional
        Number of training epochs.
    batch_size, alpha0 : optional
        Coupled statistical batch size and initial EMA step. If omitted, both
        are selected by the fitted recipe.
    operating_point : {"throughput", "quality"}, optional
        Recipe used when ``batch_size`` and ``alpha0`` are omitted.
        ``throughput`` takes the largest fitted batch within the recipe's
        tolerance; ``quality`` takes the smaller one.
    seed : int, optional
        Seed for initialization, batch permutations, and cluster splitting.
    verbose : bool, optional
        Print one line of statistics after each pass.
    assignment_chunk_size : int, optional
        Maximum rows assigned at once. This does not alter the statistical
        batch size or the EMA schedule. The default is 65,536 rows for NumPy
        and 8,192 for PyTorch. With NumPy input, it can change floating-point
        assignment results if Faiss selects a different exact-search kernel
        for a different query size.
    centroid_chunk_size : int, optional
        Maximum centroids examined at once by native PyTorch assignment.
        Triton streams centroids internally and ignores this option.
    assignment_precision : {"float32", "float16", "bfloat16"}, optional
        Precision used to round PyTorch assignment operands. FP16 and BF16
        require an NVIDIA CUDA SM80-or-later GPU and PyTorch ``addmm`` support
        for FP32 output. Native FP32 follows PyTorch's global float32 matmul
        precision; Triton FP32 requests IEEE dot-product input precision.
        Triton supports all three precisions. Row norms, centroid norms, and
        dots use the same rounded operands. Scores, returned distances,
        statistics, and centroid state remain FP32. FP16 inputs and centroids
        must be finite with ``abs(value) <= 65504``. Reviving a centroid near
        this limit may raise if its split exceeds the range.
    assignment_backend : {"native", "triton"}, optional
        PyTorch assignment implementation selected explicitly; native is the
        default. Triton requires a compatible installation and an NVIDIA CUDA
        SM80-or-later GPU. It uses fixed choices and does not benchmark at
        runtime or fall back to native.

    Attributes
    ----------
    centroids : ndarray or Tensor
        Trained centroids with shape ``(k, d)``.
    obj : ndarray or Tensor
        Sum of squared distances observed while centroids move during each
        pass. The last value does not score the closing-mean centroids.
    iteration_stats : list of dict
        Per-pass objective, cumulative loop time, imbalance, and split count.
        Time excludes all work before the training loop, including validation,
        initialization, setup allocations, and the initial synchronization.
    recipe : dict
        Resolved recipe used by the most recent call to :meth:`train`.
    n_splits : int
        Total number of dead clusters revived during the training run.

    Notes
    -----
    NumPy input uses Faiss for exact L2 assignment. PyTorch input stays on its
    CPU or CUDA device and does not call Faiss. NumPy arrays do not import
    PyTorch.

    Every epoch includes its ragged final batch. Input conversion and
    finite-value validation scan the full input. Random permutations and
    cluster splitting use local seeded generators. Results need not agree
    across NumPy, Torch CPU, and Torch CUDA. Each epoch allocates an int64
    permutation with one entry per input row.

    The final epoch also collects per-cluster sums. Every centroid that
    receives at least one row is replaced by the mean of those rows.
    """

    def __init__(
        self,
        d,
        k,
        niter=25,
        batch_size=None,
        alpha0=None,
        operating_point="throughput",
        seed=1234,
        verbose=False,
        assignment_chunk_size=None,
        centroid_chunk_size=4096,
        assignment_precision="float32",
        assignment_backend="native",
    ):
        self.d = int(d)
        self.k = int(k)
        self.niter = int(niter)
        self.batch_size = batch_size
        self.alpha0 = alpha0
        self.operating_point = operating_point
        self.seed = int(seed)
        self.verbose = bool(verbose)
        self.assignment_chunk_size = (
            None
            if assignment_chunk_size is None
            else int(assignment_chunk_size)
        )
        self.centroid_chunk_size = int(centroid_chunk_size)
        self.assignment_precision = assignment_precision
        self.assignment_backend = assignment_backend

        _validate_params(
            self.d,
            self.k,
            self.niter,
            self.operating_point,
            self.batch_size,
            self.alpha0,
            self.seed,
        )
        if self.assignment_precision not in (
            "float32",
            "float16",
            "bfloat16",
        ):
            raise ValueError(
                "assignment_precision must be 'float32', 'float16', "
                "or 'bfloat16'"
            )
        if self.assignment_backend not in ("native", "triton"):
            raise ValueError("assignment_backend must be 'native' or 'triton'")
        # Mirrors e_means_triton._MAX_K without importing optional Triton.
        if self.assignment_backend == "triton" and self.k > 2**31 - 256:
            raise ValueError("Triton assignment requires k <= 2**31 - 256")
        if (
            self.assignment_chunk_size is not None
            and self.assignment_chunk_size < 1
        ):
            raise ValueError("assignment_chunk_size must be positive")
        if self.centroid_chunk_size < 1:
            raise ValueError("centroid_chunk_size must be positive")

        self.reset()

    def reset(self):
        """Clear the result of a previous training run."""
        self.centroids = None
        self.obj = None
        self.iteration_stats = None
        self.recipe = None
        self.n_splits = 0
        self._usage = None
        self._sums = None

    def _get_assignment_chunk_size(self, is_torch):
        if self.assignment_chunk_size is not None:
            return self.assignment_chunk_size
        return 8192 if is_torch else 65536

    def _check_assignment_support(self, x, is_torch):
        if (
            self.assignment_backend == "native"
            and self.assignment_precision == "float32"
        ):
            return
        name = (
            "Triton"
            if self.assignment_backend == "triton"
            else self.assignment_precision
        )
        if self.assignment_backend == "triton":
            _get_triton_assign_rows()
        if not is_torch or x.device.type != "cuda":
            raise ValueError(
                "%s assignment requires a PyTorch CUDA tensor" % name
            )

        import torch

        if (
            getattr(torch.version, "hip", None) is not None
            or torch.cuda.get_device_capability(x.device) < (8, 0)
        ):
            raise RuntimeError(
                "%s assignment requires an NVIDIA SM80-or-later GPU" % name
            )

    def _prepare_torch_assignment(self, centroids):
        import torch

        if self.assignment_precision != "float32":
            centroids = centroids.to(
                getattr(torch, self.assignment_precision)
            )
        return centroids, centroids.float().square().sum(dim=1)

    def _read_centroids(self):
        # This clamp is part of canonical_v1. It can shrink an untouched row
        # after its usage decays below the clamp, before revival replaces it.
        if _is_torch_tensor(self._usage):
            denominator = self._usage.clamp_min(_READOUT_EPSILON)
        else:
            denominator = np.maximum(self._usage, _READOUT_EPSILON)
        return self._sums / denominator[:, None]

    def _revive_unused(self, rng):
        # Sample donors without replacement in proportion to their usage.
        if _is_torch_tensor(self._usage):
            import torch

            threshold = self._usage.sum() * _UNUSED_THRESHOLD / self.k
            unused = self._usage < threshold
            donors = (~unused) & (self._usage > 0)
            nunused, ndonors = torch.stack(
                (unused.sum(), donors.sum())
            ).tolist()
            nsplit = min(nunused, ndonors)
            if nsplit == 0:
                return 0

            uniform = torch.rand(
                self.k,
                dtype=self._usage.dtype,
                device=self._usage.device,
                generator=rng,
            ).clamp_min(torch.finfo(self._usage.dtype).tiny)
            keys = torch.full_like(self._usage, float("-inf"))
            keys[donors] = self._usage[donors].log() - (
                -uniform[donors].log()
            ).log()
            source = torch.topk(keys, nsplit).indices
            target = unused.nonzero(as_tuple=False).flatten()[:nsplit]
            centroids = self._read_centroids()[source]
            sign = torch.ones(
                self.d, dtype=centroids.dtype, device=centroids.device
            )
        else:
            threshold = _numpy_unused_threshold(self._usage, self.k)
            unused = self._usage < threshold
            donors = (~unused) & (self._usage > 0)
            nsplit = min(int(unused.sum()), int(donors.sum()))
            if nsplit == 0:
                return 0

            uniform = np.maximum(
                rng.random_sample(self.k), np.finfo("float64").tiny
            )
            keys = np.full(self.k, -np.inf)
            keys[donors] = np.log(self._usage[donors]) - np.log(
                -np.log(uniform[donors])
            )
            source = np.argsort(keys)[-nsplit:][::-1]
            target = np.flatnonzero(unused)[:nsplit]
            centroids = self._read_centroids()[source]
            sign = np.ones(self.d, dtype="float32")

        sign[1::2] = -1.0
        donor_centroids = centroids * (1.0 - _SPLIT_EPSILON * sign)
        child_centroids = centroids * (1.0 + _SPLIT_EPSILON * sign)
        if self.assignment_precision == "float16":
            _check_torch_abs_bound(
                "revived centroids",
                _FLOAT16_MAX,
                donor_centroids,
                child_centroids,
            )
        half_usage = self._usage[source] / 2.0

        self._usage[source] = half_usage
        self._sums[source] = donor_centroids * half_usage[:, None]
        self._usage[target] = half_usage
        self._sums[target] = child_centroids * half_usage[:, None]
        self.n_splits += nsplit
        return nsplit

    def _assign_rows(self, x, centroids, centroid_norm):
        if _is_torch_tensor(x):
            if self.assignment_backend == "triton":
                return _get_triton_assign_rows()(
                    x, centroids, centroid_norm
                )

            import torch

            is_reduced = centroids.dtype != torch.float32
            if is_reduced:
                x = x.to(centroids.dtype)
            x_norm = x.float().square().sum(dim=1)
            best_score = torch.full(
                (len(x),),
                float("inf"),
                dtype=torch.float32,
                device=x.device,
            )
            labels = torch.zeros(
                len(x), dtype=torch.int64, device=x.device
            )
            addmm_kwargs = (
                {"out_dtype": torch.float32} if is_reduced else {}
            )

            for begin in range(0, self.k, self.centroid_chunk_size):
                end = min(begin + self.centroid_chunk_size, self.k)
                block = centroids[begin:end]
                block_norm = centroid_norm[begin:end]
                score = torch.addmm(
                    block_norm[None, :],
                    x,
                    block.t(),
                    alpha=-2.0,
                    **addmm_kwargs,
                )
                block_score, block_label = score.min(dim=1)
                better = block_score < best_score
                best_score = torch.where(better, block_score, best_score)
                labels = torch.where(better, block_label + begin, labels)

            return (best_score + x_norm).clamp_min(0), labels

        distance, labels = faiss.knn(x, centroids, 1)
        return distance.ravel(), labels.ravel()

    def train(self, x, init_centroids=None):
        """Train on a dense matrix and return the final pass objective.

        Parameters
        ----------
        x : ndarray or Tensor
            Training vectors with shape ``(n, d)``.
        init_centroids : ndarray or Tensor, optional
            Initial centroids with shape ``(k, d)`` and the same array type as
            ``x``. Torch centroids are moved to the device of ``x``.

        Returns
        -------
        float
            Sum of assignment distances observed during the final pass.
        """
        is_torch = _is_torch_tensor(x)
        if is_torch:
            import torch

            np_or_torch = torch
        else:
            np_or_torch = np

        precision = self.assignment_precision
        abs_bound = _FLOAT16_MAX if precision == "float16" else None
        x = _as_float32_matrix(x, "x", is_torch, abs_bound=abs_bound)
        self._check_assignment_support(x, is_torch)
        device = x.device if is_torch else None
        assignment_chunk_size = self._get_assignment_chunk_size(is_torch)
        n, d = x.shape
        if d != self.d:
            raise ValueError("x has dimension %d, expected %d" % (d, self.d))
        if n < self.k:
            raise ValueError("x must contain at least k rows")

        centroids = None
        if init_centroids is not None:
            if _is_torch_tensor(init_centroids) != is_torch:
                raise TypeError(
                    "x and init_centroids must use the same array type"
                )
            centroids = _as_float32_matrix(
                init_centroids,
                "init_centroids",
                is_torch,
                abs_bound=abs_bound,
            )
            if centroids.shape != (self.k, self.d):
                raise ValueError(
                    "init_centroids has shape %r, expected (%d, %d)"
                    % (tuple(centroids.shape), self.k, self.d)
                )
            if is_torch:
                centroids = centroids.to(device)
            centroids = _clone(centroids)

        self.reset()
        batch_size, alpha0 = _resolve_recipe(
            n,
            self.k,
            self.niter,
            self.operating_point,
            self.batch_size,
            self.alpha0,
        )
        steps_per_pass = (n + batch_size - 1) // batch_size
        total_steps = self.niter * steps_per_pass
        requested_batch_size = (
            None if self.batch_size is None else int(self.batch_size)
        )
        manual = requested_batch_size is not None
        self.recipe = {
            "name": "manual" if manual else "canonical_v1",
            "operating_point": None if manual else self.operating_point,
            "batch_size": batch_size,
            "batch_size_requested": requested_batch_size,
            "alpha0": alpha0,
            "niter": self.niter,
            "steps_per_pass": steps_per_pass,
            "total_steps": total_steps,
            "tail_policy": "keep",
            "alpha_min_factor": _ALPHA_MIN_FACTOR,
            "check_unused_every": _CHECK_UNUSED_EVERY,
            "unused_threshold": _UNUSED_THRESHOLD,
            "split_epsilon": _SPLIT_EPSILON,
            "init_usage": _INIT_USAGE,
            "readout_epsilon": _READOUT_EPSILON,
            "seed": self.seed,
            "backend": "torch" if is_torch else "faiss",
            "assignment_backend": self.assignment_backend,
            "assignment_precision": self.assignment_precision,
            "assignment_chunk_size": assignment_chunk_size,
        }
        if is_torch:
            self.recipe.update(
                device=str(device),
                centroid_chunk_size=self.centroid_chunk_size,
            )

        if centroids is None:
            if n == self.k:
                centroids = _clone(x)
            else:
                perm = _randperm(n, self.seed, is_torch, device)[: self.k]
                centroids = _clone(x[perm])

        self._usage = np_or_torch.full_like(centroids[:, 0], _INIT_USAGE)
        if is_torch:
            split_rng = _torch_generator(device, self.seed)
        else:
            split_rng = np.random.RandomState(self.seed)
        self._sums = centroids * self._usage[:, None]
        stats = []
        step = 0

        # Batch statistics stay float32 to match the canonical EMA state.
        if is_torch:
            batch_stats = self._sums.new_empty((self.k, self.d + 1))
            batch_sums = batch_stats[:, :-1]
            batch_counts = batch_stats[:, -1]
            count_column = x.new_ones((assignment_chunk_size, 1))
            _synchronize(device)
        else:
            batch_counts = np.empty_like(self._usage)
            batch_sums = np.empty_like(self._sums)
        start_time = time.perf_counter()

        for iteration in range(self.niter):
            order = _randperm(
                n, _epoch_seed(self.seed, iteration), is_torch, device
            )
            pass_counts = np_or_torch.zeros_like(
                self._usage, dtype=np_or_torch.float64
            )
            objective = 0.0
            if iteration == self.niter - 1:
                pass_sums = np_or_torch.zeros_like(self._sums)
            else:
                pass_sums = None
            splits_before = self.n_splits

            for batch_begin in range(0, n, batch_size):
                batch_index = order[batch_begin : batch_begin + batch_size]
                if is_torch:
                    batch_stats.zero_()
                else:
                    batch_counts[:] = 0
                    batch_sums[:] = 0
                centroids = self._read_centroids()
                if is_torch:
                    centroids, centroid_norm = (
                        self._prepare_torch_assignment(centroids)
                    )
                else:
                    centroid_norm = None

                for chunk_begin in range(
                    0, len(batch_index), assignment_chunk_size
                ):
                    index = batch_index[
                        chunk_begin : chunk_begin + assignment_chunk_size
                    ]
                    rows = x[index]
                    distance, labels = self._assign_rows(
                        rows, centroids, centroid_norm
                    )
                    objective += distance.sum(dtype=np_or_torch.float64)
                    if is_torch:
                        values = torch.cat(
                            (rows, count_column[: len(rows)]), dim=1
                        )
                        batch_stats.index_add_(0, labels, values)
                    else:
                        batch_counts += np.bincount(
                            labels, minlength=self.k
                        )
                        np.add.at(batch_sums, labels, rows)

                # Release the k*d readout before revival constructs another.
                del centroids, centroid_norm
                pass_counts += batch_counts
                if pass_sums is not None:
                    pass_sums += batch_sums
                alpha = _scheduled_alpha(alpha0, step, total_steps)
                if alpha == 1.0:
                    # Preserve centroids absent from an alpha-one batch.
                    touched = batch_counts > 0
                    if is_torch:
                        self._usage.copy_(
                            torch.where(touched, batch_counts, self._usage)
                        )
                        self._sums.copy_(
                            torch.where(
                                touched[:, None], batch_sums, self._sums
                            )
                        )
                    else:
                        self._usage[touched] = batch_counts[touched]
                        self._sums[touched] = batch_sums[touched]
                else:
                    decay = 1.0 - alpha
                    self._usage *= decay
                    self._usage += alpha * batch_counts
                    self._sums *= decay
                    self._sums += alpha * batch_sums
                if (step + 1) % _CHECK_UNUSED_EVERY == 0:
                    self._revive_unused(split_rng)
                step += 1

            if is_torch:
                _synchronize(device)
            objective_value = objective.item()
            imbalance = (
                self.k
                * np_or_torch.dot(pass_counts, pass_counts).item()
                / (n * n)
            )
            stat = {
                "obj": objective_value,
                "time": time.perf_counter() - start_time,
                "imbalance_factor": imbalance,
                "nsplit": self.n_splits - splits_before,
            }
            stats.append(stat)
            if self.verbose:
                print(
                    "e-means iteration %d: objective=%.6g imbalance=%.3f "
                    "nsplit=%d"
                    % (
                        iteration + 1,
                        objective_value,
                        imbalance,
                        stat["nsplit"],
                    )
                )

        centroids = self._read_centroids()
        nonempty = pass_counts > 0
        objective_values = [stat["obj"] for stat in stats]
        # Canonical capture accumulates float32 sums and divides in float32.
        if is_torch:
            closing_mean = pass_sums / pass_counts.clamp_min(1).float()[:, None]
            centroids = torch.where(nonempty[:, None], closing_mean, centroids)
            self.centroids = centroids.contiguous()
            self.obj = torch.tensor(
                objective_values, dtype=torch.float64, device=device
            )
        else:
            centroids[nonempty] = (
                pass_sums[nonempty]
                / pass_counts[nonempty, None].astype("float32")
            )
            self.centroids = np.ascontiguousarray(centroids, dtype="float32")
            self.obj = np.asarray(objective_values, dtype="float64")
        self.iteration_stats = stats
        self._usage = None
        self._sums = None
        return objective_value

    def assign(self, x):
        """Return squared L2 distances and centroid indices.

        See ``assignment_precision`` for operand rounding."""
        if self.centroids is None:
            raise RuntimeError("train must be called before assign")
        is_torch = _is_torch_tensor(self.centroids)
        if _is_torch_tensor(x) != is_torch:
            raise TypeError("x and centroids must use the same array type")
        if is_torch:
            import torch

        precision = self.assignment_precision
        abs_bound = _FLOAT16_MAX if precision == "float16" else None
        x = _as_float32_matrix(x, "x", is_torch, abs_bound=abs_bound)
        self._check_assignment_support(x, is_torch)
        centroids = self.centroids.detach() if is_torch else self.centroids
        assignment_chunk_size = self._get_assignment_chunk_size(is_torch)
        if x.shape[1] != self.d:
            raise ValueError(
                "x has dimension %d, expected %d" % (x.shape[1], self.d)
            )
        if is_torch and x.device != centroids.device:
            raise ValueError("x and centroids must be on the same device")
        if abs_bound is not None:
            _check_torch_abs_bound("centroids", abs_bound, centroids)

        if is_torch:
            distance = torch.empty(
                len(x), dtype=torch.float32, device=x.device
            )
            labels = torch.empty(
                len(x), dtype=torch.int64, device=x.device
            )
            centroids, centroid_norm = self._prepare_torch_assignment(
                centroids
            )
        else:
            distance = np.empty(len(x), dtype="float32")
            labels = np.empty(len(x), dtype="int64")
            centroid_norm = None

        for begin in range(0, len(x), assignment_chunk_size):
            end = min(begin + assignment_chunk_size, len(x))
            distance[begin:end], labels[begin:end] = self._assign_rows(
                x[begin:end], centroids, centroid_norm
            )
        return distance, labels
