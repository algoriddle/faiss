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
_INIT_USAGE = 2.0**-16
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


def _as_float32_matrix(x, name, is_torch, allow_empty=False):
    if is_torch:
        import torch

        if not isinstance(x, torch.Tensor):
            raise TypeError("%s must be a torch.Tensor" % name)
        if x.layout != torch.strided:
            raise ValueError("%s must be a dense strided tensor" % name)
        if x.ndim != 2:
            raise ValueError("%s must be a 2-D tensor" % name)
        if (x.shape[0] == 0 and not allow_empty) or x.shape[1] == 0:
            raise ValueError("%s must not be empty" % name)
        if x.device.type not in ("cpu", "cuda"):
            raise ValueError("%s must be on a CPU or CUDA device" % name)
        if x.dtype != torch.float32:
            raise ValueError("%s must have dtype torch.float32" % name)
        x = x.detach().contiguous()
        if x.device.type == "cpu" and not bool(torch.isfinite(x).all()):
            raise ValueError("%s contains NaN or infinity" % name)
        return x

    x = np.asarray(x)
    if x.ndim != 2:
        raise ValueError("%s must be a 2-D array" % name)
    if (x.shape[0] == 0 and not allow_empty) or x.shape[1] == 0:
        raise ValueError("%s must not be empty" % name)
    x = np.ascontiguousarray(x, dtype="float32")
    if not np.isfinite(x).all():
        raise ValueError("%s contains NaN or infinity" % name)
    return x


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


def _separate_splits(donor, child):
    """Move one coordinate only when the FP32 split operands coincide.

    Zero donors become distinct subnormal operands; their squared distances
    can still tie. No value scan or dynamic collapsed-row list is needed.
    """
    if _is_torch_tensor(child):
        import torch

        collapsed = (donor == child).all(dim=1)
        value = child[:, 0]
        neighbor = torch.nextafter(value, torch.full_like(value, float("inf")))
        inward = torch.nextafter(value, torch.full_like(value, float("-inf")))
        neighbor = torch.where((neighbor.abs() < float("inf")), neighbor, inward)
        child[:, 0] = torch.where(collapsed, neighbor, value)
    else:
        collapsed = (donor == child).all(axis=1)
        value = child[:, 0]
        with np.errstate(over="ignore"):
            neighbor = np.nextafter(value, np.float32(np.inf))
        neighbor = np.where(
            np.isfinite(neighbor), neighbor,
            np.nextafter(value, np.float32(-np.inf)),
        )
        child[:, 0] = np.where(collapsed, neighbor, value)


def _synchronize(device):
    if device.type == "cuda":
        import torch

        torch.cuda.synchronize(device)


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
        batch size or the EMA schedule. Defaults are 65,536 rows for NumPy,
        8,192 for PyTorch CPU, and dimension-dependent 8,192 to 1,048,576
        for CUDA. With NumPy input, it can change floating-point
        assignment results if Faiss selects a different exact-search kernel
        for a different query size.
        CUDA chunks can also change floating assignment results through
        GEMM or tiling selection.
    centroid_chunk_size : int, optional
        Maximum centroids examined at once by the PyTorch assignment path.

    Attributes
    ----------
    centroids : ndarray or Tensor
        Trained centroids with shape ``(k, d)``.
    obj : ndarray or Tensor
        Sum of squared distances observed while centroids move during each
        pass. The last value does not score the closing-mean centroids.
    iteration_stats : list of dict
        Per-pass objective, cumulative loop time, imbalance, and split count.
        Time starts after initialization, setup allocations, and the initial
        synchronization. Caller-driven time also includes caller loading between
        begin and end_epoch; it is not a compute-only measurement.
    recipe : dict
        Resolved recipe used by the most recent call to :meth:`begin`.
    n_splits : int
        Total number of dead clusters revived during the training run.

    Notes
    -----
    NumPy input uses Faiss for exact L2 assignment. PyTorch input stays on its
    CPU or CUDA device and does not call Faiss. NumPy arrays do not import
    PyTorch.

    Every epoch includes its ragged final batch. CPU input conversion and
    finite-value validation scan the input. CUDA accepts native FP32 and requires
    caller-validated finite values; metadata checks do not scan CUDA values.
    Random permutations and
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

        _validate_params(
            self.d,
            self.k,
            self.niter,
            self.operating_point,
            self.batch_size,
            self.alpha0,
            self.seed,
        )
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
        self._centroids = None
        self._pending_stats = None
        self._batch_stats = self._stats_operand = None
        self._pass_counts = self._pass_sums = self._objective = None
        self._split_rng = None

    def _get_assignment_chunk_size(self, is_torch, cuda=False):
        if self.assignment_chunk_size is not None:
            return self.assignment_chunk_size
        if cuda:
            # Fewer, larger chunks amortize per-chunk host work and fill the
            # device. Target roughly 256 MiB per FP32 statistics matrix or
            # native Torch score tile.
            width = max(self.d, min(self.k, self.centroid_chunk_size))
            return min(2**20, max(8192, 2**26 // width))
        return 8192 if is_torch else 65536

    def _revive_unused(self, rng):
        source, target = self._choose_splits(rng)
        return 0 if source is None else self._apply_splits(source, target)

    def _choose_splits(self, rng):
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
                return None, None

            uniform = torch.rand(
                self.k,
                dtype=self._usage.dtype,
                device=self._usage.device,
                generator=rng,
            ).clamp_min(torch.finfo(self._usage.dtype).tiny)
            if self._usage.is_cuda:
                # Fixed shapes avoid compaction's host reads.
                keys = torch.where(
                    donors, self._usage.log() - (-uniform.log()).log(), float("-inf")
                )
                source = torch.topk(keys, nsplit).indices
                target = torch.sort(unused.byte(), descending=True, stable=True).indices[:nsplit]
            else:
                keys = torch.full_like(self._usage, float("-inf"))
                keys[donors] = self._usage[donors].log() - (
                    -uniform[donors].log()
                ).log()
                source = torch.topk(keys, nsplit).indices
                target = unused.nonzero(as_tuple=False).flatten()[:nsplit]
        else:
            threshold = _numpy_unused_threshold(self._usage, self.k)
            unused = self._usage < threshold
            donors = (~unused) & (self._usage > 0)
            nsplit = min(int(unused.sum()), int(donors.sum()))
            if nsplit == 0:
                return None, None

            uniform = np.maximum(
                rng.random_sample(self.k), np.finfo("float64").tiny
            )
            keys = np.full(self.k, -np.inf)
            keys[donors] = np.log(self._usage[donors]) - np.log(
                -np.log(uniform[donors])
            )
            source = np.argsort(keys)[-nsplit:][::-1]
            target = np.flatnonzero(unused)[:nsplit]

        return source, target

    def _apply_splits(self, source, target):
        centroids = self._centroids[source]
        if _is_torch_tensor(centroids):
            import torch

            sign = torch.ones(self.d, dtype=centroids.dtype, device=centroids.device)
        else:
            sign = np.ones(self.d, dtype="float32")
        sign[1::2] = -1.0
        offset = _SPLIT_EPSILON * sign
        donor_centroids = centroids * (1.0 - offset)
        child_centroids = centroids * (1.0 + offset)
        _separate_splits(donor_centroids, child_centroids)
        half_usage = self._usage[source] / 2.0

        self._usage[source] = half_usage
        self._centroids[source] = donor_centroids
        self._usage[target] = half_usage
        self._centroids[target] = child_centroids
        self.n_splits += len(source)
        return len(source)

    def _assign_rows(self, x, centroids, centroid_norm=None):
        if _is_torch_tensor(x):
            import torch

            x_norm = (x * x).sum(dim=1)
            best_score = torch.full(
                (len(x),), float("inf"), dtype=x.dtype, device=x.device
            )
            labels = torch.zeros(
                len(x), dtype=torch.int64, device=x.device
            )

            for begin in range(0, self.k, self.centroid_chunk_size):
                end = min(begin + self.centroid_chunk_size, self.k)
                block = centroids[begin:end]
                block_norm = (
                    (block * block).sum(dim=1)
                    if centroid_norm is None
                    else centroid_norm[begin:end]
                )
                score = torch.addmm(
                    block_norm[None, :],
                    x,
                    block.t(),
                    alpha=-2.0,
                )
                block_score, block_label = score.min(dim=1)
                better = block_score < best_score
                torch.where(better, block_score, best_score, out=best_score)
                torch.where(better, block_label + begin, labels, out=labels)

            return (best_score + x_norm).clamp_min(0), labels

        distance, labels = faiss.knn(x, centroids, 1)
        return distance.ravel(), labels.ravel()

    def begin(self, global_n, init_centroids):
        """Initialize a run with caller-owned batches and explicit FP32 state.

        ``init_centroids`` has shape (k, d); its array type and device select
        execution. It is copied. Each epoch must supply the resolved batch
        sizes, including the final remainder, then call :meth:`end_epoch`.
        Loading, ordering, transfers and prefetch are the caller's responsibility.
        """
        n = int(global_n)
        if n != global_n or n < self.k:
            raise ValueError("global_n must be an integer at least k")
        is_torch = _is_torch_tensor(init_centroids)
        centroids = _as_float32_matrix(init_centroids, "init_centroids", is_torch)
        if centroids.shape != (self.k, self.d):
            raise ValueError("init_centroids must have shape (k, d)")
        self.reset()
        self._centroids = _clone(centroids)
        self._is_torch = is_torch
        if is_torch:
            import torch

            self._usage = centroids.new_full((self.k,), _INIT_USAGE)
            self._batch_stats = centroids.new_empty((self.k, self.d + 1))
            self._split_rng = _torch_generator(centroids.device, self.seed)
        else:
            self._usage = np.full(self.k, _INIT_USAGE, dtype="float32")
            self._batch_stats = np.empty((self.k, self.d + 1), dtype="float32")
            self._split_rng = np.random.RandomState(self.seed)
        batch_size, alpha0 = _resolve_recipe(
            n, self.k, self.niter, self.operating_point,
            self.batch_size, self.alpha0,
        )
        steps_per_pass = (n + batch_size - 1) // batch_size
        manual = self.batch_size is not None
        self.recipe = {
            "name": "manual" if manual else "canonical_v1",
            "operating_point": None if manual else self.operating_point,
            "batch_size": batch_size,
            "batch_size_requested": int(self.batch_size) if manual else None,
            "alpha0": alpha0,
            "niter": self.niter,
            "steps_per_pass": steps_per_pass,
            "total_steps": self.niter * steps_per_pass,
            "tail_policy": "keep",
            "alpha_min_factor": _ALPHA_MIN_FACTOR,
            "check_unused_every": _CHECK_UNUSED_EVERY,
            "unused_threshold": _UNUSED_THRESHOLD,
            "split_epsilon": _SPLIT_EPSILON,
            "init_usage": _INIT_USAGE,
            "seed": self.seed,
            "backend": "torch" if is_torch else "faiss",
            "assignment_chunk_size": self._get_assignment_chunk_size(
                is_torch, is_torch and centroids.is_cuda),
        }
        if is_torch:
            self.recipe.update(
                device=str(centroids.device),
                centroid_chunk_size=self.centroid_chunk_size,
            )
            self._stats_operand = None
        self._n = n
        self._step = self._epoch = 0
        self.iteration_stats = []
        self._start_epoch()
        if is_torch:
            _synchronize(centroids.device)
        self._start_time = time.perf_counter()

    def _require_running(self):
        if self._centroids is None or self._epoch >= self.niter:
            raise RuntimeError("begin a run before accumulating or updating")

    def _start_epoch(self):
        self._epoch_steps = self._epoch_rows = 0
        self._splits_before = self.n_splits
        if self._is_torch:
            import torch

            if self._pass_counts is None:
                self._pass_counts = self._usage.new_zeros(self.k, dtype=torch.float64)
                self._objective = self._usage.new_zeros((), dtype=torch.float64)
            else:
                self._pass_counts.zero_()
                self._objective.zero_()
            self._pass_sums = (
                self._centroids.new_zeros((self.k, self.d))
                if self._epoch == self.niter - 1 else None
            )
        else:
            self._pass_counts = np.zeros(self.k, dtype="float64")
            self._objective = np.float64(0)
            self._pass_sums = (
                np.zeros_like(self._centroids)
                if self._epoch == self.niter - 1 else None
            )

    def accumulate(self, x):
        """Assign a full local batch without advancing centroids or schedule.

        Return a mapping with FP32 ``sums`` (k,d), FP32 ``counts`` (k),
        scalar FP64 ``objective`` and host integer ``n``. Sums/counts are
        model-owned workspace: consume them with update before accumulating
        again. Copy them if they must outlive that update. CPU callers may
        combine matching frozen-snapshot statistics themselves, summing all
        four fields, then call the same local update.

        A typed (0,d) batch yields zeros. CUDA FP32 values must be finite;
        value validation is the caller's responsibility and does not run here.
        """
        self._require_running()
        if _is_torch_tensor(x) != self._is_torch:
            raise TypeError("x and init_centroids must use the same array type")
        x = _as_float32_matrix(x, "x", self._is_torch, allow_empty=True)
        if x.shape[1] != self.d:
            raise ValueError("x must have dimension d")
        if self._is_torch and x.device != self._centroids.device:
            raise ValueError("x and centroids must be on the same device")
        return self._accumulate(x)

    def _accumulate(self, x, index=None):
        # Indexed matrix training gathers only assignment chunks, never B*d.
        self._require_running()
        if self._pending_stats is not None:
            raise RuntimeError(
                "consume accumulated statistics before accumulating again"
            )
        if self._epoch_steps == self.recipe["steps_per_pass"]:
            raise RuntimeError("call end_epoch before the next batch")
        n = len(x) if index is None else len(index)
        expected_n = min(self.recipe["batch_size"], self._n - self._epoch_rows)
        if n > expected_n:
            raise ValueError("batch exceeds the prescribed logical batch size")
        stats = self._accumulate_core(x, index)
        self._pending_stats = stats
        return stats

    def _accumulate_core(self, x, index=None):
        # Tensor work shared by checked accumulation and native replay.
        n = len(x) if index is None else len(index)
        if self._is_torch:
            import torch

            self._batch_stats.zero_()
            objective = self._objective.new_zeros(())
            centroid_norm = (self._centroids * self._centroids).sum(dim=1)
            # Repeated FP32 additions can stall above 2**24 rows.
            accumulation = (self._batch_stats.new_zeros(
                self._batch_stats.shape, dtype=torch.float64)
                if n > 2**24 else self._batch_stats)
            operand = None
        else:
            self._batch_stats.fill(0)
            objective = np.float64(0)
            centroid_norm = None
            accumulation = (np.zeros(self._batch_stats.shape, dtype="float64")
                            if n > 2**24 else self._batch_stats)
        sums, counts = accumulation[:, :-1], accumulation[:, -1]
        chunk_size = self.recipe["assignment_chunk_size"]
        for begin in range(0, n, chunk_size):
            end = min(begin + chunk_size, n)
            rows = x[begin:end] if index is None else x[index[begin:end]]
            distance, labels = self._assign_rows(rows, self._centroids, centroid_norm)
            if self._is_torch:
                objective += distance.sum(dtype=torch.float64)
                if operand is None:
                    if self._stats_operand is None:
                        with torch.inference_mode(self._batch_stats.is_inference()):
                            self._stats_operand = self._centroids.new_ones(
                                (min(chunk_size, self.recipe["batch_size"]), self.d + 1))
                    operand = (self._stats_operand if accumulation is self._batch_stats
                               else self._stats_operand.double())
                values = operand[:len(rows)]
                values[:, :-1].copy_(rows)
                accumulation.index_add_(0, labels, values)
            else:
                objective += distance.sum(dtype="float64")
                counts += np.bincount(labels, minlength=self.k)
                np.add.at(sums, labels, rows)
        if n > 2**24:
            if self._is_torch:
                self._batch_stats.copy_(accumulation)
            else:
                np.copyto(self._batch_stats, accumulation, casting="unsafe")
            sums, counts = self._batch_stats[:, :-1], self._batch_stats[:, -1]
        stats = {"sums": sums, "counts": counts, "objective": objective, "n": n}
        return stats

    def _prepare_statistics(self, stats, expected_n):
        """Validate one contribution and return statistics for the shared update."""
        sums, counts, objective, n = (
            stats["sums"], stats["counts"], stats["objective"], stats["n"]
        )
        for value, shape, dtype in (
            (sums, (self.k, self.d), "float32"),
            (counts, (self.k,), "float32"),
            (objective, (), "float64"),
        ):
            if self._is_torch:
                import torch

                if not isinstance(value, torch.Tensor):
                    raise TypeError("statistics must be tensors")
                if (value.layout != torch.strided or value.shape != shape
                        or value.dtype != getattr(torch, dtype)
                        or value.device != self._centroids.device):
                    raise ValueError(
                        "statistics have incompatible shape, dtype or device"
                    )
            elif (not isinstance(value, (np.ndarray, np.generic))
                    or value.shape != shape or value.dtype != np.dtype(dtype)):
                raise ValueError("statistics have incompatible shape or dtype")
        if self._is_torch:
            sums, counts, objective = sums.detach(), counts.detach(), objective.detach()
        if not isinstance(n, (int, np.integer)) or n != expected_n:
            raise ValueError("stats n must match the prescribed batch size")
        return None, sums, counts, objective

    def update(self, stats):
        """Consume one local frozen-snapshot batch and perform EMA then revival.

        ``n`` equals the prescribed batch size. CPU communication may combine
        statistics externally before this call. Values and row membership remain
        caller obligations; shape, dtype, device and host row count are checked.
        """
        self._require_running()
        if self._epoch_steps == self.recipe["steps_per_pass"]:
            raise RuntimeError("call end_epoch before the next update")
        expected_n = min(self.recipe["batch_size"], self._n - self._epoch_rows)
        ids, sums, counts, objective = self._prepare_statistics(stats, expected_n)
        alpha = _scheduled_alpha(
            self.recipe["alpha0"], self._step, self.recipe["total_steps"]
        )
        self._apply_statistics(ids, sums, counts, objective,
                               alpha, 1.0 - alpha, alpha == 1)
        self._finish_step(expected_n)

    def _apply_statistics(self, ids, sums, counts, objective,
                          alpha, decay, replace_usage=False):
        # Capture already-global statistics once before reuse/EMA/revival.
        self._objective += objective  # Local contribution; reduced once per epoch.
        if ids is None:
            self._pass_counts += counts
            if self._pass_sums is not None:
                self._pass_sums += sums
        else:
            self._pass_counts.index_add_(0, ids, counts.double())
            if self._pass_sums is not None:
                self._pass_sums.index_add_(0, ids, sums)
        if self._is_torch:
            import torch

            touched = counts > 0
            # Fixed-shape masks avoid dynamic row selection/host reads. Inactive
            # lanes divide zero by one; only touched lanes divide by statistics.
            safe_counts = torch.where(touched, counts, 1)
            mean = sums.clone()
            mean.masked_fill_(~touched[:, None], 0)
            mean /= safe_counts[:, None]
            if not replace_usage:
                self._usage *= decay
            usage = self._usage if ids is None else self._usage[ids]
            centroids = self._centroids if ids is None else self._centroids[ids]
            retained = usage if not replace_usage else torch.zeros_like(usage)
            mass = retained + alpha * counts
            weight = alpha * counts / torch.where(touched, mass, 1)
            position = mean - centroids
            position.mul_(weight[:, None]).add_(centroids)
            torch.where((weight == 1)[:, None], mean, position, out=position)
            torch.where(touched[:, None], position, centroids, out=centroids)
            torch.where(touched, mass, usage, out=usage)
            if ids is not None:
                self._centroids.index_copy_(0, ids, centroids)
                self._usage.index_copy_(0, ids, usage)
        else:
            touched = counts > 0
            if not replace_usage:
                self._usage *= decay
            retained = (
                self._usage[touched] if not replace_usage
                else np.zeros_like(counts[touched])
            )
            mass = retained + alpha * counts[touched]
            mean = sums[touched] / counts[touched, None]
            weight = alpha * counts[touched] / mass
            position = self._centroids[touched] + weight[:, None] * (
                mean - self._centroids[touched]
            )
            position[weight == 1] = mean[weight == 1]
            self._centroids[touched] = position
            self._usage[touched] = mass

    def _finish_step(self, expected_n):
        if (self._step + 1) % _CHECK_UNUSED_EVERY == 0:
            self._revive_unused(self._split_rng)
        self._step += 1
        self._epoch_steps += 1
        self._epoch_rows += expected_n
        self._pending_stats = None

    def _epoch_objective(self):
        return self._objective

    def end_epoch(self):
        """Finish one complete pass and record its moving-pass objective."""
        self._require_running()
        if (self._pending_stats is not None or self._epoch_rows != self._n
                or self._epoch_steps != self.recipe["steps_per_pass"]):
            raise RuntimeError("epoch is incomplete; consume all prescribed batches")
        if self._is_torch:
            import torch

            objective = self._epoch_objective()
            objective, count_square = torch.stack(
                (objective, torch.dot(self._pass_counts, self._pass_counts))
            ).tolist()
        else:
            objective = float(self._objective)
            count_square = float(np.dot(self._pass_counts, self._pass_counts))
        stat = {
            "obj": objective,
            "time": time.perf_counter() - self._start_time,
            "imbalance_factor": self.k * count_square / (self._n * self._n),
            "nsplit": self.n_splits - self._splits_before,
        }
        self.iteration_stats.append(stat)
        if self.verbose:
            print(
                "e-means iteration %d: objective=%.6g imbalance=%.3f nsplit=%d"
                % (self._epoch + 1, objective,
                   stat["imbalance_factor"], stat["nsplit"])
            )
        self._epoch += 1
        if self._epoch < self.niter:
            self._start_epoch()

    def finalize(self):
        """Set and return FP32 closing means from the final pass's assignments.

        Never-assigned centroids retain their current positions. No new search
        is performed. All epochs must be complete; repeated finalization returns
        the same result.
        """
        if self.centroids is not None:
            return self.centroids
        if self._centroids is None or self._epoch != self.niter:
            raise RuntimeError("complete all epochs before finalize")
        nonempty = self._pass_counts > 0
        centroids = self._centroids if self._is_torch else self._centroids.copy()
        objectives = [stat["obj"] for stat in self.iteration_stats]
        if self._is_torch:
            import torch

            mean = self._pass_sums / self._pass_counts.clamp_min(1).float()[:, None]
            self.centroids = torch.where(
                nonempty[:, None], mean, centroids
            ).contiguous()
            self.obj = torch.tensor(
                objectives, dtype=torch.float64, device=centroids.device
            )
        else:
            centroids[nonempty] = (
                self._pass_sums[nonempty]
                / self._pass_counts[nonempty, None].astype("float32")
            )
            self.centroids = np.ascontiguousarray(centroids, dtype="float32")
            self.obj = np.asarray(objectives, dtype="float64")
        self._centroids = self._usage = self._batch_stats = None
        self._pass_sums = self._pass_counts = self._objective = None
        self._pending_stats = None
        if self._is_torch:
            self._stats_operand = None
        return self.centroids

    def _replay_eligible(self, x):
        # Caller-driven/custom runs keep their original eager implementation.
        if (type(self) is not Emeans or not self._is_torch or not x.is_cuda
                or self.batch_size is not None or self.alpha0 is not None
                or self.assignment_chunk_size is not None
                or self.centroid_chunk_size != 4096
                or self.recipe["batch_size"] > 2**24
                or self.recipe["steps_per_pass"] < 3):
            return False
        return x.is_contiguous()

    def _capture_step(self, x, index, rate, batches, epoch_rates, position):
        import torch

        current = torch.cuda.current_stream(x.device)
        stream = torch.cuda.Stream(device=x.device)
        graph = torch.cuda.CUDAGraph()
        # Each epoch owns its pool; retired host pool IDs cannot be recaptured.
        pool = torch.cuda.MemPool()
        captured = False
        stats = None
        try:
            stream.wait_stream(current)
            with torch.cuda.stream(stream):
                torch.cuda.current_blas_handle()
                # The context manager empties process caches; use primitives.
                graph.capture_begin(pool=pool.id, capture_error_mode="thread_local")
                try:
                    torch.index_select(batches, 0, position, out=index.unsqueeze(0))
                    torch.index_select(epoch_rates, 0, position, out=rate.unsqueeze(0))
                    stats = self._accumulate_core(x, index)
                    self._apply_statistics(None, stats["sums"], stats["counts"],
                                           stats["objective"], rate[0], rate[1])
                    position.add_(1)
                finally:
                    graph.capture_end()
            captured = True
            current.wait_stream(stream)
        finally:
            if not captured:
                # No retry/rollback of failed capture or execution errors.
                stats = None
                graph.reset()
        return graph, pool

    def train(self, x, init_centroids=None):
        """Train a dense FP32 matrix using the same caller-driven numerical loop.

        Return the final moving-pass objective. Initialization and epoch shuffles
        use the existing local seeded policy. No whole logical batch is gathered.
        Explicit initial centroids must match x's array type and device.
        """
        is_torch = _is_torch_tensor(x)
        x = _as_float32_matrix(x, "x", is_torch)
        n, d = x.shape
        if d != self.d or n < self.k:
            raise ValueError("x must have dimension d and at least k rows")
        device = x.device if is_torch else None
        if init_centroids is None:
            init_centroids = (
                x if n == self.k
                else x[_randperm(n, self.seed, is_torch, device)[:self.k]]
            )
        elif _is_torch_tensor(init_centroids) != is_torch:
            raise TypeError("x and init_centroids must use the same array type")
        if is_torch and init_centroids.device != device:
            raise ValueError("x and init_centroids must be on the same device")
        self.begin(n, init_centroids)
        rates = None
        order = index = rate = epoch_rates = position = batches = None
        graph_has_sums = False
        if self._replay_eligible(x):
            import torch

            alphas = [_scheduled_alpha(self.recipe["alpha0"], step,
                                       self.recipe["total_steps"])
                      for step in range(self.recipe["total_steps"])]
            # Independently round the host-computed alpha and decay to FP32.
            rates = torch.tensor([(a, 1.0 - a) for a in alphas],
                                 dtype=torch.float32, device=device)
            index = torch.empty(self.recipe["batch_size"], dtype=torch.int64,
                                device=device)
            rate = torch.empty(2, dtype=torch.float32, device=device)
            epoch_rates = rate.new_empty((self.recipe["steps_per_pass"], 2))
            position = torch.zeros(1, dtype=torch.int64, device=device)
        graph = pool = None
        try:
            for epoch in range(self.niter):
                shuffled = _randperm(n, _epoch_seed(self.seed, epoch), is_torch, device)
                if rates is None or order is None:
                    order = shuffled
                else:
                    # Capture reads this storage; eager primes/tails use its new values.
                    order.copy_(shuffled)
                shuffled = None
                primed = False
                if rates is not None:
                    # Stable storage, original epoch order and independently rounded rates.
                    batches = order[:n - n % len(index)].view(-1, len(index))
                    epoch_rates.copy_(rates[self._step:self._step + self.recipe["steps_per_pass"]])
                    position.zero_()
                for begin in range(0, n, self.recipe["batch_size"]):
                    batch = order[begin:begin + self.recipe["batch_size"]]
                    replay = (rates is not None and len(batch) == len(index)
                              and alphas[self._step] != 1)
                    if replay and primed:
                        if graph is None:
                            graph_has_sums = self._pass_sums is not None
                            graph, pool = self._capture_step(
                                x, index, rate, batches, epoch_rates, position)
                        graph.replay()
                        if self._pass_sums is not None and not graph_has_sums:
                            # Same dense ATen closing add, before revival/next stats zero.
                            self._pass_sums.add_(self._batch_stats[:, :-1])
                        # Replay writes do not execute Python mutation notifications.
                        torch.autograd.graph.increment_version((self._centroids, self._usage))
                        self._finish_step(len(batch))
                    else:
                        stats = self._accumulate(x, batch)
                        self.update(stats)
                        if rates is not None:
                            position.add_(1)
                        primed = primed or replay
                self.end_epoch()
                # Reporting drains work; retain the owned graph/pool for the next epoch.
        except BaseException:
            if rates is not None:
                torch.cuda.current_stream(device).synchronize()
            raise
        finally:
            if graph is not None:
                graph.reset()
            graph = pool = None
            order = shuffled = batches = None
            index = rate = epoch_rates = position = None
        self.finalize()
        return self.iteration_stats[-1]["obj"]

    def assign(self, x):
        """Return squared L2 distances and centroid indices."""
        if self.centroids is None:
            raise RuntimeError("train must be called before assign")
        is_torch = _is_torch_tensor(self.centroids)
        if _is_torch_tensor(x) != is_torch:
            raise TypeError("x and centroids must use the same array type")
        if is_torch:
            import torch

        x = _as_float32_matrix(x, "x", is_torch)
        centroids = self.centroids.detach() if is_torch else self.centroids
        assignment_chunk_size = self._get_assignment_chunk_size(
            is_torch, is_torch and self.centroids.is_cuda)
        if x.shape[1] != self.d:
            raise ValueError(
                "x has dimension %d, expected %d" % (x.shape[1], self.d)
            )
        if is_torch and x.device != centroids.device:
            raise ValueError("x and centroids must be on the same device")

        if is_torch:
            distance = torch.empty(
                len(x), dtype=torch.float32, device=x.device
            )
            labels = torch.empty(
                len(x), dtype=torch.int64, device=x.device
            )
            centroid_norm = (centroids * centroids).sum(dim=1)
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
