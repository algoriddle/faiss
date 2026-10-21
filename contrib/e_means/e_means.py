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


def _as_input_matrix(x, name, is_torch, allow_empty=False):
    if not is_torch:
        return _as_float32_matrix(x, name, False, allow_empty)
    import torch

    if not isinstance(x, torch.Tensor):
        raise TypeError("%s must be a torch.Tensor" % name)
    if x.dtype not in (torch.float32, torch.float16, torch.bfloat16,
                       getattr(torch, "float8_e4m3fn", None), torch.int8, torch.uint8):
        raise ValueError("%s must be FP32, FP16, BF16, E4M3, INT8 or UINT8" % name)
    if x.device.type != "cuda" and x.dtype in (
            getattr(torch, "float8_e4m3fn", None), torch.int8, torch.uint8):
        raise ValueError("native code input requires CUDA")
    # Reuse shape/device checks; CPU computes in FP32 on represented values.
    if x.device.type == "cpu":
        return _as_float32_matrix(x.float(), name, True, allow_empty)
    # Metadata checks must not widen a resident CUDA population.
    if x.layout != torch.strided or x.ndim != 2:
        raise ValueError("%s must be a dense 2-D tensor" % name)
    if (not len(x) and not allow_empty) or not x.shape[1]:
        raise ValueError("%s must not be empty" % name)
    if x.device.type != "cuda":
        raise ValueError("%s must be on CPU or CUDA" % name)
    return x.detach().contiguous()


def _check_torch_abs_bound(name, bound, *tensors):
    import torch

    extrema = torch.stack([value for x in tensors for value in torch.aminmax(x)])
    if not bool((extrema.abs() <= bound).all()):
        raise ValueError("%s must be finite with abs(value) <= %g" % (name, bound))


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


def _separate_splits(donor, child, dtype=None):
    """Move one coordinate only when represented split operands coincide.

    Zero donors become distinct subnormal operands; their squared distances
    can still tie. No value scan or dynamic collapsed-row list is needed.
    """
    if _is_torch_tensor(child):
        import torch

        donor_q = donor if dtype is None else donor.to(dtype)
        child_q = child if dtype is None else child.to(dtype)
        collapsed = (donor_q == child_q).all(dim=1)
        value = child_q[:, 0]
        neighbor = torch.nextafter(value, torch.full_like(value, float("inf")))
        inward = torch.nextafter(value, torch.full_like(value, float("-inf")))
        neighbor = torch.where(torch.isfinite(neighbor), neighbor, inward)
        child[:, 0] = torch.where(collapsed, neighbor.float(), child[:, 0])
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


def _get_triton_assign_rows():
    try:
        from .e_means_triton import assign_rows
    except ImportError as error:
        raise ImportError("Triton assignment requires the triton package") from error
    return assign_rows


def _get_triton_int8_functions():
    try:
        from .e_means_triton import assign_int8_rows, prepare_int8_centroids
    except ImportError as error:
        raise ImportError("Triton assignment requires the triton package") from error
    return assign_int8_rows, prepare_int8_centroids


def _get_cute_functions():
    try:
        from . import e_means_cute as cute
    except ImportError as error:
        raise ImportError("CuTe assignment requires its pinned optional dependencies") from error
    return cute._assign_rows, cute._prepare_centroids, cute._check_device


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
        Maximum centroids examined at once by the PyTorch assignment path.

    assignment_backend : {"native", "triton", "cute"}, optional
        Native is the default. Triton requires NVIDIA CUDA SM80 or later and
        a compatible Triton installation. It uses bounded tiled/split assignment,
        FP32 dot input precision selected by ``fp32_dot``, and no runtime
        autotuning or backend fallback.
        centroid_chunk_size applies only to native Torch assignment.
        CuTe supports native BF16/E4M3/INT8/UINT8 or a matching explicit
        quantizer on exactly SM90,
        with quack-kernels==0.6.5, nvidia-cutlass-dsl==4.7.1 and cuda-python>=13,<14.
        E4M3 uses tensor-core partial accumulation with FP32 promotion every
        128 contraction coordinates. Ranking is approximate; distances are
        recomputed from the selected represented operands.
        Floating assignment ranks FP32 scores, choosing lower IDs for bit-equal scores;
        a float64-only tie need not select the lowest ID. No fallback.
        Calls using one model share mutable CuTe workspace and require caller ordering.
    quantizer : CastQuantizer, Fp8Quantizer or Int8Quantizer, optional
        Explicit stateless FP16/BF16 conversion for CUDA batches. None uses
        native FP32/FP16/BF16 input, fixed by the first batch including an empty
        one. Centroid operands and FP32 statistics use those same represented
        values. Fp8Quantizer requires explicit calibration/device placement and
        Triton or CuTe: floating batches encode per chunk; matching native E4M3 codes
        bypass encoding. Statistics, centroids and distances use decoded units.
        None with native E4M3 uses raw code values, also requiring Triton or CuTe.
        Int8Quantizer accepts matching native INT8 codes only, requires Triton or CuTe,
        and returns decoded units. Raw INT8/UINT8 with None retain numeric units;
        UINT8 translation is internal. INT8 ranking is exact with d <= 43919,
        qualified NVIDIA SM80/SM90/SM100, and FP32 returned distances.
        CPU computes in FP32 and requires quantizer=None.
    fp32_dot : {"ieee", "tf32x3"}, optional
        How Triton assignment multiplies FP32 rows by FP32 centroids. "ieee"
        (the default) uses exact FP32 products without tensor cores. "tf32x3"
        splits each operand into a TF32 value and a TF32 remainder and sums
        three tensor-core products, which is several times faster and close to
        FP32 accuracy; on data whose coordinates are very large compared with
        the distances between centroids in two or three dimensions, "ieee"
        stays more accurate. Requires ``assignment_backend="triton"`` and FP32
        rows (no quantizer).

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
    finite-value validation scan the input. CUDA accepts native FP32/FP16/BF16
    and requires caller-validated finite values fitting the represented format;
    metadata checks do not scan CUDA values.
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
        quantizer=None,
        assignment_backend="native",
        fp32_dot="ieee",
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
        self.assignment_backend = assignment_backend
        if assignment_backend not in ("native", "triton", "cute"):
            raise ValueError("assignment_backend must be native, triton or cute")
        self.fp32_dot = fp32_dot
        if fp32_dot not in ("ieee", "tf32x3"):
            raise ValueError("fp32_dot must be 'ieee' or 'tf32x3'")
        if fp32_dot != "ieee" and (assignment_backend != "triton" or quantizer is not None):
            raise ValueError("fp32_dot='tf32x3' requires Triton assignment of FP32 rows")
        if assignment_backend == "triton" and self.k > 2**31 - 256:
            raise ValueError("Triton assignment requires k <= 2**31 - 256")
        if assignment_backend == "cute" and self.k > 2**31 - 192:
            raise ValueError("CuTe padded centroid IDs must fit signed int32")
        self.quantizer = quantizer
        self._fp8_quantizer = self._int8_quantizer = False
        if quantizer is not None:
            from .e_means_quantizer import CastQuantizer
            from .e_means_fp8 import Fp8Quantizer
            from .e_means_int8 import Int8Quantizer

            if not isinstance(quantizer, (CastQuantizer, Fp8Quantizer, Int8Quantizer)) or quantizer.d != self.d:
                raise ValueError("quantizer must be a matching CastQuantizer, Fp8Quantizer or Int8Quantizer")
            self._fp8_quantizer = isinstance(quantizer, Fp8Quantizer)
            self._int8_quantizer = isinstance(quantizer, Int8Quantizer)
            if self._int8_quantizer:
                if assignment_backend not in ("triton", "cute"):
                    raise ValueError("INT8 requires Triton or CuTe assignment")
                if self.d > 43919:
                    raise ValueError("INT8 assignment requires d <= 43919")
            if self._fp8_quantizer and assignment_backend not in ("triton", "cute"):
                raise ValueError("FP8 requires Triton or CuTe assignment")

        if assignment_backend == "cute" and quantizer is not None:
            import torch

            if quantizer.dtype not in (torch.bfloat16, torch.float8_e4m3fn, torch.int8):
                raise ValueError("CuTe assignment requires BF16, E4M3 or INT8")

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
        self._input_dtype = None
        self._fp8_peak = None
        self._cute_assignment_cache = None

    def _get_assignment_chunk_size(self, is_torch):
        if self.assignment_chunk_size is not None:
            return self.assignment_chunk_size
        return 8192 if is_torch else 65536

    def _prepare_input(self, x, allow_empty=False):
        state = self._centroids if self._centroids is not None else self.centroids
        is_torch = _is_torch_tensor(state)
        if _is_torch_tensor(x) != is_torch:
            raise TypeError("x and centroids must use the same array type")
        if self.quantizer is not None and (not is_torch or state.device.type != "cuda"):
            raise ValueError("CPU E-means requires quantizer=None")
        if self.assignment_backend == "cute" and (not is_torch or state.device.type != "cuda"):
            raise ValueError("CuTe assignment requires CUDA tensors")
        if self.assignment_backend == "triton":
            if not is_torch or state.device.type != "cuda":
                raise ValueError("Triton assignment requires CUDA tensors")
            _get_triton_assign_rows()
        x = _as_input_matrix(x, "x", is_torch, allow_empty)
        if x.shape[1] != self.d:
            raise ValueError("x must have dimension d")
        if not is_torch:
            return x
        if x.device != state.device:
            raise ValueError("x and centroids must be on the same device")
        import torch

        dtype = self.quantizer.dtype if self.quantizer is not None else x.dtype
        if self.assignment_backend == "cute":
            if dtype not in (torch.bfloat16, torch.float8_e4m3fn, torch.int8, torch.uint8):
                raise ValueError("CuTe assignment requires native BF16/E4M3/INT8/UINT8 or a matching quantizer")
            _, _, check = _get_cute_functions()
            check(x.device)
        if dtype in (torch.int8, torch.uint8):
            from .e_means_int8 import _check_int8_support

            if self.assignment_backend not in ("triton", "cute"):
                raise ValueError("INT8 requires Triton or CuTe assignment")
            if self.d > 43919:
                raise ValueError("INT8 assignment requires d <= 43919")
            if x.dtype != dtype:
                raise TypeError("INT8 training requires matching native codes; no JIT INT8")
            _check_int8_support(x.device)
        elif dtype == getattr(torch, "float8_e4m3fn", None):
            from .e_means_fp8 import _check_fp8_support

            if self.assignment_backend not in ("triton", "cute"):
                raise ValueError("FP8 requires Triton or CuTe assignment")
            _check_fp8_support(x.device)
        elif x.dtype in (getattr(torch, "float8_e4m3fn", None), torch.int8, torch.uint8):
            raise ValueError("native codes must match the quantizer dtype")
        if self._fp8_quantizer or self._int8_quantizer:
            self.quantizer._validate(x, "x")
        if x.device.type == "cuda" and (dtype != torch.float32 or self.assignment_backend == "triton"):
            if (getattr(torch.version, "hip", None) is not None
                    or torch.cuda.get_device_capability(x.device) < (8, 0)):
                raise RuntimeError("selected assignment requires NVIDIA SM80 or later")
        if self._input_dtype is not None and dtype != self._input_dtype:
            raise ValueError("native input dtype must be consistent across the run")
        self._input_dtype = dtype
        if self.recipe is not None:
            self.recipe["representation"] = str(dtype).removeprefix("torch.")
        return x

    def _prepare_rows(self, rows):
        if self._int8_quantizer and rows.dtype != self.quantizer.dtype:
            raise TypeError("INT8 training requires matching native codes; no JIT INT8")
        if (self.assignment_backend == "cute" and self._fp8_quantizer
                and rows.dtype != self.quantizer.dtype):
            return rows.float()
        if self.quantizer is None or rows.dtype == self.quantizer.dtype:
            return rows
        return self.quantizer.compute_codes(rows)

    def _prepare_torch_assignment(self, centroids):
        import torch

        if self.assignment_backend == "cute":
            _, prepare, _ = _get_cute_functions()
            dtype = self._input_dtype or torch.bfloat16
            if dtype is torch.float8_e4m3fn and self._fp8_peak is None:
                self._fp8_peak = centroids.new_zeros(())
            self._cute_assignment_cache = prepare(
                centroids, self._cute_assignment_cache,
                transform=self.quantizer if self._fp8_quantizer or self._int8_quantizer else None,
                peak_out=self._fp8_peak, dtype=dtype)
            return self._cute_assignment_cache, None
        dtype = self._input_dtype or torch.float32
        if dtype in (torch.int8, torch.uint8):
            from .e_means_quantizer import _transform
            from .e_means_int8 import representation

            _, prepare = _get_triton_int8_functions()
            value = _transform(centroids, self.quantizer if self._int8_quantizer else None)
            return prepare(value, representation(dtype), validate=False)
        if dtype == getattr(torch, "float8_e4m3fn", None):
            from .e_means_fp8 import _transform, _update_fp8_peak

            if self._fp8_peak is None:
                self._fp8_peak = centroids.new_zeros(())
            transformed = _transform(centroids, self.quantizer)
            _update_fp8_peak(transformed, self._fp8_peak)
            represented = transformed.to(dtype)
            if self.d % 64:
                from .e_means_triton import _pad_fp8

                represented = _pad_fp8(represented)
        else:
            represented = centroids.to(dtype)
        return represented, represented.float().square().sum(dim=1)

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
        donor_centroids = centroids * (1.0 - _SPLIT_EPSILON * sign)
        child_centroids = centroids * (1.0 + _SPLIT_EPSILON * sign)
        if self._input_dtype is not None:
            import torch

            if self._input_dtype == torch.float16:
                # Preserve the existing guard on internally generated splits.
                _check_torch_abs_bound(
                    "revived centroids", 65504.0, donor_centroids, child_centroids
                )
        if _is_torch_tensor(child_centroids) and self._input_dtype in (torch.int8, torch.uint8):
            from .e_means_int8 import representation, separate_splits

            # Retain the source guard on internally generated finite FP32 splits.
            _check_torch_abs_bound("revived centroids", torch.finfo(torch.float32).max,
                                   donor_centroids, child_centroids)
            separate_splits(donor_centroids, child_centroids,
                            representation(self._input_dtype), self.quantizer)
        elif _is_torch_tensor(child_centroids) and self._input_dtype == getattr(torch, "float8_e4m3fn", None):
            from .e_means_fp8 import separate_splits

            separate_splits(donor_centroids, child_centroids, self.quantizer)
        else:
            _separate_splits(donor_centroids, child_centroids, self._input_dtype)
        half_usage = self._usage[source] / 2.0

        self._usage[source] = half_usage
        self._centroids[source] = donor_centroids
        self._usage[target] = half_usage
        self._centroids[target] = child_centroids
        self.n_splits += len(source)
        return len(source)

    def _assign_rows(self, x, centroids, centroid_norm=None, distance_out=None, labels_out=None):
        if _is_torch_tensor(x):
            if self.assignment_backend == "cute":
                assign, _, _ = _get_cute_functions()
                return assign(x, centroids, distance_out, labels_out)
            if self.assignment_backend == "triton":
                import torch

                if self._input_dtype in (torch.int8, torch.uint8):
                    if self.fp32_dot != "ieee":
                        raise ValueError("fp32_dot='%s' requires FP32 rows" % self.fp32_dot)
                    from .e_means_int8 import representation

                    assign, _ = _get_triton_int8_functions()
                    distance, labels = assign(x, centroids, centroid_norm,
                                              representation(self._input_dtype))
                else:
                    distance, labels = _get_triton_assign_rows()(
                        x, centroids, centroid_norm, fp32_dot=self.fp32_dot
                    )
                if self._fp8_quantizer or self._int8_quantizer:
                    from .e_means_quantizer import _ldexp

                    distance = _ldexp(distance, self.quantizer._distance_exponent)
                return distance, labels
            import torch

            x_norm = x.float().square().sum(dim=1)
            best_score = torch.full(
                (len(x),), float("inf"), dtype=torch.float32, device=x.device
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
                    **({"out_dtype": torch.float32} if x.dtype != torch.float32 else {}),
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
        if self.assignment_backend == "cute" and (not is_torch or centroids.device.type != "cuda"):
            raise ValueError("CuTe assignment requires CUDA tensors")
        if self.assignment_backend == "triton" and (not is_torch or centroids.device.type != "cuda"):
            raise ValueError("Triton assignment requires CUDA tensors")
        if self.quantizer is not None and (not is_torch or centroids.device.type != "cuda"):
            raise ValueError("CPU E-means requires quantizer=None")
        if self._fp8_quantizer or self._int8_quantizer:
            self.quantizer._validate(centroids, "init_centroids")
        self.reset()
        self._input_dtype = self.quantizer.dtype if self.quantizer is not None else None
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
            "assignment_backend": self.assignment_backend,
            "fp32_dot": self.fp32_dot,
            "assignment_chunk_size": self._get_assignment_chunk_size(is_torch),
        }
        if is_torch:
            self.recipe.update(
                device=str(centroids.device),
                centroid_chunk_size=self.centroid_chunk_size,
            )
            self._stats_operand = centroids.new_ones(
                (min(self.recipe["assignment_chunk_size"], batch_size), self.d + 1)
            )
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
        self._fp8_peak = None
        self._splits_before = self.n_splits
        if self._is_torch:
            import torch

            self._pass_counts = self._usage.new_zeros(self.k, dtype=torch.float64)
            self._objective = self._usage.new_zeros((), dtype=torch.float64)
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

        A typed (0,d) batch yields zeros and selects native input metadata.
        CUDA values must be finite and fit the selected representation; use
        explicit quantizer validation when needed. No CUDA value scan runs here.
        """
        self._require_running()
        x = self._prepare_input(x, allow_empty=True)
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
        if self._is_torch:
            import torch

            self._batch_stats.zero_()
            objective = self._objective.new_zeros(())
            # Empty batches still bind metadata but bypass nonempty preparation.
            centroids, centroid_norm = (
                self._prepare_torch_assignment(self._centroids) if n else (None, None)
            )
        else:
            self._batch_stats.fill(0)
            objective = np.float64(0)
            centroids, centroid_norm = self._centroids, None
        sums, counts = self._batch_stats[:, :-1], self._batch_stats[:, -1]
        chunk_size = self.recipe["assignment_chunk_size"]
        for begin in range(0, n, chunk_size):
            end = min(begin + chunk_size, n)
            rows = x[begin:end] if index is None else x[index[begin:end]]
            rows = self._prepare_rows(rows)
            distance, labels = self._assign_rows(rows, centroids, centroid_norm)
            if self._is_torch:
                objective += distance.sum(dtype=torch.float64)
                values = self._stats_operand[:len(rows)]
                codes = (centroids.bindings[len(rows)].workspace
                         if self.assignment_backend == "cute" and self._fp8_quantizer else rows)
                represented = (self.quantizer.decode(codes)
                               if self.quantizer is not None else rows.float())
                values[:, :-1].copy_(represented)
                self._batch_stats.index_add_(0, labels, values)
            else:
                objective += distance.sum(dtype="float64")
                counts += np.bincount(labels, minlength=self.k)
                np.add.at(sums, labels, rows)
        stats = {"sums": sums, "counts": counts, "objective": objective, "n": n}
        self._pending_stats = stats
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
        alpha = _scheduled_alpha(
            self.recipe["alpha0"], self._step, self.recipe["total_steps"]
        )
        touched = counts > 0
        if self._is_torch:
            import torch

            # Fixed-shape masks avoid dynamic row selection/host reads. Inactive
            # lanes divide zero by one; only touched lanes divide by statistics.
            safe_counts = torch.where(touched, counts, 1)
            mean = sums.clone()
            mean.masked_fill_(~touched[:, None], 0)
            mean /= safe_counts[:, None]
            if alpha != 1:
                self._usage *= 1.0 - alpha
            usage = self._usage if ids is None else self._usage[ids]
            centroids = self._centroids if ids is None else self._centroids[ids]
            retained = usage if alpha != 1 else torch.zeros_like(usage)
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
            if alpha != 1:
                self._usage *= 1.0 - alpha
            retained = (
                self._usage[touched] if alpha != 1
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

            fp8 = self._input_dtype == getattr(torch, "float8_e4m3fn", None)
            if fp8:
                from .e_means_fp8 import _transform, _update_fp8_peak, _check_fp8_range

                if self._fp8_peak is None:
                    self._fp8_peak = self._centroids.new_zeros(())
                _update_fp8_peak(_transform(self._centroids, self.quantizer), self._fp8_peak)
            objective = self._epoch_objective()
            totals = torch.stack(
                (objective, torch.dot(self._pass_counts, self._pass_counts))
                + ((self._fp8_peak.double(),) if fp8 else ())
            ).tolist()
            objective, count_square = totals[:2]
            if fp8:
                _check_fp8_range(totals[-1], "centroids")
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

    def train(self, x, init_centroids=None):
        """Train a dense supported matrix through the same caller-driven loop.

        Return the final moving-pass objective. Initialization and epoch shuffles
        use the existing local seeded policy. No whole logical batch is gathered.
        Explicit initial centroids must match x's array type and device.
        """
        is_torch = _is_torch_tensor(x)
        x = _as_input_matrix(x, "x", is_torch)
        n, d = x.shape
        if d != self.d or n < self.k:
            raise ValueError("x must have dimension d and at least k rows")
        device = x.device if is_torch else None
        if init_centroids is None:
            init_centroids = (
                x if n == self.k
                else x[_randperm(n, self.seed, is_torch, device)[:self.k]]
            )
            if is_torch:
                rows = self._prepare_rows(init_centroids)
                # Fused CuTe JIT defers chunk encoding; seed positions still need codes.
                if (self.assignment_backend == "cute" and self._fp8_quantizer
                        and rows.dtype != self.quantizer.dtype):
                    rows = self.quantizer.compute_codes(rows)
                init_centroids = (self.quantizer.decode(rows)
                                  if self.quantizer is not None else rows.float())
        elif _is_torch_tensor(init_centroids) != is_torch:
            raise TypeError("x and init_centroids must use the same array type")
        if is_torch and init_centroids.device != device:
            raise ValueError("x and init_centroids must be on the same device")
        self.begin(n, init_centroids)
        x = self._prepare_input(x)
        for epoch in range(self.niter):
            order = _randperm(n, _epoch_seed(self.seed, epoch), is_torch, device)
            for begin in range(0, n, self.recipe["batch_size"]):
                stats = self._accumulate(
                    x, order[begin:begin + self.recipe["batch_size"]]
                )
                self.update(stats)
            self.end_epoch()
        self.finalize()
        return self.iteration_stats[-1]["obj"]

    def assign(self, x):
        """Return squared L2 distances and centroid indices.

        Distances use rounded centroid operands in represented-value coordinates.
        """
        if self.centroids is None:
            raise RuntimeError("train must be called before assign")
        is_torch = _is_torch_tensor(self.centroids)
        if _is_torch_tensor(x) != is_torch:
            raise TypeError("x and centroids must use the same array type")
        if is_torch:
            import torch

        x = self._prepare_input(x)
        centroids = (self.centroids.detach()
                     if is_torch and self.assignment_backend != "cute" else self.centroids)
        assignment_chunk_size = self._get_assignment_chunk_size(is_torch)

        if is_torch:
            distance = torch.empty(
                len(x), dtype=torch.float32, device=x.device
            )
            labels = torch.empty(
                len(x), dtype=torch.int64, device=x.device
            )
            if self._input_dtype in (torch.int8, torch.uint8):
                _check_torch_abs_bound("centroids", torch.finfo(torch.float32).max, centroids)
            self._fp8_peak = None
            centroids, centroid_norm = self._prepare_torch_assignment(centroids)
            if self._input_dtype == getattr(torch, "float8_e4m3fn", None):
                from .e_means_fp8 import _check_fp8_range

                _check_fp8_range(float(self._fp8_peak), "centroids")
        else:
            distance = np.empty(len(x), dtype="float32")
            labels = np.empty(len(x), dtype="int64")
            centroid_norm = None

        for begin in range(0, len(x), assignment_chunk_size):
            end = min(begin + assignment_chunk_size, len(x))
            rows = self._prepare_rows(x[begin:end])
            if self.assignment_backend == "cute":
                self._assign_rows(rows, centroids, centroid_norm, distance[begin:end], labels[begin:end])
            else:
                distance[begin:end], labels[begin:end] = self._assign_rows(rows, centroids, centroid_norm)
        return distance, labels
