# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Host adapter for E-means BF16, E4M3, and native INT8 CuTe assignment.

Optional CuTe dependencies are resolved only after an explicit ``cute``
backend request. Assignment selects between the tiled kernel and an admitted
low-dimension row-owned kernel; there is no fallback backend. E4M3 ranks FP32
scores, while native bytes rank exact INT32 scores.
"""

from collections import OrderedDict
from dataclasses import dataclass, field
from functools import lru_cache
import importlib
import importlib.metadata
import importlib.util
import inspect
import weakref


_QUACK_VERSION = "0.6.5"
_CUTLASS_DSL_VERSION = "4.7.1"
_CUDA_PYTHON_MAJOR = 13
_MIN_OPTIN_SMEM = 227 * 1024
_TILE_N = 192
_MAX_BINDINGS = 2


@lru_cache(maxsize=1)
def _stack_status():
    """Return a stable compatibility result without importing on Faiss load."""
    exact_versions = (
        ("quack-kernels", _QUACK_VERSION),
        ("nvidia-cutlass-dsl", _CUTLASS_DSL_VERSION),
    )
    try:
        for distribution, expected in exact_versions:
            if importlib.metadata.version(distribution) != expected:
                return False, "%s version mismatch" % distribution
        cuda_version = importlib.metadata.version("cuda-python")
        if int(cuda_version.split(".", 1)[0]) != _CUDA_PYTHON_MAJOR:
            return False, "cuda-python requires major version 13"
        for module in ("cutlass", "cuda.bindings.driver", "quack"):
            if importlib.util.find_spec(module) is None:
                return False, "%s is unavailable" % module
        cuda_driver = importlib.import_module("cuda.bindings.driver")
        if not callable(getattr(cuda_driver, "CUstream", None)):
            return False, "cuda-python ABI mismatch"
        ops = importlib.import_module("quack.epilogue.ops")
        lane_helper = getattr(ops, "_lane_warp_info_n", None)
        if not callable(lane_helper) or tuple(
            inspect.signature(lane_helper).parameters
        ) != ("tiled_copy", "reference_src", "tidx"):
            return False, "QuACK ABI mismatch"
        for module, symbol in (
            ("quack.gemm_sm90", "GemmSm90"),
            ("quack.tile_scheduler", "TileScheduler"),
            ("quack.gemm_tvm_ffi_utils", "compile_gemm_kernel"),
        ):
            if not hasattr(importlib.import_module(module), symbol):
                return False, "QuACK ABI mismatch"
    except (
        AttributeError,
        ImportError,
        ModuleNotFoundError,
        TypeError,
        ValueError,
        importlib.metadata.PackageNotFoundError,
    ):
        return False, "optional CuTe stack is unavailable"
    return True, "ready"


@lru_cache(maxsize=1)
def _int8_stack_status():
    """Validate the stock-v0.6.5 helper replaced by the local compile hook."""
    try:
        sm90_utils = importlib.import_module("quack.sm90_utils")
        helper = getattr(sm90_utils, "partition_fragment_ABC", None)
        signature = (
            tuple(inspect.signature(helper).parameters)
            if callable(helper)
            else ()
        )
        if not callable(helper) or signature != (
            "thr_mma", "shape_mnk", "sA", "sB", "swap_AB"
        ):
            return False, "QuACK SM90 fragment helper ABI mismatch"
    except (
        AttributeError,
        ImportError,
        ModuleNotFoundError,
        TypeError,
        ValueError,
    ):
        return False, "QuACK SM90 fragment helper is unavailable"
    return True, "ready"


def _require_stack():
    available, reason = _stack_status()
    if not available:
        raise RuntimeError(
            "CuTe assignment requires quack-kernels==%s, "
            "nvidia-cutlass-dsl==%s, and cuda-python>=13,<14 (%s)"
            % (
                _QUACK_VERSION,
                _CUTLASS_DSL_VERSION,
                reason,
            )
        )


def _require_int8_stack():
    """Require the stock helper ABI used by the local INT8 compile hook."""
    _require_stack()
    available, reason = _int8_stack_status()
    if not available:
        raise RuntimeError("CuTe INT8 assignment is unavailable: %s" % reason)


def _check_device(device):
    """Fail before compilation unless the fixed kernel is legal."""
    import torch

    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("CuTe assignment requires a PyTorch CUDA tensor")
    if getattr(torch.version, "hip", None) is not None:
        raise RuntimeError("CuTe assignment requires NVIDIA CUDA SM90")
    with torch.cuda.device(device):
        _require_stack()
        capability = torch.cuda.get_device_capability(device)
        if capability != (9, 0):
            raise RuntimeError(
                "CuTe assignment requires exactly SM90; device is sm%d%d"
                % capability
            )
        properties = torch.cuda.get_device_properties(device)
        smem = getattr(properties, "shared_memory_per_block_optin", None)
        if smem is None:
            smem = getattr(properties, "shared_memory_per_block", 0)
        if smem < _MIN_OPTIN_SMEM:
            raise RuntimeError(
                "CuTe assignment requires at least %d bytes of opt-in "
                "shared memory"
                % _MIN_OPTIN_SMEM
            )


def _padded_extent(value, alignment):
    return ((int(value) + alignment - 1) // alignment) * alignment


def _check_extent(name, value):
    if value < 1 or value >= 2**31:
        raise ValueError("CuTe assignment requires 0 < %s < 2**31" % name)


def _check_padded_span(name, rows, dim, element_size=2):
    pitch = _padded_extent(dim, 16 // element_size)
    if rows * pitch >= 2**31:
        raise ValueError("CuTe padded %s storage must fit signed int32" % name)
    return pitch


def _tensor_version(tensor):
    """Return the mutation version, or disable reuse when unavailable."""
    try:
        return tensor._version
    except (AttributeError, RuntimeError):
        return None


def _aligned_operand(rows, dim, device, dtype):
    import torch

    pitch = _padded_extent(dim, 16 // dtype.itemsize)
    if pitch == dim:
        return torch.empty((rows, dim), dtype=dtype, device=device)
    return torch.zeros((rows, pitch), dtype=dtype, device=device)[
        :, :dim
    ]


@dataclass
class _PreparedCentroids:
    source_ref: object
    source_version: int
    transform: object
    transform_token: object
    centroids: object
    norms: object
    padded_norms: object
    finite_validated: bool
    bindings: OrderedDict = field(default_factory=OrderedDict)

    @property
    def num_centroids(self):
        return self.centroids.shape[0]

    @property
    def dim(self):
        return self.centroids.shape[1]


@dataclass
class _Binding:
    workspace: object
    codes: object
    scores: object
    launches: tuple
    peak: object
    row_norms: object = None
    distances: object = None
    labels: object = None
    prepared_row_norms: object = None
    shared_stage_norm_profitable: bool = False


def _can_refresh(cached, centroids):
    return (
        isinstance(cached, _PreparedCentroids)
        and cached.centroids.device == centroids.device
        and cached.centroids.shape == centroids.shape
    )


def _transform_token(transform):
    if transform is None:
        return None
    representation = getattr(transform, "representation", None)
    if representation is not None:
        return representation
    # The prepared object retains transform, so its identity cannot be reused.
    bias = transform._bias
    return (
        id(transform),
        transform.scale,
        None
        if bias is None
        else (bias.data_ptr(), _tensor_version(bias), bias.device),
    )


def _refresh_prepared(
    cached, centroids, source_version, transform, peak_out,
    validate_centroids,
):
    import torch

    if cached.centroids.dtype is torch.int8:
        from faiss.contrib.e_means_cute_argmin import stage_int8_rows
        from faiss.contrib.e_means_int8 import (
            _ENCODE_CHUNK_ELEMENTS,
            quantize,
        )

        if validate_centroids:
            if not bool(torch.isfinite(centroids).all()):
                raise ValueError("INT8 centroids must be finite")
            cached.finite_validated = True
        else:
            cached.finite_validated = False
        chunk = max(1, _ENCODE_CHUNK_ELEMENTS // centroids.shape[1])
        for begin in range(0, len(centroids), chunk):
            end = min(begin + chunk, len(centroids))
            mirror = cached.centroids[begin:end]
            mirror.copy_(quantize(centroids[begin:end].detach(), transform))
            stage_int8_rows(
                mirror, mirror, cached.norms[begin:end], "int8"
            )
    elif transform is None:
        cached.centroids.copy_(centroids.detach())
        cached.norms.copy_(
            cached.centroids.float().square().sum(dim=1)
        )
    else:
        from faiss.contrib.e_means_cute_argmin import affine_cast
        from faiss.contrib.e_means_fp8 import _check_fp8_range

        local_peak = peak_out
        if local_peak is None:
            local_peak = centroids.new_zeros(())
        affine_cast(
            centroids.detach(),
            transform._bias,
            transform.scale,
            cached.centroids,
            local_peak,
        )
        wide = cached.centroids.float()
        wide.square_()
        torch.sum(wide, dim=1, out=cached.norms)
        del wide
        if peak_out is None:
            _check_fp8_range(local_peak, "centroids")
    cached.source_ref = weakref.ref(centroids)
    cached.source_version = source_version
    cached.transform = transform
    cached.transform_token = _transform_token(transform)
    return cached


def _prepare_centroids(
    centroids, cached=None, transform=None, peak_out=None,
    validate_centroids=True,
):
    """Round one FP32 centroid state and compute norms from that mirror.

    Repeated public assignment reuses an unchanged installed centroid tensor.
    A new same-shape training readout refreshes its mirror and norms in place,
    preserving pointer-bound assignment workspaces across statistical batches.
    Refresh and assignment are ordered on the caller's current CUDA stream.
    """
    import torch

    if not isinstance(centroids, torch.Tensor):
        raise TypeError("centroids must be a tensor")
    _check_device(centroids.device)
    if centroids.ndim != 2:
        raise ValueError("centroids must be rank two")
    if (
        not centroids.is_cuda
        or centroids.dtype is not torch.float32
        or not centroids.is_contiguous()
    ):
        raise ValueError("centroids must be contiguous CUDA FP32")
    k, dim = centroids.shape
    _check_extent("K", k)
    _check_extent("d", dim)
    representation = getattr(transform, "representation", None)
    if representation not in (None, "int8", "uint8"):
        raise ValueError("unknown native-byte representation")
    if representation is not None:
        _require_int8_stack()
    if transform is None:
        dtype = torch.bfloat16
    elif representation is not None:
        dtype = torch.int8
    else:
        dtype = torch.float8_e4m3fn
    _check_padded_span("centroid", k, dim, dtype.itemsize)
    padded_k = _padded_extent(k, _TILE_N)
    if padded_k >= 2**31:
        raise ValueError("CuTe padded centroid IDs must fit signed int32")
    source_version = _tensor_version(centroids)
    transform_token = _transform_token(transform)
    if (
        isinstance(cached, _PreparedCentroids)
        and cached.source_ref() is centroids
        and source_version is not None
        and cached.source_version == source_version
        and cached.transform_token == transform_token
    ):
        if (
            representation is not None
            and validate_centroids
            and not cached.finite_validated
        ):
            if not bool(torch.isfinite(centroids).all()):
                raise ValueError("INT8 centroids must be finite")
            cached.finite_validated = True
        return cached

    with torch.cuda.device(centroids.device):
        if _can_refresh(cached, centroids) and cached.centroids.dtype is dtype:
            return _refresh_prepared(
                cached, centroids, source_version, transform, peak_out,
                validate_centroids,
            )
        mirror = _aligned_operand(k, dim, centroids.device, dtype)
        norm_dtype = torch.int32 if dtype is torch.int8 else torch.float32
        storage = torch.empty(
            padded_k, dtype=norm_dtype, device=centroids.device
        )
        if padded_k != k:
            # TMA zero-fills padded centroid rows, so their dot term is zero.
            # The INT32_MAX norm sentinel remains strictly above every legal
            # real score admitted by the raw-score guard and cannot win.
            storage[k:].fill_(
                torch.iinfo(torch.int32).max
                if dtype is torch.int8
                else float("inf")
            )
        prepared = _PreparedCentroids(
            None,
            None,
            None,
            None,
            mirror,
            storage[:k],
            storage.view(1, padded_k),
            False,
        )
        return _refresh_prepared(
            prepared, centroids, source_version, transform, peak_out,
            validate_centroids,
        )


def _float_row_owned_region(batch, clusters):
    """Return whether B/K lies in a measured floating region."""
    if batch == 8192:
        return clusters in (16384, 65536, 262144)
    return batch == 16384 and clusters <= 16384


def _float_row_owned_resources(
    dtype, batch, clusters, pitch_bytes, plan
):
    """Apply dtype-specific legality after B/K admission."""
    import torch

    if plan is None or not _float_row_owned_region(batch, clusters):
        return False
    if batch == 16384:
        return True
    if dtype is torch.bfloat16:
        return pitch_bytes % 32 == 0
    # One-byte operands were insensitive to pitch at K=16384. Above that,
    # admit either the small K tile or a 32-byte-aligned row pitch.
    return clusters == 16384 or plan[0] <= 128 or pitch_bytes % 32 == 0


def _float_row_owned_overlap(batch, clusters):
    """Return whether an admitted floating plan uses faithful overlap.

    The extra schedule is retained for the measured BF16 K=65536 assignment
    win. It is not needed by the broader serial row-owned gains.
    """
    return batch == 8192 and clusters in (16384, 65536)


def _shared_stage_norm_pressure_bytes(batch, dim, clusters):
    """Return the selector's FP32-temporary plus E4M3-mirror scale."""
    padded_dim = (dim + 15) // 16 * 16
    return 4 * batch * dim + clusters * padded_dim


def _shared_stage_norm_profitable(batch, dim, clusters, l2_cache_size):
    """Return the H100-calibrated shared-stage admission decision."""
    if l2_cache_size is None:
        return False
    # H100/SM90 evidence uses L2 capacity as an empirically bracketed scale,
    # not proven cache causality. Other architectures must revalidate even
    # with the same L2 size; a different size alone does not falsify the rule.
    return (
        _shared_stage_norm_pressure_bytes(batch, dim, clusters)
        <= l2_cache_size
    )


def _native_row_owned_region(batch, clusters, dim):
    """Return whether B/K/d lies in the measured native-byte envelope."""
    if batch == 8192:
        return clusters in (16384, 65536, 262144)
    if batch == 32768:
        return clusters == 16384 and dim == 128
    if batch != 16384:
        return False
    if clusters == 16384:
        # This endpoint is evidenced at the public/training d=128 boundary.
        return dim == 128
    return clusters in (65536, 262144)


def _native_row_owned_resources(
    batch, clusters, dim, pitch_bytes, plan
):
    if plan is None or not _native_row_owned_region(
        batch, clusters, dim
    ):
        return False
    if batch in (16384, 32768):
        return True
    return clusters == 16384 or plan[0] <= 128 or pitch_bytes % 32 == 0


def _new_int8_binding(workspace, prepared):
    import torch

    from faiss.contrib.e_means_cute_row_owned import (
        launch_row_owned,
        prepare_row_owned,
        row_owned_plan,
    )
    from faiss.contrib.e_means_cute_kernel import (
        launch_tiled_gemm,
        prepare_tiled_gemm,
    )

    batch = len(workspace)
    row_norms = torch.empty(
        batch, dtype=torch.int32, device=workspace.device
    )
    distances = torch.empty(
        batch, dtype=torch.float32, device=workspace.device
    )
    labels = torch.empty(
        batch, dtype=torch.int64, device=workspace.device
    )
    plan = row_owned_plan(
        batch,
        prepared.num_centroids,
        prepared.dim,
        workspace.device,
        workspace.dtype,
    )
    use_row_owned = _native_row_owned_resources(
        batch,
        prepared.num_centroids,
        prepared.dim,
        workspace.stride(0) * workspace.element_size(),
        plan,
    )
    if use_row_owned:
        core = prepare_row_owned(
            workspace,
            prepared,
            row_norms,
            distances,
            labels,
            plan,
        )
        launches = ((launch_row_owned, core),)
    else:
        keys = torch.full(
            (1, batch, 1),
            0x7FFFFFFFFFFFFFFF,
            dtype=torch.int64,
            device=workspace.device,
        )
        core = prepare_tiled_gemm(
            workspace, prepared, keys=keys,
            row_norms=row_norms, distances=distances, labels=labels,
        )
        launches = ((launch_tiled_gemm, core),)
    return _Binding(
        workspace=workspace,
        codes=None,
        scores=None,
        launches=launches,
        peak=None,
        row_norms=row_norms,
        distances=distances,
        labels=labels,
    )


def _new_binding(batch, prepared):
    import torch

    from faiss.contrib.e_means_cute_kernel import (
        launch_tiled_gemm,
        prepare_tiled_gemm,
    )
    from faiss.contrib.e_means_cute_row_owned import (
        launch_row_owned,
        prepare_row_owned,
        row_owned_plan,
    )

    _check_extent("B", batch)
    _check_padded_span(
        "row", batch, prepared.dim, prepared.centroids.element_size()
    )
    with torch.cuda.device(prepared.centroids.device):
        workspace = _aligned_operand(
            batch,
            prepared.dim,
            prepared.centroids.device,
            prepared.centroids.dtype,
        )
        if workspace.dtype is torch.int8:
            return _new_int8_binding(workspace, prepared)
        scores = torch.empty(
            batch, dtype=torch.float32, device=workspace.device
        )
        is_fp8 = workspace.dtype is torch.float8_e4m3fn
        peak = (
            torch.zeros((), dtype=torch.float32, device=workspace.device)
            if is_fp8
            else None
        )
        row_norms = (
            torch.empty(batch, dtype=torch.float32, device=workspace.device)
            if is_fp8
            else None
        )
        if is_fp8:
            l2_cache_size = getattr(
                torch.cuda.get_device_properties(workspace.device),
                "L2_cache_size",
                None,
            )
            shared_stage_norm_profitable = _shared_stage_norm_profitable(
                batch, prepared.dim, prepared.num_centroids, l2_cache_size
            )
        else:
            shared_stage_norm_profitable = False
        plan = row_owned_plan(
            batch,
            prepared.num_centroids,
            prepared.dim,
            workspace.device,
            workspace.dtype,
        )
        use_row_owned = _float_row_owned_resources(
            workspace.dtype,
            batch,
            prepared.num_centroids,
            workspace.stride(0) * workspace.element_size(),
            plan,
        )
        if use_row_owned:
            labels = torch.empty(
                batch, dtype=torch.int64, device=workspace.device
            )
            codes = None
            kernel = prepare_row_owned(
                workspace,
                prepared,
                None,
                scores,
                labels,
                plan,
                overlap_wgmma=_float_row_owned_overlap(
                    batch, prepared.num_centroids
                ),
            )
            launcher = launch_row_owned
        else:
            labels = None
            codes = torch.empty(
                batch, dtype=torch.int32, device=workspace.device
            )
            kernel = prepare_tiled_gemm(workspace, prepared, codes, scores)
            launcher = launch_tiled_gemm
    return _Binding(
        workspace=workspace,
        codes=codes,
        scores=scores,
        launches=((launcher, kernel),),
        peak=peak,
        row_norms=row_norms,
        labels=labels,
        shared_stage_norm_profitable=shared_stage_norm_profitable,
    )


def _binding_for(batch, prepared):
    binding = prepared.bindings.pop(batch, None)
    if binding is None:
        binding = _new_binding(batch, prepared)
    prepared.bindings[batch] = binding
    while len(prepared.bindings) > _MAX_BINDINGS:
        prepared.bindings.popitem(last=False)
    return binding


def _assign_rows(rows, prepared, validate=False, shared_stage_norm=False):
    """Assign contiguous FP32 or native-byte rows with stable outputs."""
    import torch

    _require_stack()
    if not isinstance(prepared, _PreparedCentroids):
        raise TypeError("prepared centroids have the wrong type")
    if not isinstance(rows, torch.Tensor) or rows.ndim != 2:
        raise ValueError("rows must be a rank-two tensor")
    representation = getattr(prepared.transform, "representation", None)
    if representation is not None:
        _require_int8_stack()
    if representation is None:
        if (
            not rows.is_cuda
            or rows.dtype is not torch.float32
            or not rows.is_contiguous()
        ):
            raise ValueError("rows must be contiguous CUDA FP32")
    else:
        expected_dtype = {
            "int8": torch.int8,
            "uint8": torch.uint8,
        }[representation]
        if (
            not rows.is_cuda
            or rows.dtype is not expected_dtype
            or not rows.is_contiguous()
        ):
            raise ValueError(
                "rows do not match the prepared CuTe representation"
            )
    if rows.device != prepared.centroids.device:
        raise ValueError("rows and centroids must be on the same device")
    if rows.shape[1] != prepared.dim:
        raise ValueError("rows and centroids must have the same dimension")
    if rows.shape[0] < 1:
        raise ValueError("rows must not be empty")
    if shared_stage_norm and (
        validate or representation is not None or prepared.transform is None
    ):
        raise ValueError("shared stage/norm requires prevalidated E4M3 rows")

    with torch.cuda.device(rows.device):
        binding = _binding_for(rows.shape[0], prepared)
        if representation is not None:
            from faiss.contrib.e_means_cute_argmin import stage_int8_rows

            stage_int8_rows(
                rows,
                binding.workspace,
                binding.row_norms,
                representation,
            )
            distances = None
        elif prepared.transform is None:
            binding.workspace.copy_(rows)
            distances = torch.linalg.vector_norm(
                binding.workspace, dim=1, dtype=torch.float32
            )
        else:
            from faiss.contrib.e_means_cute_argmin import (
                affine_cast,
                launch_row_norms,
                prepare_row_norms,
            )
            from faiss.contrib.e_means_fp8 import (
                _check_fp8_range,
                _transform_fp8,
            )

            use_shared_stage_norm = (
                shared_stage_norm and binding.shared_stage_norm_profitable
            )
            if use_shared_stage_norm:
                if binding.prepared_row_norms is None:
                    binding.prepared_row_norms = prepare_row_norms(
                        binding.workspace, binding.row_norms
                    )
                _transform_fp8(
                    rows,
                    prepared.transform._bias,
                    prepared.transform.scale,
                    alignment=16,
                    out=binding.workspace,
                )
                launch_row_norms(binding.prepared_row_norms)
            else:
                if validate:
                    binding.peak.zero_()
                affine_cast(
                    rows,
                    prepared.transform._bias,
                    prepared.transform.scale,
                    binding.workspace,
                    binding.peak if validate else None,
                    packed_rows=True,
                    norms=binding.row_norms,
                )
                if validate:
                    _check_fp8_range(binding.peak, "x")
            distances = binding.row_norms
        for launcher, prepared_launch in binding.launches:
            launcher(prepared_launch)
        if representation is not None:
            return binding.distances.clone(), binding.labels.clone()
        if prepared.transform is None:
            distances.square_()
            distances.add_(binding.scores).clamp_min_(0)
        else:
            # Fused row norms live in the persistent binding, so return a
            # fresh tensor rather than mutating the buffer.
            distances = distances.add(binding.scores).clamp_min_(0)
            distances.mul_(prepared.transform.scale**-2)
        labels = (
            binding.labels.clone()
            if binding.labels is not None
            else binding.codes.to(dtype=torch.int64)
        )
    return distances, labels
