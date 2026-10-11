# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Host adapter for E-means BF16 CuTe assignment on SM90.

Optional CuTe dependencies are resolved only after an explicit ``cute``
backend request. Assignment selects between the tiled kernel and an admitted
low-dimension row-owned kernel; there is no fallback backend.
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


def _check_padded_span(name, rows, dim):
    pitch = _padded_extent(dim, 8)
    if rows * pitch >= 2**31:
        raise ValueError("CuTe padded %s storage must fit signed int32" % name)
    return pitch


def _tensor_version(tensor):
    """Return the mutation version, or disable reuse when unavailable."""
    try:
        return tensor._version
    except (AttributeError, RuntimeError):
        return None


def _aligned_bf16(rows, dim, device):
    import torch

    pitch = _padded_extent(dim, 8)
    if pitch == dim:
        return torch.empty((rows, dim), dtype=torch.bfloat16, device=device)
    return torch.zeros((rows, pitch), dtype=torch.bfloat16, device=device)[
        :, :dim
    ]


@dataclass
class _PreparedCentroids:
    source_ref: object
    source_version: int
    centroids: object
    norms: object
    padded_norms: object
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
    labels: object
    launcher: object
    kernel: object


def _can_refresh(cached, centroids):
    return (
        isinstance(cached, _PreparedCentroids)
        and cached.centroids.device == centroids.device
        and cached.centroids.shape == centroids.shape
    )


def _refresh_prepared(cached, centroids, source_version):
    cached.centroids.copy_(centroids.detach())
    norms = cached.centroids.float().square().sum(dim=1)
    cached.norms.copy_(norms)
    cached.source_ref = weakref.ref(centroids)
    cached.source_version = source_version
    return cached


def _prepare_centroids(centroids, cached=None):
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
    _check_padded_span("centroid", k, dim)
    padded_k = _padded_extent(k, _TILE_N)
    if padded_k >= 2**31:
        raise ValueError("CuTe padded centroid IDs must fit signed int32")
    source_version = _tensor_version(centroids)
    if (
        isinstance(cached, _PreparedCentroids)
        and cached.source_ref() is centroids
        and source_version is not None
        and cached.source_version == source_version
    ):
        return cached

    with torch.cuda.device(centroids.device):
        if _can_refresh(cached, centroids):
            return _refresh_prepared(cached, centroids, source_version)
        mirror = _aligned_bf16(k, dim, centroids.device)
        storage = torch.empty(
            padded_k, dtype=torch.float32, device=centroids.device
        )
        if padded_k != k:
            storage[k:].fill_(float("inf"))
        prepared = _PreparedCentroids(
            None,
            None,
            mirror,
            storage[:k],
            storage.view(1, padded_k),
        )
        return _refresh_prepared(prepared, centroids, source_version)


def _bf16_row_owned_admitted(batch, clusters, pitch_bytes, plan):
    """Apply the measured SM90 BF16 admission law to a legal plan."""
    if plan is None:
        return False
    if batch == 8192:
        return (
            clusters in (16384, 65536, 262144)
            and pitch_bytes % 32 == 0
        )
    return batch == 16384 and clusters <= 16384


def _float_row_owned_overlap(batch, clusters):
    """Return whether an admitted floating plan uses faithful overlap.

    The extra schedule is retained for the measured BF16 K=65536 assignment
    win. It is not needed by the broader serial row-owned gains.
    """
    return batch == 8192 and clusters in (16384, 65536)


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
    _check_padded_span("row", batch, prepared.dim)
    with torch.cuda.device(prepared.centroids.device):
        workspace = _aligned_bf16(
            batch, prepared.dim, prepared.centroids.device
        )
        scores = torch.empty(
            batch, dtype=torch.float32, device=workspace.device
        )
        plan = row_owned_plan(
            batch,
            prepared.num_centroids,
            prepared.dim,
            workspace.device,
        )
        use_row_owned = _bf16_row_owned_admitted(
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
    return _Binding(workspace, codes, scores, labels, launcher, kernel)


def _binding_for(batch, prepared):
    binding = prepared.bindings.pop(batch, None)
    if binding is None:
        binding = _new_binding(batch, prepared)
    prepared.bindings[batch] = binding
    while len(prepared.bindings) > _MAX_BINDINGS:
        prepared.bindings.popitem(last=False)
    return binding


def _assign_rows(rows, prepared):
    """Assign contiguous FP32 rows and return stable FP32/int64 outputs."""
    import torch

    _require_stack()
    if not isinstance(prepared, _PreparedCentroids):
        raise TypeError("prepared centroids have the wrong type")
    if not isinstance(rows, torch.Tensor) or rows.ndim != 2:
        raise ValueError("rows must be a rank-two tensor")
    if (
        not rows.is_cuda
        or rows.dtype is not torch.float32
        or not rows.is_contiguous()
    ):
        raise ValueError("rows must be contiguous CUDA FP32")
    if rows.device != prepared.centroids.device:
        raise ValueError("rows and centroids must be on the same device")
    if rows.shape[1] != prepared.dim:
        raise ValueError("rows and centroids must have the same dimension")
    if rows.shape[0] < 1:
        raise ValueError("rows must not be empty")

    with torch.cuda.device(rows.device):
        binding = _binding_for(rows.shape[0], prepared)
        binding.workspace.copy_(rows)
        distances = torch.linalg.vector_norm(
            binding.workspace, dim=1, dtype=torch.float32
        )
        binding.launcher(binding.kernel)
        distances.square_().add_(binding.scores).clamp_min_(0)
        labels = (
            binding.labels.clone()
            if binding.labels is not None
            else binding.codes.to(dtype=torch.int64)
        )
    return distances, labels
