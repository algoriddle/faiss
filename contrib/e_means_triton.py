# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Triton nearest-centroid assignment for E-means.

Dispatch uses fixed shape and dtype rules without runtime autotuning. Kernels
stream centroid tiles with bounded score workspaces instead of materializing
the native backend's row-by-centroid score tensor.
"""

import math

import torch

try:
    import triton
    import triton.language as tl
except ImportError as error:
    raise ImportError(
        "Triton assignment requires the triton package"
    ) from error


_INT32_MAX = 2**31 - 1
_SPLIT_K_BLOCK_B = 64
_SPLIT_K_BLOCK_K = 128
_SPLIT_K_BLOCK_D = 128
_SPLIT_K_MAX_PARTITIONS = 128
_SPLIT_K_NUM_WARPS = 8
_SPLIT_K_NUM_STAGES = 3
_SPLIT_K_TARGET_PROGRAMS = 512
_SPLIT_K_MAX_B = 4096
_SPLIT_K_MIN_TILES = 64
_SPLIT_K_WIDE_MIN_TILES = 32
_SPLIT_K_WIDE_D = 512
_DIRECT_BLOCK_B = 64
_DIRECT_BLOCK_K = 256
_DIRECT_FP8_BLOCK_D = 128
_DIRECT_FP8_MIN_D = 512
_DIRECT_REDUCED_BLOCK_D = 64
_DIRECT_FP32_BLOCK_D = 32
_FULL_D_MIN_B = 1024
_FULL_D_MAX_D = 128
_FULL_D_CONFIG = (64, 128, 4, 3)
_FLOAT8_E4M3FN = getattr(torch, "float8_e4m3fn", None)
_INT8_BLOCK_B = 64
_INT8_BLOCK_K = 256
_INT8_BLOCK_D = 128
_INT8_PREPARE_BLOCK_D = 256
_INT8_MAX_D = 43919
_INT8_D_ALIGNMENT = 16
_MAX_K = 2**31 - max(
    _SPLIT_K_MAX_PARTITIONS,
    _SPLIT_K_BLOCK_K,
    _DIRECT_BLOCK_K,
)
_int8_libdevice = None


def _load_int8_libdevice():
    """Load the INT8 rounding dependency only when INT8 is selected."""
    global _int8_libdevice
    if _int8_libdevice is None:
        try:
            from triton.language.extra import libdevice
        except ImportError as error:
            raise ImportError(
                "INT8 assignment requires Triton libdevice support"
            ) from error
        if not hasattr(libdevice, "rint"):
            raise ImportError(
                "INT8 assignment requires Triton libdevice.rint support"
            )
        _int8_libdevice = libdevice
    return _int8_libdevice


def _centroid_offset_is_wide(shape, stride):
    """Return whether any physical centroid lane needs int64."""
    physical_k = max(
        triton.cdiv(shape[0], _DIRECT_BLOCK_K) * _DIRECT_BLOCK_K,
        shape[0] + _SPLIT_K_BLOCK_K - 1,
    )
    physical_d = (
        triton.cdiv(shape[1], _SPLIT_K_BLOCK_D) * _SPLIT_K_BLOCK_D
    )
    maximum = (physical_k - 1) * stride[0] + (physical_d - 1) * stride[1]
    return maximum > _INT32_MAX


@triton.jit
def _assign_kernel(
    x_ptr,
    centroid_ptr,
    centroid_norm_ptr,
    distance_ptr,
    label_ptr,
    B,
    K,
    D,
    stride_xb,
    stride_xd,
    stride_ck,
    stride_cd,
    BLOCK_B: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
    FULL_D: tl.constexpr,
    REDUCED: tl.constexpr,
    FP16: tl.constexpr,
    FP8: tl.constexpr,
    WIDE_CENTROID: tl.constexpr,
):
    row_program = tl.program_id(0).to(tl.int64)
    lane = tl.arange(0, BLOCK_B).to(tl.int64)
    row = row_program * BLOCK_B + lane
    row_mask = row < B
    row_norm = tl.zeros((BLOCK_B,), tl.float32)

    if FULL_D:
        column = tl.arange(0, BLOCK_D)
        full_x = tl.load(
            x_ptr
            + row[:, None] * stride_xb
            + column[None, :] * stride_xd,
            mask=row_mask[:, None] & (column[None, :] < D),
            other=0.0,
        )
        if REDUCED and not FP8:
            full_x = (
                full_x.to(tl.float16) if FP16 else full_x.to(tl.bfloat16)
            )
        x_float = full_x.to(tl.float32)
        row_norm = tl.sum(x_float * x_float, axis=1)
    else:
        for d0 in range(0, D, BLOCK_D):
            column = d0 + tl.arange(0, BLOCK_D)
            x = tl.load(
                x_ptr
                + row[:, None] * stride_xb
                + column[None, :] * stride_xd,
                mask=row_mask[:, None] & (column[None, :] < D),
                other=0.0,
            )
            if REDUCED and not FP8:
                x = x.to(tl.float16) if FP16 else x.to(tl.bfloat16)
            x = x.to(tl.float32)
            row_norm += tl.sum(x * x, axis=1)

    best_score = tl.full((BLOCK_B,), float("inf"), tl.float32)
    best_label = tl.zeros((BLOCK_B,), tl.int32)
    for k0 in range(0, K, BLOCK_K):
        centroid = k0 + tl.arange(0, BLOCK_K)
        centroid_mask = centroid < K
        if FULL_D:
            column = tl.arange(0, BLOCK_D)
            column_mask = column < D
            if WIDE_CENTROID:
                centroid_offset = (
                    centroid[:, None].to(tl.int64) * stride_ck
                    + column[None, :].to(tl.int64) * stride_cd
                )
            else:
                centroid_offset = (
                    centroid[:, None] * stride_ck
                    + column[None, :] * stride_cd
                )
            c = tl.load(
                centroid_ptr + centroid_offset,
                mask=centroid_mask[:, None] & column_mask[None, :],
                other=0.0,
            )
            if FP8:
                score = tl.dot(
                    full_x,
                    tl.trans(c),
                    out_dtype=tl.float32,
                    max_num_imprecise_acc=max(32, BLOCK_D),
                )
            else:
                score = tl.dot(
                    full_x,
                    tl.trans(c),
                    out_dtype=tl.float32,
                )
        else:
            score = tl.zeros((BLOCK_B, BLOCK_K), tl.float32)
            for d0 in range(0, D, BLOCK_D):
                column = d0 + tl.arange(0, BLOCK_D)
                column_mask = column < D
                x = tl.load(
                    x_ptr
                    + row[:, None] * stride_xb
                    + column[None, :] * stride_xd,
                    mask=row_mask[:, None] & column_mask[None, :],
                    other=0.0,
                )
                if REDUCED and not FP8:
                    x = x.to(tl.float16) if FP16 else x.to(tl.bfloat16)
                if WIDE_CENTROID:
                    centroid_offset = (
                        centroid[:, None].to(tl.int64) * stride_ck
                        + column[None, :].to(tl.int64) * stride_cd
                    )
                else:
                    centroid_offset = (
                        centroid[:, None] * stride_ck
                        + column[None, :] * stride_cd
                    )
                c = tl.load(
                    centroid_ptr + centroid_offset,
                    mask=centroid_mask[:, None] & column_mask[None, :],
                    other=0.0,
                )
                if FP8:
                    score = tl.dot(
                        x,
                        tl.trans(c),
                        acc=score,
                        out_dtype=tl.float32,
                        max_num_imprecise_acc=max(32, BLOCK_D),
                    )
                elif REDUCED:
                    score = tl.dot(
                        x,
                        tl.trans(c),
                        acc=score,
                        out_dtype=tl.float32,
                    )
                else:
                    score = tl.dot(
                        x,
                        tl.trans(c),
                        acc=score,
                        input_precision="ieee",
                        out_dtype=tl.float32,
                    )

        norm = tl.load(
            centroid_norm_ptr + centroid,
            mask=centroid_mask,
            other=float("inf"),
        )
        score = norm[None, :] - 2.0 * score
        block_score, block_label = tl.min(
            score,
            axis=1,
            return_indices=True,
            return_indices_tie_break_left=True,
        )
        block_label = block_label.to(tl.int32) + k0
        better = block_score < best_score
        best_score = tl.where(better, block_score, best_score)
        best_label = tl.where(better, block_label, best_label)

    distance = tl.maximum(best_score + row_norm, 0.0)
    tl.store(distance_ptr + row, distance, mask=row_mask)
    tl.store(label_ptr + row, best_label.to(tl.int64), mask=row_mask)


@triton.jit
def _assign_split_k_kernel(
    x_ptr,
    centroid_ptr,
    centroid_norm_ptr,
    partial_score_ptr,
    partial_label_ptr,
    B,
    K,
    D,
    stride_xb,
    stride_xd,
    stride_ck,
    stride_cd,
    PARTITIONS: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
    FP16: tl.constexpr,
    FP8: tl.constexpr,
    WIDE_CENTROID: tl.constexpr,
):
    row_program = tl.program_id(0).to(tl.int64)
    partition = tl.program_id(1)
    lane = tl.arange(0, BLOCK_B).to(tl.int64)
    row = row_program * BLOCK_B + lane
    row_mask = row < B
    partition_size = (K - 1) // PARTITIONS + 1
    centroid_start = partition * partition_size
    centroid_end = tl.minimum(centroid_start + partition_size, K)
    best_score = tl.full((BLOCK_B,), float("inf"), tl.float32)
    best_label = tl.zeros((BLOCK_B,), tl.int32)

    k0 = centroid_start
    while k0 < centroid_end:
        centroid = k0 + tl.arange(0, BLOCK_K)
        centroid_mask = centroid < centroid_end
        score = tl.zeros((BLOCK_B, BLOCK_K), tl.float32)

        for d0 in range(0, D, BLOCK_D):
            column = d0 + tl.arange(0, BLOCK_D)
            column_mask = column < D
            x = tl.load(
                x_ptr
                + row[:, None] * stride_xb
                + column[None, :] * stride_xd,
                mask=row_mask[:, None] & column_mask[None, :],
                other=0.0,
            )
            if not FP8:
                x = x.to(tl.float16) if FP16 else x.to(tl.bfloat16)
            if WIDE_CENTROID:
                centroid_offset = (
                    centroid[:, None].to(tl.int64) * stride_ck
                    + column[None, :].to(tl.int64) * stride_cd
                )
            else:
                centroid_offset = (
                    centroid[:, None] * stride_ck
                    + column[None, :] * stride_cd
                )
            c = tl.load(
                centroid_ptr + centroid_offset,
                mask=centroid_mask[:, None] & column_mask[None, :],
                other=0.0,
            )
            if FP8:
                score = tl.dot(
                    x,
                    tl.trans(c),
                    acc=score,
                    out_dtype=tl.float32,
                    max_num_imprecise_acc=max(32, BLOCK_D),
                )
            else:
                score = tl.dot(
                    x,
                    tl.trans(c),
                    acc=score,
                    out_dtype=tl.float32,
                )

        norm = tl.load(
            centroid_norm_ptr + centroid,
            mask=centroid_mask,
            other=float("inf"),
        )
        score = norm[None, :] - 2.0 * score
        block_score, block_label = tl.min(
            score,
            axis=1,
            return_indices=True,
            return_indices_tie_break_left=True,
        )
        block_label = block_label.to(tl.int32) + k0
        better = block_score < best_score
        best_score = tl.where(better, block_score, best_score)
        best_label = tl.where(better, block_label, best_label)
        k0 += BLOCK_K

    offset = row * PARTITIONS + partition
    tl.store(partial_score_ptr + offset, best_score, mask=row_mask)
    tl.store(partial_label_ptr + offset, best_label, mask=row_mask)


@triton.jit
def _reduce_split_k_kernel(
    x_ptr,
    partial_score_ptr,
    partial_label_ptr,
    distance_ptr,
    label_ptr,
    B,
    D,
    stride_xb,
    stride_xd,
    PARTITIONS: tl.constexpr,
    BLOCK_PARTITIONS: tl.constexpr,
    BLOCK_D: tl.constexpr,
    FP16: tl.constexpr,
    FP8: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    row_norm = tl.zeros((), tl.float32)
    for d0 in range(0, D, BLOCK_D):
        column = d0 + tl.arange(0, BLOCK_D)
        x = tl.load(
            x_ptr + row * stride_xb + column * stride_xd,
            mask=column < D,
            other=0.0,
        )
        if not FP8:
            x = x.to(tl.float16) if FP16 else x.to(tl.bfloat16)
        x = x.to(tl.float32)
        row_norm += tl.sum(x * x, axis=0)

    partition = tl.arange(0, BLOCK_PARTITIONS)
    mask = partition < PARTITIONS
    offset = row * PARTITIONS + partition
    score = tl.load(
        partial_score_ptr + offset,
        mask=mask,
        other=float("inf"),
    )
    # Partitions are in centroid order, so a left tie is the lowest label.
    best_score, best_partition = tl.min(
        score,
        axis=0,
        return_indices=True,
        return_indices_tie_break_left=True,
    )
    partial_label = tl.load(partial_label_ptr + offset, mask=mask, other=0)
    best_label = tl.sum(
        tl.where(partition == best_partition, partial_label, 0), axis=0
    )
    tl.store(distance_ptr + row, tl.maximum(best_score + row_norm, 0.0))
    tl.store(label_ptr + row, best_label.to(tl.int64))


@triton.jit
def _stage_int8_rows_kernel(
    x_ptr,
    prepared_ptr,
    D,
    stride_xb,
    stride_xd,
    stride_pb,
    UNSIGNED: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    for d0 in range(0, D, BLOCK_D):
        column = d0 + tl.arange(0, BLOCK_D)
        mask = column < D
        value = tl.load(
            x_ptr + row * stride_xb + column * stride_xd,
            mask=mask,
            other=0,
        )
        if UNSIGNED:
            value = value.to(tl.int16) - 128
        value = tl.where(mask, value, 0).to(tl.int8)
        tl.store(
            prepared_ptr + row * stride_pb + column,
            value,
            mask=column < stride_pb,
        )


@triton.jit
def _prepare_int8_kernel(
    centroid_ptr,
    prepared_ptr,
    norm_ptr,
    D,
    stride_ck,
    stride_cd,
    stride_pk,
    UNSIGNED: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    centroid = tl.program_id(0).to(tl.int64)
    norm = tl.zeros((), tl.int32)
    for d0 in range(0, D, BLOCK_D):
        column = d0 + tl.arange(0, BLOCK_D)
        mask = column < D
        value = tl.load(
            centroid_ptr + centroid * stride_ck + column * stride_cd,
            mask=mask,
            other=0.0,
        )
        if UNSIGNED:
            value = _int8_libdevice.rint(value - 128.0)
        else:
            value = _int8_libdevice.rint(value)
        value = tl.maximum(-128.0, tl.minimum(127.0, value))
        value = tl.where(mask, value, 0).to(tl.int8)
        tl.store(
            prepared_ptr + centroid * stride_pk + column,
            value,
            mask=column < stride_pk,
        )
        value = value.to(tl.int32)
        norm += tl.sum(value * value, axis=0)
    tl.store(norm_ptr + centroid, norm)


@triton.jit
def _assign_int8_kernel(
    x_ptr,
    centroid_ptr,
    centroid_norm_ptr,
    distance_ptr,
    label_ptr,
    B,
    K,
    D,
    stride_xb,
    stride_xd,
    stride_ck,
    stride_cd,
    UNSIGNED: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
    WIDE_CENTROID: tl.constexpr,
):
    row_program = tl.program_id(0).to(tl.int64)
    lane = tl.arange(0, BLOCK_B).to(tl.int64)
    row = row_program * BLOCK_B + lane
    row_mask = row < B
    row_norm = tl.zeros((BLOCK_B,), tl.int32)

    for d0 in range(0, D, BLOCK_D):
        column = d0 + tl.arange(0, BLOCK_D)
        column_mask = column < D
        value = tl.load(
            x_ptr
            + row[:, None] * stride_xb
            + column[None, :] * stride_xd,
            mask=row_mask[:, None] & column_mask[None, :],
            other=0,
        )
        if UNSIGNED:
            value = value.to(tl.int16) - 128
        value = tl.where(column_mask[None, :], value, 0).to(tl.int8)
        value = value.to(tl.int32)
        row_norm += tl.sum(value * value, axis=1)

    best_score = tl.full((BLOCK_B,), 2147483647, tl.int32)
    best_label = tl.zeros((BLOCK_B,), tl.int32)
    for k0 in range(0, K, BLOCK_K):
        centroid = k0 + tl.arange(0, BLOCK_K)
        centroid_mask = centroid < K
        score = tl.zeros((BLOCK_B, BLOCK_K), tl.int32)
        for d0 in range(0, D, BLOCK_D):
            column = d0 + tl.arange(0, BLOCK_D)
            column_mask = column < D
            value = tl.load(
                x_ptr
                + row[:, None] * stride_xb
                + column[None, :] * stride_xd,
                mask=row_mask[:, None] & column_mask[None, :],
                other=0,
            )
            if UNSIGNED:
                value = value.to(tl.int16) - 128
            value = tl.where(column_mask[None, :], value, 0).to(tl.int8)
            if WIDE_CENTROID:
                centroid_offset = (
                    centroid[:, None].to(tl.int64) * stride_ck
                    + column[None, :].to(tl.int64) * stride_cd
                )
            else:
                centroid_offset = (
                    centroid[:, None] * stride_ck
                    + column[None, :] * stride_cd
                )
            c = tl.load(
                centroid_ptr + centroid_offset,
                mask=centroid_mask[:, None] & column_mask[None, :],
                other=0,
            )
            score = tl.dot(
                value,
                tl.trans(c),
                acc=score,
                out_dtype=tl.int32,
            )

        norm = tl.load(
            centroid_norm_ptr + centroid,
            mask=centroid_mask,
            other=2147483647,
        )
        score = norm[None, :] - 2 * score
        block_score, block_label = tl.min(
            score,
            axis=1,
            return_indices=True,
            return_indices_tie_break_left=True,
        )
        block_label = block_label.to(tl.int32) + k0
        better = block_score < best_score
        best_score = tl.where(better, block_score, best_score)
        best_label = tl.where(better, block_label, best_label)

    distance = best_score.to(tl.int64) + row_norm.to(tl.int64)
    distance = tl.maximum(distance, 0).to(tl.float32)
    tl.store(distance_ptr + row, distance, mask=row_mask)
    tl.store(label_ptr + row, best_label.to(tl.int64), mask=row_mask)


def _use_split_k(b, k, d, is_reduced):
    """Use split-K in regions where it adds parallelism."""
    k_tiles = triton.cdiv(k, _SPLIT_K_BLOCK_K)
    return (
        is_reduced
        and b <= _SPLIT_K_MAX_B
        and (
            k_tiles >= _SPLIT_K_MIN_TILES
            or (
                d >= _SPLIT_K_WIDE_D
                and k_tiles >= _SPLIT_K_WIDE_MIN_TILES
            )
        )
    )


def _split_k_partitions(b, k):
    """Return enough partitions for about four waves, up to one per K tile."""
    k_blocks = triton.cdiv(k, _SPLIT_K_BLOCK_K)
    b_blocks = triton.cdiv(b, _SPLIT_K_BLOCK_B)
    partitions = 1
    while (
        partitions < min(k_blocks, _SPLIT_K_MAX_PARTITIONS)
        and b_blocks * partitions < _SPLIT_K_TARGET_PROGRAMS
    ):
        partitions *= 2
    return min(partitions, k_blocks)


def _launch_assignment(
    x, centroids, centroid_norm, distance, labels, wide_centroid
):
    b, d = x.shape
    k = centroids.shape[0]
    is_reduced = centroids.dtype != torch.float32
    is_fp16 = centroids.dtype == torch.float16
    is_fp8 = centroids.dtype == _FLOAT8_E4M3FN
    split_k = _use_split_k(b, k, d, is_reduced)
    if not split_k:
        full_d = (
            is_reduced
            and b >= _FULL_D_MIN_B
            and _DIRECT_REDUCED_BLOCK_D < d <= _FULL_D_MAX_D
        )
        if full_d:
            block_b, block_k, num_warps, num_stages = _FULL_D_CONFIG
            block_d = max(16, triton.next_power_of_2(d))
        else:
            block_b = _DIRECT_BLOCK_B
            block_k = _DIRECT_BLOCK_K
            if is_fp8 and d >= _DIRECT_FP8_MIN_D:
                block_d = _DIRECT_FP8_BLOCK_D
            elif is_reduced:
                block_d = _DIRECT_REDUCED_BLOCK_D
            else:
                block_d = _DIRECT_FP32_BLOCK_D
            num_warps = 8
            num_stages = 3
        _assign_kernel[(triton.cdiv(b, block_b),)](
            x,
            centroids,
            centroid_norm,
            distance,
            labels,
            b,
            k,
            d,
            x.stride(0),
            x.stride(1),
            centroids.stride(0),
            centroids.stride(1),
            BLOCK_B=block_b,
            BLOCK_K=block_k,
            BLOCK_D=block_d,
            FULL_D=full_d,
            REDUCED=is_reduced,
            FP16=is_fp16,
            FP8=is_fp8,
            WIDE_CENTROID=wide_centroid,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        return

    partitions = _split_k_partitions(b, k)
    partial_score = torch.empty(
        (b, partitions), dtype=torch.float32, device=x.device
    )
    partial_label = torch.empty(
        (b, partitions), dtype=torch.int32, device=x.device
    )
    _assign_split_k_kernel[
        (triton.cdiv(b, _SPLIT_K_BLOCK_B), partitions)
    ](
        x,
        centroids,
        centroid_norm,
        partial_score,
        partial_label,
        b,
        k,
        d,
        x.stride(0),
        x.stride(1),
        centroids.stride(0),
        centroids.stride(1),
        PARTITIONS=partitions,
        BLOCK_B=_SPLIT_K_BLOCK_B,
        BLOCK_K=_SPLIT_K_BLOCK_K,
        BLOCK_D=_SPLIT_K_BLOCK_D,
        FP16=is_fp16,
        FP8=is_fp8,
        WIDE_CENTROID=wide_centroid,
        num_warps=_SPLIT_K_NUM_WARPS,
        num_stages=_SPLIT_K_NUM_STAGES,
    )
    _reduce_split_k_kernel[(b,)](
        x,
        partial_score,
        partial_label,
        distance,
        labels,
        b,
        d,
        x.stride(0),
        x.stride(1),
        PARTITIONS=partitions,
        BLOCK_PARTITIONS=triton.next_power_of_2(partitions),
        BLOCK_D=128,
        FP16=is_fp16,
        FP8=is_fp8,
        num_warps=4,
    )


def _transform_fp8(x, bias, scale, validate=False, peak_out=None):
    """Return the E4M3 mirror under one shared affine transform."""
    from faiss.contrib.e_means_fp8 import (
        _check_fp8_range,
        _check_fp8_support,
    )

    _check_fp8_support(x.device)
    if x.dtype != torch.float32:
        raise TypeError("FP8 assignment preparation requires float32 input")
    if scale is None:
        raise ValueError("FP8 assignment requires a scale")
    valid = math.isfinite(scale) and scale > 0.0
    if not valid or not math.log2(scale).is_integer():
        raise ValueError("FP8 assignment scale must be a power of two")
    if scale > torch.finfo(torch.float32).max:
        raise ValueError("FP8 assignment scale exceeds the FP32 range")
    if bias is not None and (
        bias.device != x.device
        or bias.dtype != torch.float32
        or bias.shape != (x.shape[1],)
    ):
        raise ValueError("FP8 assignment bias does not match the input")
    transformed = (
        x * scale if bias is None else torch.add(bias, x, alpha=scale)
    )
    if validate:
        peak = transformed.detach().abs().max()
        if peak_out is None:
            _check_fp8_range(peak, "centroids")
        else:
            if (
                peak_out.device != x.device
                or peak_out.layout != torch.strided
                or peak_out.shape != ()
                or peak_out.dtype != torch.float32
                or peak_out.requires_grad
            ):
                raise ValueError("FP8 peak accumulator does not match input")
            peak_out.copy_(torch.maximum(peak_out, peak))
    d = x.shape[1]
    physical_d = (d + 63) // 64 * 64
    if physical_d == d:
        return transformed.to(_FLOAT8_E4M3FN)
    padded = torch.zeros(
        (len(x), physical_d), dtype=_FLOAT8_E4M3FN, device=x.device
    )
    padded[:, :d].copy_(transformed)
    return padded


def prepare_fp8_centroids(centroids, bias, scale, peak_out=None):
    """Return transformed E4M3 centroids and their matched FP32 norms."""
    prepared = _transform_fp8(
        centroids, bias, scale, validate=True, peak_out=peak_out
    )
    centroid_norm = prepared.float().square().sum(dim=1)
    return prepared, centroid_norm


def _int8_unsigned(source):
    if source == "int8":
        return False
    if source == "uint8":
        return True
    raise ValueError("unsupported INT8 representation %r" % source)


def _int8_physical_d(d):
    return (
        (d + _INT8_D_ALIGNMENT - 1)
        // _INT8_D_ALIGNMENT
        * _INT8_D_ALIGNMENT
    )


def _check_int8_source(x, source):
    from faiss.contrib.e_means_int8 import _check_int8_support

    _check_int8_support(x.device)
    return _int8_unsigned(source)


def prepare_int8_centroids(centroids, representation, *, validate=True):
    """Return signed-byte centroids and their matched INT32 norms."""
    unsigned = _check_int8_source(centroids, representation)
    _load_int8_libdevice()
    if centroids.dtype != torch.float32 or centroids.ndim != 2:
        raise TypeError("INT8 centroid preparation requires an FP32 matrix")
    if validate and not bool(torch.isfinite(centroids).all()):
        raise ValueError("INT8 centroids must be finite")
    k, d = centroids.shape
    if d > _INT8_MAX_D:
        raise ValueError("INT8 assignment requires d <= 43919")
    with torch.cuda.device(centroids.device):
        physical_d = _int8_physical_d(d)
        prepared = torch.empty(
            (k, physical_d), dtype=torch.int8, device=centroids.device
        )
        norm = torch.empty(k, dtype=torch.int32, device=centroids.device)
        _prepare_int8_kernel[(k,)](
            centroids,
            prepared,
            norm,
            d,
            centroids.stride(0),
            centroids.stride(1),
            prepared.stride(0),
            UNSIGNED=unsigned,
            BLOCK_D=_INT8_PREPARE_BLOCK_D,
            num_warps=4,
        )
    return prepared, norm


def assign_int8_rows(
    x,
    centroids,
    centroid_norm,
    representation,
):
    """Return exact-ranking INT8 distances and centroid indices."""
    if (
        x.device.type != "cuda"
        or centroids.device != x.device
        or centroid_norm.device != x.device
    ):
        raise ValueError(
            "Triton assignment requires tensors on one CUDA device"
        )
    unsigned = _check_int8_source(x, representation)
    expected_dtype = {
        "int8": (torch.int8,),
        "uint8": (torch.uint8,),
    }[representation]
    if x.dtype not in expected_dtype or centroids.dtype != torch.int8:
        raise TypeError(
            "INT8 assignment input does not match its representation"
        )
    if centroid_norm.dtype != torch.int32:
        raise TypeError("INT8 centroid norms must be int32")
    if x.ndim != 2 or centroids.ndim != 2 or centroid_norm.ndim != 1:
        raise ValueError("INT8 assignment expects matrices and a norm vector")
    b, d = x.shape
    k, centroid_d = centroids.shape
    if d > _INT8_MAX_D:
        raise ValueError("INT8 assignment requires d <= 43919")
    if k > _MAX_K:
        raise ValueError("Triton assignment requires k <= 2**31 - 256")
    physical_d = _int8_physical_d(d)
    if physical_d != centroid_d or centroid_norm.shape[0] != k:
        raise ValueError("INT8 assignment shapes do not match")
    if not centroid_norm.is_contiguous():
        raise ValueError("centroid_norm must be contiguous")

    with torch.cuda.device(x.device):
        if physical_d != d:
            staged = torch.empty(
                (b, physical_d), dtype=torch.int8, device=x.device
            )
            _stage_int8_rows_kernel[(b,)](
                x,
                staged,
                d,
                x.stride(0),
                x.stride(1),
                staged.stride(0),
                UNSIGNED=unsigned,
                BLOCK_D=_INT8_PREPARE_BLOCK_D,
                num_warps=4,
            )
            x = staged
            unsigned = False
        distance = torch.empty(b, dtype=torch.float32, device=x.device)
        labels = torch.empty(b, dtype=torch.int64, device=x.device)
        wide_centroid = _centroid_offset_is_wide(
            centroids.shape, centroids.stride()
        )
        _assign_int8_kernel[(triton.cdiv(b, _INT8_BLOCK_B),)](
            x,
            centroids,
            centroid_norm,
            distance,
            labels,
            b,
            k,
            physical_d,
            x.stride(0),
            x.stride(1),
            centroids.stride(0),
            centroids.stride(1),
            UNSIGNED=unsigned,
            BLOCK_B=_INT8_BLOCK_B,
            BLOCK_K=_INT8_BLOCK_K,
            BLOCK_D=_INT8_BLOCK_D,
            WIDE_CENTROID=wide_centroid,
            num_warps=8,
            num_stages=3,
        )
    return distance, labels


def assign_rows(x, centroids, centroid_norm, bias=None, scale=None):
    """Return nearest-centroid squared distances and indices."""
    if (
        x.device.type != "cuda"
        or centroids.device != x.device
        or centroid_norm.device != x.device
    ):
        raise ValueError(
            "Triton assignment requires tensors on one CUDA device"
        )
    is_fp8 = centroids.dtype == _FLOAT8_E4M3FN
    supported_dtype = centroids.dtype in (
        torch.float32,
        torch.float16,
        torch.bfloat16,
    ) or is_fp8
    if x.dtype != torch.float32 or not supported_dtype:
        raise TypeError(
            "Triton assignment requires FP32 rows and FP32, FP16, BF16, "
            "or E4M3 centroids"
        )
    if not is_fp8 and (bias is not None or scale is not None):
        raise ValueError("assignment transform is only valid for FP8")
    if centroid_norm.dtype != torch.float32:
        raise TypeError("centroid_norm must be float32")
    if x.ndim != 2 or centroids.ndim != 2 or centroid_norm.ndim != 1:
        raise ValueError("Triton assignment expects matrices and a norm vector")
    b, d = x.shape
    k, centroid_d = centroids.shape
    if k > _MAX_K:
        raise ValueError("Triton assignment requires k <= 2**31 - 256")
    if not centroid_norm.is_contiguous():
        raise ValueError("centroid_norm must be contiguous")
    expected_d = (d + 63) // 64 * 64 if is_fp8 else d
    if expected_d != centroid_d or centroid_norm.shape[0] != k:
        raise ValueError("Triton assignment shapes do not match")
    with torch.cuda.device(x.device):
        assignment_x = _transform_fp8(x, bias, scale) if is_fp8 else x
        distance = torch.empty(b, dtype=torch.float32, device=x.device)
        labels = torch.empty(b, dtype=torch.int64, device=x.device)
        wide_centroid = _centroid_offset_is_wide(
            centroids.shape, centroids.stride()
        )
        _launch_assignment(
            assignment_x,
            centroids,
            centroid_norm,
            distance,
            labels,
            wide_centroid,
        )
        if is_fp8:
            distance.mul_(scale**-2)
    return distance, labels
