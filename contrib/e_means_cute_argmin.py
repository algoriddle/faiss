# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Packed argmin, native-byte staging, and output decode for CuTe."""

from collections import OrderedDict
import math
from typing import NamedTuple

import cuda.bindings.driver as cuda
import cutlass.cute as cute
from cutlass import (
    Float8E4M3FN, Float32, Int8, Int32, Int64, Uint32, const_expr,
)
from cutlass.cute.runtime import from_dlpack
from quack import utils as quack_utils
import torch

_THREADS = 256
INITIAL_KEY = 0x7F80000000000000
_COMPILED = OrderedDict()
_AFFINE_PREPARE_COMPILED = OrderedDict()
_STAGE_ROW_NORMS_COMPILED = OrderedDict()
_INT8_STAGE_COMPILED = OrderedDict()
_PACKED_ROWS = 8
INTEGER_INITIAL_KEY = 0x7FFFFFFFFFFFFFFF


def require_tensor(
    tensor,
    name,
    *,
    dtype=None,
    shape=None,
    device=None,
    contiguous=True,
):
    """Validate one tensor passed to a raw-pointer CuTe binding."""
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("%s must be a tensor" % name)
    if not tensor.is_cuda:
        raise ValueError("%s must be a CUDA tensor" % name)
    if dtype is not None and tensor.dtype is not dtype:
        raise ValueError(
            "%s must be %s" % (name, str(dtype).split(".")[-1])
        )
    if shape is not None and tuple(tensor.shape) != tuple(shape):
        raise ValueError("%s must have shape %s" % (name, tuple(shape)))
    if device is not None and tensor.device != device:
        raise ValueError("%s must be on %s" % (name, device))
    if contiguous and not tensor.is_contiguous():
        raise ValueError("%s must be contiguous" % name)
    return tensor


@cute.jit
def encode_argmin_key(score: Float32, code: Int32) -> Int64:
    """Encode native FP32 ordering, canonical signed zero, then lower ID."""
    score = Float32(0.0) if score == 0.0 else score
    bits = score.bitcast(Uint32)
    ordered = ~bits if (bits >> 31) != 0 else bits ^ Uint32(0x80000000)
    return (
        (ordered.to(Int64) - Int64(0x80000000)) * Int64(0x100000000)
        + code.to(Int64)
    )


@cute.jit
def decode_argmin_score(key: Int64) -> Float32:
    """Invert the ordered FP32 word in a packed argmin key."""
    signed_order = (key >> Int64(32)).to(Int32)
    ordered = (signed_order.to(Int64) + Int64(0x80000000)).to(Uint32)
    score_bits = (
        ordered ^ Uint32(0x80000000)
        if (ordered >> 31) != 0
        else ~ordered
    )
    return score_bits.bitcast(Float32)


@torch.no_grad()
def stage_int8_rows(x, out, norms, representation):
    """Stage native-byte rows into an INT8 mirror with INT32 norms."""
    if representation not in ("int8", "uint8"):
        raise ValueError("unknown native-byte representation")
    expected = torch.int8 if representation == "int8" else torch.uint8
    require_tensor(
        x, "native-byte rows", dtype=expected, contiguous=False
    )
    if x.ndim != 2:
        raise ValueError("native-byte rows must be rank two")
    require_tensor(
        out,
        "signed row mirror",
        dtype=torch.int8,
        shape=x.shape,
        device=x.device,
        contiguous=False,
    )
    require_tensor(
        norms,
        "row norms",
        dtype=torch.int32,
        shape=(len(x),),
        device=x.device,
    )
    if x.stride(1) != 1 or out.stride(1) != 1:
        raise ValueError("native-byte rows and mirror need unit inner stride")
    input_pitch = x.stride(0)
    output_pitch = out.stride(0)
    if input_pitch < x.shape[1] or output_pitch < x.shape[1]:
        raise ValueError("native-byte row pitches must cover the dimension")
    if output_pitch % 16 or out.data_ptr() % 16:
        raise ValueError("signed row mirror must retain 16-byte alignment")
    input_storage = x.as_strided((len(x), input_pitch), (input_pitch, 1))
    output_storage = out.as_strided((len(x), output_pitch), (output_pitch, 1))
    with torch.cuda.device(x.device):
        x_cute = from_dlpack(input_storage.reshape(-1), assumed_align=1)
        out_cute = from_dlpack(output_storage.reshape(-1), assumed_align=16)
        norms_cute = from_dlpack(norms, assumed_align=4)
        stream = cuda.CUstream(torch.cuda.current_stream(x.device).cuda_stream)
        key = (
            _PACKED_ROWS,
            tuple(x.shape),
            input_pitch,
            output_pitch,
            representation,
            x.device.index,
        )
        compiled = _INT8_STAGE_COMPILED.pop(key, None)
        if compiled is None:
            compiled = cute.compile(
                _StageRowNormsSm90(
                    integer=True,
                    unsigned=representation == "uint8",
                ),
                x_cute,
                out_cute,
                norms_cute,
                Int64(x.shape[1]),
                Int64(input_pitch),
                Int64(output_pitch),
                stream,
            )
        _INT8_STAGE_COMPILED[key] = compiled
        while len(_INT8_STAGE_COMPILED) > 4:
            _INT8_STAGE_COMPILED.popitem(last=False)
        compiled(
            x_cute, out_cute, norms_cute, Int64(x.shape[1]),
            Int64(input_pitch), Int64(output_pitch), stream,
        )


class _AffineCastSm90:
    """Store an E4M3 mirror while reducing its transformed peak."""

    def __init__(
        self, centered, rows_per_block=1, with_peak=True, with_norms=False
    ):
        self.centered = centered
        self.rows_per_block = rows_per_block
        self.with_peak = with_peak
        self.with_norms = with_norms

    @cute.jit
    def __call__(
        self, x, bias, out, peak, norms, scale, dim, pitch, stream
    ):
        rows = x.shape[0] // dim
        self.kernel(x, bias, out, peak, norms, scale, dim, pitch).launch(
            grid=(
                (rows + self.rows_per_block - 1) // self.rows_per_block,
                1,
                1,
            ),
            block=(32 * self.rows_per_block, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        x: cute.Tensor,
        bias: cute.Tensor,
        out: cute.Tensor,
        peak: cute.Tensor,
        norms: cute.Tensor,
        scale: Float32,
        dim: Int64,
        pitch: Int64,
    ):
        thread, _, _ = cute.arch.thread_idx()
        block, _, _ = cute.arch.block_idx()
        thread = thread.to(Int64)
        lane = thread % Int64(32)
        row = (
            block.to(Int64) * Int64(self.rows_per_block)
            + thread // Int64(32)
        )
        rows = x.shape[0] // dim
        column = lane
        local_peak = Float32(0.0)
        total = Float32(0.0)
        if row < rows:
            while column < pitch:
                output_index = row * pitch + column
                if column < dim:
                    value = x[row * dim + column] * scale
                    if const_expr(self.centered):
                        value += bias[column]
                    if const_expr(self.with_peak):
                        absolute = -value if value < 0.0 else value
                        absolute = Float32.inf if value != value else absolute
                        local_peak = (
                            absolute if absolute > local_peak else local_peak
                        )
                    rounded = value.to(Float8E4M3FN)
                    out[output_index] = rounded
                    if const_expr(self.with_norms):
                        stored = rounded.to(Float32)
                        total += stored * stored
                else:
                    out[output_index] = Float32(0.0).to(Float8E4M3FN)
                column += Int64(32)
        if const_expr(self.with_peak):
            for offset in (16, 8, 4, 2, 1):
                other_peak = cute.arch.shuffle_sync_bfly(
                    local_peak, offset=offset
                )
                local_peak = (
                    other_peak if other_peak > local_peak else local_peak
                )
        if const_expr(self.with_norms):
            for offset in (16, 8, 4, 2, 1):
                total += cute.arch.shuffle_sync_bfly(total, offset=offset)
            if row < rows and lane == 0:
                norms[row] = total
        if const_expr(self.with_peak):
            if lane == 0:
                cute.arch.atomic_fmax(
                    quack_utils.elem_pointer(peak, 0).llvm_ptr,
                    local_peak,
                    sem="relaxed",
                    scope="gpu",
                )


class _PeakOnlyAffineCastSm90(_AffineCastSm90):
    """Expose the shared affine cast through the narrow peak-only ABI."""

    def __init__(self, centered, rows_per_block=1):
        super().__init__(centered, rows_per_block, True, False)

    @cute.jit
    def __call__(self, x, bias, out, peak, scale, dim, pitch, stream):
        # Binding peak twice omits the unused norms descriptor from the host
        # ABI; this specialization never writes through norms.
        super().__call__(
            x, bias, out, peak, peak, scale, dim, pitch, stream
        )


class _StageRowNormsSm90:
    """Stage optional signed bytes and reduce exact ordered row norms."""

    def __init__(
        self,
        integer=False,
        unsigned=False,
        rows_per_block=_PACKED_ROWS,
    ):
        self.integer = integer
        self.unsigned = unsigned
        self.rows_per_block = rows_per_block

    @cute.jit
    def __call__(
        self, x, out, norms, dim, input_pitch, output_pitch, stream
    ):
        rows = norms.shape[0]
        self.kernel(
            x, out, norms, dim, input_pitch, output_pitch
        ).launch(
            grid=((rows + self.rows_per_block - 1) // self.rows_per_block,
                  1, 1),
            block=(32 * self.rows_per_block, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        x: cute.Tensor,
        out: cute.Tensor,
        norms: cute.Tensor,
        dim: Int64,
        input_pitch: Int64,
        output_pitch: Int64,
    ):
        thread, _, _ = cute.arch.thread_idx()
        block, _, _ = cute.arch.block_idx()
        thread = thread.to(Int64)
        lane = thread % Int64(32)
        row = (block.to(Int64) * Int64(self.rows_per_block)
               + thread // Int64(32))
        rows = norms.shape[0]
        column = lane
        if const_expr(self.integer):
            total = Int32(0)
            if row < rows:
                while column < output_pitch:
                    output_index = row * output_pitch + column
                    if column < dim:
                        value = x[
                            row * input_pitch + column
                        ].to(Int32)
                        if const_expr(self.unsigned):
                            value -= Int32(128)
                        out[output_index] = value.to(Int8)
                        total += value * value
                    else:
                        out[output_index] = Int32(0).to(Int8)
                    column += Int64(32)
        else:
            total = Float32(0.0)
            if row < rows:
                while column < dim:
                    stored = x[
                        row * input_pitch + column
                    ].to(Float32)
                    total += stored * stored
                    column += Int64(32)
        for offset in (16, 8, 4, 2, 1):
            total += cute.arch.shuffle_sync_bfly(total, offset=offset)
        if row < rows and lane == 0:
            norms[row] = total


class PreparedRowNorms(NamedTuple):
    compiled: object
    args: tuple
    mirror: torch.Tensor
    norms: torch.Tensor


@torch.no_grad()
def prepare_row_norms(mirror, norms):
    """Bind one persistent E4M3 mirror and its exact FP32 row norms."""
    require_tensor(mirror, "E4M3 mirror", dtype=torch.float8_e4m3fn,
                   contiguous=False)
    if mirror.ndim != 2 or mirror.stride(1) != 1:
        raise ValueError("E4M3 mirror must be a row-major matrix")
    require_tensor(norms, "row norms", dtype=torch.float32,
                   shape=(len(mirror),), device=mirror.device)
    dim = mirror.shape[1]
    pitch = mirror.stride(0)
    if pitch < dim:
        raise ValueError("E4M3 mirror pitch must cover its dimension")
    storage = mirror.as_strided((len(mirror), pitch), (pitch, 1))
    with torch.cuda.device(mirror.device):
        mirror_cute = from_dlpack(storage.reshape(-1), assumed_align=16)
        norms_cute = from_dlpack(norms, assumed_align=4)
        stream = cuda.CUstream(
            torch.cuda.current_stream(mirror.device).cuda_stream
        )
        key = (
            _PACKED_ROWS,
            tuple(mirror.shape),
            pitch,
            mirror.device.index,
        )
        # The duplicated mirror and pitch keep one physical source for float
        # and integer reducers; this specialization never writes through out.
        compiled = _STAGE_ROW_NORMS_COMPILED.pop(key, None)
        if compiled is None:
            compiled = cute.compile(
                _StageRowNormsSm90(),
                mirror_cute,
                mirror_cute,
                norms_cute,
                Int64(dim),
                Int64(pitch),
                Int64(pitch),
                stream,
            )
        _STAGE_ROW_NORMS_COMPILED[key] = compiled
        while len(_STAGE_ROW_NORMS_COMPILED) > 4:
            _STAGE_ROW_NORMS_COMPILED.popitem(last=False)
    args = (
        mirror_cute,
        mirror_cute,
        norms_cute,
        Int64(dim),
        Int64(pitch),
        Int64(pitch),
    )
    return PreparedRowNorms(compiled, args, mirror, norms)


@torch.no_grad()
def launch_row_norms(prepared):
    """Reduce exact row norms on the caller's current CUDA stream."""
    if not isinstance(prepared, PreparedRowNorms):
        raise TypeError("prepared row norms have the wrong type")
    stream = cuda.CUstream(
        torch.cuda.current_stream(prepared.mirror.device).cuda_stream
    )
    prepared.compiled(*prepared.args, stream)


@torch.no_grad()
def affine_cast(
    x, bias, scale, out, peak, *, packed_rows=False, norms=None
):
    """Write an E4M3 mirror and optionally reduce its peak and row norms."""
    require_tensor(x, "rows", dtype=torch.float32)
    require_tensor(
        out,
        "E4M3 mirror",
        dtype=torch.float8_e4m3fn,
        contiguous=False,
    )
    with_peak = peak is not None
    if with_peak:
        require_tensor(
            peak, "peak", dtype=torch.float32, shape=(), device=x.device
        )
    valid_scale = math.isfinite(scale) and scale > 0.0
    if not valid_scale or not math.log2(scale).is_integer():
        raise ValueError("FP8 assignment scale must be a power of two")
    if scale > torch.finfo(torch.float32).max:
        raise ValueError("FP8 assignment scale exceeds the FP32 range")
    if with_peak and peak.requires_grad:
        raise ValueError("FP8 peak accumulator does not match input")
    if out.device != x.device or out.shape != x.shape or out.stride(1) != 1:
        raise ValueError("E4M3 mirror does not match rows")
    centered = bias is not None
    if centered:
        require_tensor(
            bias,
            "bias",
            dtype=torch.float32,
            shape=(x.shape[1],),
            device=x.device,
        )
        bound_bias = bias
    else:
        bound_bias = x[0]
    with_norms = norms is not None
    peak_only = with_peak and not with_norms
    if with_norms:
        require_tensor(
            norms, "row norms", dtype=torch.float32,
            shape=(len(x),), device=x.device,
        )
    bound_peak = peak if with_peak else x[0]
    pitch = out.stride(0)
    storage = out.as_strided((len(x), pitch), (pitch, 1))
    with torch.cuda.device(x.device):
        x_cute = from_dlpack(x.reshape(-1), assumed_align=4)
        bias_cute = from_dlpack(bound_bias, assumed_align=4)
        out_cute = from_dlpack(storage.reshape(-1), assumed_align=16)
        peak_cute = from_dlpack(bound_peak, assumed_align=4)
        if not peak_only:
            bound_norms = norms if with_norms else x[0]
            norms_cute = from_dlpack(bound_norms, assumed_align=4)
        stream = cuda.CUstream(
            torch.cuda.current_stream(x.device).cuda_stream
        )
        rows_per_block = _PACKED_ROWS if packed_rows else 1
        key = (
            tuple(x.shape),
            pitch,
            centered,
            rows_per_block,
            with_peak,
            with_norms,
            x.device.index,
        )
        compiled = _AFFINE_PREPARE_COMPILED.pop(key, None)
        if peak_only:
            operation = _PeakOnlyAffineCastSm90(centered, rows_per_block)
            args = (x_cute, bias_cute, out_cute, peak_cute)
        else:
            operation = _AffineCastSm90(
                centered, rows_per_block, with_peak, with_norms
            )
            args = (x_cute, bias_cute, out_cute, peak_cute, norms_cute)
        args += (Float32(scale), Int64(x.shape[1]), Int64(pitch))
        if compiled is None:
            compiled = cute.compile(operation, *args, stream)
        _AFFINE_PREPARE_COMPILED[key] = compiled
        while len(_AFFINE_PREPARE_COMPILED) > 4:
            _AFFINE_PREPARE_COMPILED.popitem(last=False)
        compiled(*args, stream)


class _DecodeArgminOutputsSm90:
    @cute.jit
    def __call__(
        self,
        keys: cute.Tensor,
        codes: cute.Tensor,
        scores: cute.Tensor,
        stream: cuda.CUstream,
    ):
        self.kernel(keys, codes, scores).launch(
            grid=(cute.ceil_div(codes.shape[0], _THREADS), 1, 1),
            block=(_THREADS, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        keys: cute.Tensor,
        codes: cute.Tensor,
        scores: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        row = bidx * _THREADS + tidx
        if row < codes.shape[0]:
            key = keys[row]
            codes[row] = (key & Int64(0xFFFFFFFF)).to(Int32)
            scores[row] = decode_argmin_score(key)
            keys[row] = Int64(INITIAL_KEY)


class PreparedArgminDecode(NamedTuple):
    compiled: object
    keys_cute: object
    codes_cute: object
    scores_cute: object
    keys: torch.Tensor
    codes: torch.Tensor
    scores: torch.Tensor


@torch.no_grad()
def prepare_argmin_decode(keys, codes, scores):
    """Bind stable private outputs and compile their decode/reset kernel."""
    require_tensor(keys, "keys", dtype=torch.int64)
    count = keys.numel()
    require_tensor(
        codes,
        "codes",
        dtype=torch.int32,
        shape=(count,),
        device=keys.device,
    )
    require_tensor(
        scores,
        "scores",
        dtype=torch.float32,
        shape=(count,),
        device=keys.device,
    )
    flat_keys = keys.reshape(-1)
    keys_cute = from_dlpack(flat_keys, assumed_align=8)
    codes_cute = from_dlpack(codes, assumed_align=4)
    scores_cute = from_dlpack(scores, assumed_align=4)
    stream = cuda.CUstream(torch.cuda.current_stream(keys.device).cuda_stream)
    cache_key = (count, keys.device.index)
    compiled = _COMPILED.pop(cache_key, None)
    if compiled is None:
        compiled = cute.compile(
            _DecodeArgminOutputsSm90(),
            keys_cute,
            codes_cute,
            scores_cute,
            stream,
        )
    _COMPILED[cache_key] = compiled
    while len(_COMPILED) > 2:
        _COMPILED.popitem(last=False)
    return PreparedArgminDecode(
        compiled,
        keys_cute,
        codes_cute,
        scores_cute,
        keys,
        codes,
        scores,
    )


@torch.no_grad()
def launch_argmin_decode(prepared):
    """Decode on the caller's current stream and restore the key sentinel."""
    stream = cuda.CUstream(
        torch.cuda.current_stream(prepared.scores.device).cuda_stream
    )
    prepared.compiled(
        prepared.keys_cute,
        prepared.codes_cute,
        prepared.scores_cute,
        stream,
    )


@torch.no_grad()
def finalize_int8_outputs(prepared):
    """Decode packed winners and reconstruct distances through INT64."""
    keys, row_norms, distances, labels = prepared
    flat = keys.reshape(-1)
    labels.copy_(flat).bitwise_and_(0xFFFFFFFF)
    flat.bitwise_right_shift_(32).add_(row_norms)
    distances.copy_(flat)
    flat.fill_(INTEGER_INITIAL_KEY)
