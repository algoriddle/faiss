# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Native-byte helpers for INT8 E-means assignment.

Floating corpora can be converted explicitly before training::

    q = Int8Quantizer(x.shape[1])
    q.train(representative_sample)
    x_i8 = q.compute_codes(x)
    model.train(x_i8)
    source_centroids = q.decode(model.centroids)

The quantizer trains a per-coordinate FP32 shift and one global power-of-two
scale. One global scale preserves L2 geometry before rounding; separate scales
would reweight the coordinates. ``compute_codes`` processes internal chunks,
rounds, and clips to INT8. Finite outliers clip; nonfinite values are rejected.

E-means returns squared byte-lattice distances. Dividing by ``scale**2`` gives
the squared distance between decoded lattice points, not the original vectors.
Decode learned FP32 centroids with ``q.decode(model.centroids)``.
"""

import math

import torch


_ENCODE_CHUNK_ELEMENTS = 8 * 1024 * 1024
_TARGET_PEAK = 112.0


def _matrix(x, name):
    if not isinstance(x, torch.Tensor):
        raise TypeError("%s must be a torch.Tensor" % name)
    if x.layout != torch.strided or x.ndim != 2:
        raise ValueError("%s must be a dense 2-D tensor" % name)
    if x.shape[0] == 0 or x.shape[1] == 0:
        raise ValueError("%s must not be empty" % name)
    if x.device.type not in ("cpu", "cuda"):
        raise ValueError("%s must be on a CPU or CUDA device" % name)
    return x.detach()


class Int8Quantizer:
    """An explicit affine preprocessor for floating-point corpora."""

    def __init__(self, d):
        self.d = int(d)
        if self.d < 1:
            raise ValueError("d must be positive")
        self.code_size = self.d
        self.is_trained = False
        self.shift = None
        self.scale = None
        self._bias = None

    def train(self, sample):
        """Train from a representative floating-point sample."""
        self.is_trained = False
        self.shift = None
        self.scale = None
        self._bias = None
        sample = _matrix(sample, "sample")
        if sample.shape[1] != self.d:
            raise ValueError("sample has the wrong dimension")
        if sample.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            raise TypeError("sample must be FP32, FP16, or BF16")
        sample = sample.float()
        if not bool(torch.isfinite(sample).all()):
            raise ValueError("sample must be finite")
        shift = sample.mean(dim=0)
        if not bool(torch.isfinite(shift).all()):
            raise ValueError("the quantizer shift must be finite")
        peak = float((sample - shift).abs().max())
        if not math.isfinite(peak):
            raise ValueError("the centered sample peak must be finite")
        exponent = 0 if peak == 0.0 else math.floor(
            math.log2(_TARGET_PEAK / peak)
        )
        if exponent > 127:
            raise ValueError("the quantizer scale exceeds the FP32 range")
        scale = 2.0**exponent
        bias = -(shift * scale)
        if not bool(torch.isfinite(bias).all()):
            raise ValueError("the quantizer bias must be finite")
        self.shift = shift.contiguous()
        self.scale = float(scale)
        self._bias = bias.contiguous()
        self.is_trained = True

    def _validate(self, x, name):
        if not self.is_trained:
            raise RuntimeError("quantizer must be trained")
        x = _matrix(x, name)
        if x.device != self.shift.device:
            raise ValueError(
                "%s and quantizer must be on the same device" % name
            )
        if x.shape[1] != self.d:
            raise ValueError("%s has the wrong dimension" % name)
        return x

    def compute_codes(self, x):
        """Compute signed-byte codes in bounded internal chunks."""
        x = self._validate(x, "x")
        if x.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            raise TypeError("x must be FP32, FP16, or BF16")
        codes = torch.empty(x.shape, dtype=torch.int8, device=x.device)
        finite = torch.ones((), dtype=torch.bool, device=x.device)
        chunk_rows = max(1, _ENCODE_CHUNK_ELEMENTS // self.d)
        for begin in range(0, len(x), chunk_rows):
            end = begin + chunk_rows
            value = torch.add(
                self._bias, x[begin:end].float(), alpha=self.scale
            )
            extrema = torch.stack(torch.aminmax(value))
            finite.logical_and_(torch.isfinite(extrema).all())
            value.round_().clamp_(-128, 127)
            codes[begin:end].copy_(value.to(torch.int8))
        if not bool(finite):
            raise ValueError("x must be finite under the quantizer transform")
        return codes

    def decode(self, value):
        """Restore INT8 codes or FP32 learned state to source coordinates."""
        value = self._validate(value, "value")
        if value.dtype not in (torch.int8, torch.float32):
            raise TypeError("value must be INT8 or FP32")
        if value.dtype == torch.float32 and not bool(
            torch.isfinite(value).all()
        ):
            raise ValueError("value must be finite")
        return value.float().div(self.scale).add(self.shift)


def _check_int8_support(device):
    """Reject devices outside the qualified INT8 assignment targets."""
    if device.type != "cuda":
        raise ValueError("INT8 assignment requires a CUDA tensor")
    capability = torch.cuda.get_device_capability(device)
    if (
        getattr(torch.version, "hip", None) is not None
        or capability not in ((8, 0), (9, 0), (10, 0))
    ):
        raise RuntimeError(
            "INT8 assignment requires an NVIDIA SM80, SM90, or SM100 GPU"
        )


class Int8Transform:
    """The frozen signed or unsigned source representation."""

    def __init__(self, representation):
        self.representation = representation

    def recipe(self):
        return {
            "assignment_int8_representation": self.representation,
        }


def representation(dtype):
    """Return the INT8 source representation selected by a Torch dtype."""
    if dtype == torch.int8:
        return "int8"
    if dtype == torch.uint8:
        return "uint8"
    raise TypeError("INT8 assignment requires INT8 or UINT8 input")


def resolve(dtype):
    """Freeze the native-byte representation selected by a Torch dtype."""
    return Int8Transform(representation(dtype))


def quantize(value, transform):
    """Map source-coordinate values to the transform's signed-byte lattice."""
    value = value.float()
    if transform.representation == "uint8":
        value = value - 128.0
    return value.round().clamp(-128, 127).to(torch.int8)


def separate_splits(donor, child, transform):
    """Separate only split pairs that collapse to one INT8 row."""
    donor_q, child_q = (quantize(x, transform) for x in (donor, child))
    rows = (donor_q == child_q).all(dim=1).nonzero(as_tuple=True)[0]
    if len(rows) == 0:
        return donor, child

    base = donor_q.index_select(0, rows).int()
    columns = torch.minimum(base + 128, 127 - base).argmax(dim=1)
    row = torch.arange(len(rows), device=rows.device)
    current = base[row, columns]
    offset = 128.0 if transform.representation == "uint8" else 0.0
    donor = donor.clone()
    child = child.clone()
    donor[rows, columns] = (
        torch.maximum(current - 1, current.new_tensor(-128)).float()
        + offset
    )
    child[rows, columns] = (
        torch.minimum(current + 1, current.new_tensor(127)).float()
        + offset
    )
    return donor, child
