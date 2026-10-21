# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Stateless FP16/BF16 casts for represented-value GPU E-means.

Codes are native Torch tensors, not packed CPU bytes. Conversion and decoding
check metadata only. Call validate explicitly for finite/range checks; it may
synchronize. Helpers can execute on CPU, but CPU E-means requires quantizer=None.
"""

import torch


class CastQuantizer:
    """Round floating rows to FP16 or BF16 without calibration.

    Values must be finite and fit the target's finite range. Unchecked casts
    follow Torch overflow/rounding semantics and can produce infinity.
    """

    def __init__(self, d, dtype):
        self.d = int(d)
        if self.d < 1:
            raise ValueError("d must be positive")
        if dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("dtype must be torch.float16 or torch.bfloat16")
        self.dtype = dtype
        self.code_size = 2 * self.d
        self.is_trained = True

    def train(self, sample=None):
        """No-op: this representation needs no calibration sample."""

    def to(self, device):
        """Return self; a stateless cast has no device-owned parameters."""
        if torch.device(device).type not in ("cpu", "cuda"):
            raise ValueError("device must be CPU or CUDA")
        return self

    def _matrix(self, x):
        if not isinstance(x, torch.Tensor):
            raise TypeError("x must be a torch.Tensor")
        if x.layout != torch.strided or x.ndim != 2 or x.shape[1] != self.d:
            raise ValueError("x must be a dense (n, d) tensor")
        if x.device.type not in ("cpu", "cuda"):
            raise ValueError("x must be on CPU or CUDA")
        return x.detach()

    def compute_codes(self, x):
        """Return native represented rows, without a value scan."""
        x = self._matrix(x)
        if x.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            raise TypeError("x must be FP32, FP16 or BF16")
        return x.to(self.dtype).contiguous()

    def decode(self, codes):
        """Widen matching represented rows to FP32."""
        codes = self._matrix(codes)
        if codes.dtype != self.dtype:
            raise TypeError("codes must have the quantizer dtype")
        return codes.float().contiguous()

    def validate(self, x):
        """Explicitly check finite values and the target finite range."""
        x = self._matrix(x)
        if x.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            raise TypeError("x must be FP32, FP16 or BF16")
        if not bool((x.float().abs() <= torch.finfo(self.dtype).max).all()):
            raise ValueError("x must be finite and fit the quantizer range")


def _ldexp(value, exponent):
    # Stock CUDA ldexp can launch on the current device rather than the operand's.
    if value.is_cuda:
        with torch.cuda.device(value.device):
            return torch.ldexp(value, exponent)
    return torch.ldexp(value, exponent)
