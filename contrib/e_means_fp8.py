# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Fit and validate the shared affine transform for E4M3 assignment."""

import math

import torch


_CENTER_GAIN = 1.5
_FIT_ROWS = 262144
_FORMAT_MAX = 448.0
_NEAR_PARITY = 0.001
_PROBE_CENTROIDS = 1024
_PROBE_ROWS = 2048
_TARGET_PEAK = 56.0
_VALIDATION_CHUNK_SIZE = 65536


def _check_fp8_range(peak, name):
    if not (float(peak) <= _FORMAT_MAX):
        raise ValueError("%s exceeds the E4M3 representable range" % name)


def _check_fp8_support(device):
    """Reject devices that cannot execute the E4M3 assignment path."""
    if getattr(torch, "float8_e4m3fn", None) is None:
        raise RuntimeError("PyTorch does not provide float8_e4m3fn")
    if device.type != "cuda":
        raise ValueError("float8_e4m3fn assignment requires a CUDA tensor")
    capability = torch.cuda.get_device_capability(device)
    if (
        getattr(torch.version, "hip", None) is not None
        or capability not in ((9, 0), (10, 0), (10, 3))
    ):
        raise RuntimeError(
            "float8_e4m3fn assignment requires an NVIDIA SM90, SM100, "
            "or SM103 GPU"
        )


class E4M3Transform:
    """A fitted shift and exact power-of-two scale."""

    def __init__(
        self, shift, scale, centered, fit_rows, sample_peak,
        plain_excess, centered_excess, decision,
    ):
        valid = math.isfinite(scale) and scale > 0.0
        if not valid or not math.log2(scale).is_integer():
            raise ValueError("the assignment scale must be a power of two")
        self.shift = shift
        self.scale = float(scale)
        self._bias = None if shift is None else -(shift * self.scale)
        self.centered = centered
        self.fit_rows = fit_rows
        self.sample_peak = sample_peak
        self.plain_excess = plain_excess
        self.centered_excess = centered_excess
        self.decision = decision

    def apply(self, value):
        if self._bias is not None:
            return torch.add(self._bias, value, alpha=self.scale)
        return value * self.scale

    def recipe(self):
        return {
            "assignment_transform_version": 1,
            "assignment_transform_scale": self.scale,
            "assignment_transform_centered": self.centered,
            "assignment_transform_fit_rows": self.fit_rows,
            "assignment_transform_sample_peak": self.sample_peak,
            "assignment_transform_plain_excess": self.plain_excess,
            "assignment_transform_centered_excess": self.centered_excess,
            "assignment_transform_decision": self.decision,
        }


def _candidate(sample, center):
    shift = sample.mean(dim=0) if center else None
    value = sample - shift if shift is not None else sample
    peak = float(value.abs().max())
    scale = 1.0
    if peak > 0.0 and math.isfinite(peak):
        scale = 2.0 ** math.floor(math.log2(_TARGET_PEAK / peak))
    return shift, scale, peak


def _rounded(value, candidate):
    shift, scale, _ = candidate
    if shift is not None:
        value = value - shift
    return (value * scale).to(torch.float8_e4m3fn).float()


def _excess_distortion(x, centroids, candidate):
    reference = (
        centroids.double().square().sum(dim=1)[None, :]
        - 2.0 * (x.double() @ centroids.double().t())
    )
    x_rounded = _rounded(x, candidate)
    c_rounded = _rounded(centroids, candidate)
    narrowed = (
        c_rounded.double().square().sum(dim=1)[None, :]
        - 2.0 * (x_rounded.double() @ c_rounded.double().t())
    )
    reference_label = reference.argmin(dim=1)
    narrowed_label = narrowed.argmin(dim=1)
    row = torch.arange(len(x), device=x.device)
    x_norm = x.double().square().sum(dim=1)
    best = (reference[row, reference_label] + x_norm).clamp_min(0).sum()
    took = (reference[row, narrowed_label] + x_norm).clamp_min(0).sum()
    return float((took - best) / best) if float(best) > 0.0 else 0.0


def fit(x):
    """Fit the frozen E4M3 transform from a bounded FP32 prefix."""
    sample = x[: min(len(x), _FIT_ROWS)]
    plain = _candidate(sample, False)
    centered = _candidate(sample, True)
    plain_excess = centered_excess = None

    probe = sample[: _PROBE_ROWS + _PROBE_CENTROIDS]
    if len(probe) < 2 * _PROBE_CENTROIDS:
        use_center = True
        decision = "small_sample_centered"
    else:
        queries = probe[: -_PROBE_CENTROIDS]
        centroids = probe[-_PROBE_CENTROIDS :]
        plain_excess = _excess_distortion(queries, centroids, plain)
        centered_excess = _excess_distortion(queries, centroids, centered)
        if plain_excess * 2.0 <= _NEAR_PARITY:
            use_center = False
            decision = "uncentered_near_parity"
        elif centered_excess <= 0.0 < plain_excess:
            use_center = True
            decision = "centered_nonpositive_excess"
        elif centered_excess > 0.0 and (
            plain_excess / centered_excess >= _CENTER_GAIN
        ):
            use_center = True
            decision = "centered_gain"
        else:
            use_center = False
            decision = "uncentered_insufficient_gain"
        if min(plain_excess, centered_excess) > _NEAR_PARITY:
            decision += ";both_above_quality_gate"

    shift, scale, peak = centered if use_center else plain
    if scale > torch.finfo(torch.float32).max:
        raise ValueError("the E4M3 assignment scale exceeds the FP32 range")
    return E4M3Transform(
        shift, scale, use_center, len(sample), peak,
        plain_excess, centered_excess, decision,
    )


def validate_rows(x, transform, name):
    """Reject rows that the fitted transform cannot represent in E4M3."""
    peak = x.new_zeros(())
    for begin in range(0, len(x), _VALIDATION_CHUNK_SIZE):
        value = transform.apply(x[begin : begin + _VALIDATION_CHUNK_SIZE])
        peak = torch.maximum(peak, value.abs().max())
    _check_fp8_range(peak, name)


def _update_fp8_peak(x, transform, peak_out):
    peak = transform.apply(x).detach().abs().max()
    peak_out.copy_(torch.maximum(peak_out, peak))
