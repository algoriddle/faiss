# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Bounded reads of the DINO bvecs corpus."""

from pathlib import Path

import numpy as np
import torch


class DinoBvecs:
    def __init__(self, root, n):
        directory = Path(root) / "chunked_base_10B"
        self.paths = sorted(
            p for p in directory.iterdir() if p.suffix == ".bvecs"
        )
        if not self.paths:
            raise ValueError(f"no bvecs chunks in {directory}")
        with self.paths[0].open("rb") as stream:
            self.d = int(np.frombuffer(stream.read(4), dtype="<i4")[0])
        if self.d <= 0:
            raise ValueError("invalid bvecs dimension")
        self.row_bytes = self.d + 4
        lengths = []
        self.maps = []
        for index, path in enumerate(self.paths):
            if sum(lengths) < n:
                if (
                    self.paths[0].match("chunk_[0-9][0-9][0-9][0-9].bvecs")
                    and path.name != f"chunk_{index:04d}.bvecs"
                ):
                    raise ValueError(f"missing chunk in required DINO prefix: {path}")
                if Path(str(path) + ".parts.json").exists():
                    raise ValueError(f"unfinished DINO download: {path}")
            size = path.stat().st_size
            if size % self.row_bytes:
                raise ValueError(f"incomplete bvecs chunk: {path}")
            lengths.append(size // self.row_bytes)
            self.maps.append(
                np.memmap(
                    path,
                    dtype=np.uint8,
                    mode="r",
                    shape=(lengths[-1], self.row_bytes),
                )
            )
        self.ends = np.cumsum(lengths)
        if not 0 < n <= self.ends[-1]:
            raise ValueError(
                f"requested {n} rows; corpus contains {self.ends[-1]}"
            )
        self.n = n

    def read(self, begin, end):
        if not 0 <= begin <= end <= self.n:
            raise IndexError("DINO row range out of bounds")
        output = np.empty((end - begin, self.d), dtype=np.uint8)
        at = 0
        for mapping, hi, lo in zip(
            self.maps, self.ends, np.r_[0, self.ends[:-1]]
        ):
            a, b = max(begin, int(lo)), min(end, int(hi))
            if a < b:
                output[at : at + b - a] = mapping[a - lo : b - lo, 4:]
                at += b - a
        return output

    def resident(self, begin, end, device):
        output = torch.empty((end - begin, self.d), device=device)
        for a in range(begin, end, 65536):
            b = min(end, a + 65536)
            output[a - begin : b - begin] = torch.from_numpy(
                self.read(a, b)
            ).to(device)
        return output
