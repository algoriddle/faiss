# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import importlib.metadata
import unittest

import numpy as np
import torch

from faiss.contrib.e_means import Emeans
from faiss.contrib.e_means.e_means_quantizer import CastQuantizer


def _ready():
    if not torch.cuda.is_available() or getattr(torch.version, 'hip', None) is not None:
        return False
    if torch.cuda.get_device_capability() != (9, 0):
        return False
    try:
        return (importlib.metadata.version('quack-kernels') == '0.6.5'
                and importlib.metadata.version('nvidia-cutlass-dsl') == '4.7.1'
                and importlib.metadata.version('cuda-python').startswith('13.'))
    except importlib.metadata.PackageNotFoundError:
        return False


class TestEmeansCute(unittest.TestCase):
    def test_scope(self):
        model = Emeans(2, 2, assignment_backend='cute')
        for x in (np.zeros((2, 2), dtype='float32'), torch.zeros(2, 2)):
            with self.assertRaisesRegex(ValueError, 'CuTe assignment requires CUDA'):
                model.train(x)
            with self.assertRaisesRegex(ValueError, 'CuTe assignment requires CUDA'):
                model.begin(2, x)
            model.centroids = x
            with self.assertRaisesRegex(ValueError, 'CuTe assignment requires CUDA'):
                model.assign(x)
            model.reset()
        with self.assertRaisesRegex(ValueError, 'requires BF16'):
            Emeans(2, 2, assignment_backend='cute', quantizer=CastQuantizer(2, torch.float16))
        with self.assertRaisesRegex(ValueError, 'int32'):
            Emeans(2, 2**31 - 127, assignment_backend='cute')

    @unittest.skipUnless(_ready(), 'pinned CuTe stack on SM90 required')
    def test_native_and_jit_training(self):
        x = torch.tensor([[0.], [2.], [10.], [12.]], device='cuda')
        initial = x[[0, 2]]
        native = Emeans(1, 2, niter=1, batch_size=4, alpha0=.5, assignment_backend='cute')
        jit = Emeans(1, 2, niter=1, batch_size=4, alpha0=.5, assignment_backend='cute',
                     quantizer=CastQuantizer(1, torch.bfloat16))
        native.train(x.bfloat16(), initial)
        jit.train(x, initial)
        for model in (native, jit):
            torch.testing.assert_close(model.centroids, torch.tensor([[1.], [11.]], device='cuda'))
            self.assertEqual(model.obj.item(), 8.)
        distance, labels = native.assign(x.bfloat16())
        torch.testing.assert_close(distance, torch.ones(4, device='cuda'))
        torch.testing.assert_close(labels, torch.tensor([0, 0, 1, 1], device='cuda'))
        with self.assertRaisesRegex(ValueError, 'requires native BF16'):
            Emeans(1, 2, assignment_backend='cute').train(x, initial)
