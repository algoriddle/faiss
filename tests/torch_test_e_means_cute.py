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
from faiss.contrib.e_means.e_means_fp8 import Fp8Quantizer
from faiss.contrib.e_means.e_means_int8 import Int8Quantizer


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

    @unittest.skipUnless(_ready(), 'pinned CuTe stack on SM90 required')
    def test_fp8_native_and_jit_share_decoded_state_and_output_units(self):
        x = torch.tensor([[0.], [2.], [10.], [12.]], device='cuda')
        q = Fp8Quantizer(1)
        q.train(x)
        models = [Emeans(1, 2, niter=1, batch_size=4, alpha0=.5,
                         quantizer=q, assignment_backend='cute') for _ in range(2)]
        for model,rows in zip(models, (q.compute_codes(x),x)):
            model.train(rows, x[[0,2]])
            torch.testing.assert_close(model.centroids,x.new_tensor([[1.],[11.]]))
            self.assertEqual(model.obj.item(),8.)
            distance,labels = model.assign(rows)
            torch.testing.assert_close(distance,torch.ones(4,device=x.device))
            torch.testing.assert_close(labels,torch.tensor([0,0,1,1],device=x.device))

    @unittest.skipUnless(_ready(), 'pinned CuTe stack on SM90 required')
    def test_fp8_long_positive_selected_distances(self):
        d = 4096
        base = (torch.arange(d, device='cuda') % 251).float()
        initial = torch.stack((base, base + 32))
        x = torch.stack((base, base + 32, base + 40))
        q = Fp8Quantizer(d)
        q.train(torch.stack((torch.full_like(base, -128), torch.full_like(base, 128))))
        codes = q.compute_codes(x)
        operands = q.decode(q.compute_codes(initial)).double()
        represented = q.decode(codes).double()
        oracle = (represented[:, None] - operands[None]).square().sum(2)
        for rows in (codes, x):
            model = Emeans(d, 2, quantizer=q, assignment_backend='cute')
            model.centroids = initial.clone()
            distance, labels = model.assign(rows)
            torch.testing.assert_close(labels, oracle.argmin(1))
            chosen = oracle.gather(1, labels[:, None]).flatten()
            torch.testing.assert_close(distance.double(), chosen, rtol=2e-6, atol=1e-3)
            self.assertEqual(distance[0].item(), 0.)
            self.assertEqual(distance[1].item(), 0.)

    @unittest.skipUnless(_ready(), 'pinned CuTe stack on SM90 required')
    def test_native_bytes_and_caller_quantizer_use_decoded_units(self):
        x = torch.tensor([[0.],[2.],[10.],[12.]],device='cuda')
        q = Int8Quantizer(1)
        q.train(x)
        for rows,quantizer,offset in ((x.to(torch.int8),None,0),
                                     ((x+128).to(torch.uint8),None,128),
                                     (q.compute_codes(x),q,0)):
            with self.subTest(dtype=rows.dtype,calibrated=quantizer is not None):
                model = Emeans(1,2,niter=1,batch_size=4,alpha0=.5,
                               quantizer=quantizer,assignment_backend='cute')
                model.train(rows,(x+offset)[[0,2]])
                torch.testing.assert_close(model.centroids,x.new_tensor([[1.],[11.]])+offset)
                self.assertEqual(model.obj.item(),8.)
                distance,labels = model.assign(rows)
                torch.testing.assert_close(distance,torch.ones(4,device=x.device))
                torch.testing.assert_close(labels,torch.tensor([0,0,1,1],device=x.device))
        with self.assertRaisesRegex(TypeError,'no JIT INT8'):
            Emeans(1,2,quantizer=q,assignment_backend='cute').train(x)
