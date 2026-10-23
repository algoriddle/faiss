# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import importlib.metadata
import gc
import unittest
import weakref
from unittest import mock

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
    @unittest.skipUnless(_ready(), 'pinned CuTe stack on SM90 required')
    def test_captured_binding_survives_eviction_and_retires(self):
        rows = (torch.arange(10247, device='cuda') % 2).bfloat16()
        rows = rows[:, None].expand(-1, 16).contiguous()
        initial = torch.tensor([0., 1.], device='cuda')[:, None].expand(-1, 16).contiguous()
        get_chunk = Emeans._get_assignment_chunk_size
        capture_step = Emeans._capture_step
        end_epoch = Emeans.end_epoch

        # Diagnostic geometry creates a third binding size without fixing the
        # production resolver or increasing its cache limit.
        def chunk(model, is_torch, cuda=False):
            return 37 if model.assignment_backend == 'cute' and cuda else get_chunk(model, is_torch, cuda)

        for fail_after_tail in (False, True):
            with self.subTest(exception=fail_after_tail):
                owners = []
                graph_owner = None

                def capture(model, *args, **kwargs):
                    nonlocal graph_owner
                    captured = capture_step(model, *args, **kwargs)
                    owners.extend(weakref.ref(binding) for binding in captured[1].bindings.values())
                    graph_owner = weakref.ref(captured[0])
                    return captured

                def finish_epoch(model):
                    end_epoch(model)
                    gc.collect()
                    self.assertTrue(owners)
                    self.assertTrue(all(owner() is not None for owner in owners))
                    if fail_after_tail:
                        raise RuntimeError('injected post-tail failure')

                model = Emeans(16, 2, niter=2, seed=10, assignment_backend='cute')
                with mock.patch.object(Emeans, '_get_assignment_chunk_size', chunk), \
                     mock.patch.object(Emeans, '_capture_step', capture), \
                     mock.patch.object(Emeans, 'end_epoch', finish_epoch):
                    gc.collect()
                    if fail_after_tail:
                        with self.assertRaisesRegex(RuntimeError, 'injected post-tail failure'):
                            model.train(rows, initial)
                    else:
                        model.train(rows, initial)
                        torch.testing.assert_close(model.centroids, initial, rtol=0, atol=0)
                        torch.testing.assert_close(model.obj, torch.zeros_like(model.obj), rtol=0, atol=0)
                        distances, labels = model.assign(rows)
                        torch.testing.assert_close(distances, torch.zeros_like(distances), rtol=0, atol=0)
                        torch.testing.assert_close(labels, rows[:, 0].long(), rtol=0, atol=0)
                gc.collect()
                self.assertIsNotNone(graph_owner)
                self.assertIsNone(graph_owner())
                model.reset()
                gc.collect()
                self.assertTrue(all(owner() is None for owner in owners))

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
