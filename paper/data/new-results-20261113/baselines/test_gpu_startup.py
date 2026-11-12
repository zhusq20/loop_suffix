"""CPU fault-injection checks for the observed exclusive-GPU startup race."""
from contextlib import ExitStack
import os
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import gpu_startup


class StartupTests(unittest.TestCase):
    def attempt(self, app_rows, cuda_errors=(), allowed=None, mode='Exclusive_Process', clock=None):
        fake = SimpleNamespace(cuda=SimpleNamespace(set_device=Mock(side_effect=list(cuda_errors) or None),
            synchronize=Mock()), empty=Mock(return_value=object()))
        self.fake = fake
        outputs = [f'GPU-test, {mode}\n', *app_rows]
        with ExitStack() as stack:
            stack.enter_context(patch.dict(sys.modules, {'torch': fake}))
            stack.enter_context(patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': 'GPU-test'}))
            stack.enter_context(patch.object(gpu_startup, 'allowed_gpus', side_effect=allowed,
                return_value=[2]))
            stack.enter_context(patch.object(gpu_startup.subprocess, 'check_output', side_effect=outputs))
            stack.enter_context(patch.object(gpu_startup.time, 'sleep'))
            if clock is not None:
                stack.enter_context(patch.object(gpu_startup.time, 'monotonic', side_effect=clock))
            return gpu_startup.acquire_cuda_context(2, 'GPU-test')

    def test_waits_for_existing_foreign_context(self):
        _, receipt = self.attempt(['GPU-test, 999999\n', ''])
        self.assertEqual(receipt['attempts'], 1)
        self.fake.empty.assert_called_once_with(1, device='cuda')

    def test_handles_foreign_context_acquired_after_query(self):
        _, receipt = self.attempt(['', ''], [RuntimeError('CUDA-capable device(s) is/are busy or unavailable'), None])
        self.assertEqual(receipt['cuda_busy_errors'], 1)
        self.assertEqual(receipt['attempts'], 2)

    def test_unknown_cuda_error_is_not_retried(self):
        with self.assertRaisesRegex(RuntimeError, 'out of memory'):
            self.attempt([''], [RuntimeError('out of memory')])
        self.fake.cuda.set_device.assert_called_once()

    def test_released_gpu_cannot_be_acquired(self):
        with self.assertRaisesRegex(AssertionError, 'allocation changed'):
            self.attempt([], allowed=[[2], [3,4,6]])
        self.fake.cuda.set_device.assert_not_called()

    def test_timeout_does_not_signal_foreign_process(self):
        with patch.object(os, 'kill', side_effect=AssertionError('Must not signal')):
            with self.assertRaises(TimeoutError):
                self.attempt(['GPU-test, 999999\n'], clock=[0, 61])
        self.fake.cuda.set_device.assert_not_called()

    def test_refuses_shared_compute_mode(self):
        with self.assertRaisesRegex(AssertionError, 'Exclusive GPU context'):
            self.attempt([], mode='Default')
        self.fake.cuda.set_device.assert_not_called()


if __name__ == '__main__':
    unittest.main()
