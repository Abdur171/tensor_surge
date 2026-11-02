"""
Unit Tests for Continuous Iteration-Level Scheduler
Verifies prefill vs decode scheduling, dynamic batching, and completion lifecycles.
"""

import sys
import os
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from engine.block_manager import BlockManager
from engine.scheduler import ContinuousScheduler, SequenceStatus


class TestContinuousScheduler(unittest.TestCase):
    def setUp(self):
        self.block_mgr = BlockManager(total_blocks=32, block_size=16)
        self.scheduler = ContinuousScheduler(
            block_manager=self.block_mgr,
            max_batch_size=4,
            max_tokens_per_iteration=512,
        )

    def test_prefill_scheduling(self):
        seq1 = self.scheduler.add_request("s1", "Hello world", [10, 20, 30], max_tokens=5)
        seq2 = self.scheduler.add_request("s2", "Deep tech AI", [40, 50], max_tokens=5)

        batch = self.scheduler.schedule()
        self.assertIsNotNone(batch)
        self.assertTrue(batch.is_prefill)
        self.assertEqual(len(batch), 2)
        self.assertEqual(seq1.status, SequenceStatus.RUNNING)
        self.assertEqual(seq2.status, SequenceStatus.RUNNING)

    def test_decode_scheduling_and_lifecycle(self):
        seq1 = self.scheduler.add_request("s1", "Prompt", [1, 2], max_tokens=3)

        # 1. Prefill step
        batch1 = self.scheduler.schedule()
        self.assertTrue(batch1.is_prefill)

        # Simulate generating first token
        seq1.output_tokens.append(101)

        # 2. First Decode step
        batch2 = self.scheduler.schedule()
        self.assertIsNotNone(batch2)
        self.assertFalse(batch2.is_prefill)
        self.assertEqual(len(batch2), 1)

        # Simulate generating second and third tokens
        seq1.output_tokens.append(102)
        seq1.output_tokens.append(103)

        # 3. Next step should recognize completion and free memory
        self.assertTrue(seq1.is_finished)
        batch3 = self.scheduler.schedule()
        self.assertIsNone(batch3)
        self.assertEqual(seq1.status, SequenceStatus.FINISHED)
        self.assertEqual(self.block_mgr.num_allocated_blocks, 0)


if __name__ == "__main__":
    unittest.main()
