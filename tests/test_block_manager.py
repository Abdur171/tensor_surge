"""
Unit Tests for PagedAttention BlockManager
Verifies virtual-to-physical block table allocation, slot appending,
Copy-on-Write (CoW), and block lifecycle management.
"""

import sys
import os
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from engine.block_manager import BlockManager


class TestBlockManager(unittest.TestCase):
    def setUp(self):
        # 16 physical blocks of size 16 tokens (total capacity: 256 tokens)
        self.mgr = BlockManager(total_blocks=16, block_size=16)

    def test_initial_state(self):
        self.assertEqual(self.mgr.num_free_blocks, 16)
        self.assertEqual(self.mgr.num_allocated_blocks, 0)
        self.assertEqual(self.mgr.memory_utilization, 0.0)

    def test_allocate_prompt(self):
        # 35 tokens requires ceil(35 / 16) = 3 physical blocks
        success = self.mgr.allocate(seq_id="seq_1", num_tokens=35)
        self.assertTrue(success)
        self.assertEqual(self.mgr.num_free_blocks, 13)
        self.assertEqual(self.mgr.num_allocated_blocks, 3)

        table = self.mgr.get_block_table("seq_1")
        self.assertEqual(len(table), 3)

    def test_append_slot_within_block(self):
        # Allocate 10 tokens in 1 block (capacity 16)
        self.mgr.allocate(seq_id="seq_2", num_tokens=10)
        block_id, slot = self.mgr.append_slot("seq_2")
        self.assertEqual(slot, 10)
        # Still in the same block, no new block allocated
        self.assertEqual(self.mgr.num_free_blocks, 15)

    def test_append_slot_allocates_new_block_when_full(self):
        # Allocate 16 tokens (fills 1 block)
        self.mgr.allocate(seq_id="seq_3", num_tokens=16)
        self.assertEqual(self.mgr.num_allocated_blocks, 1)

        # Appending 1 token must trigger allocation of second block
        block_id, slot = self.mgr.append_slot("seq_3")
        self.assertEqual(slot, 0)
        self.assertEqual(self.mgr.num_allocated_blocks, 2)
        self.assertEqual(len(self.mgr.get_block_table("seq_3")), 2)

    def test_copy_on_write_fork(self):
        # Allocate parent sequence
        self.mgr.allocate(seq_id="parent", num_tokens=32)  # 2 blocks
        self.assertEqual(self.mgr.num_allocated_blocks, 2)

        # Fork child sequence (e.g. beam search)
        self.mgr.fork(parent_seq_id="parent", child_seq_id="child")
        # Physical blocks should be shared (ref_count == 2), no new blocks consumed yet
        self.assertEqual(self.mgr.num_allocated_blocks, 2)
        self.assertEqual(self.mgr.get_block_table("parent"), self.mgr.get_block_table("child"))

        # Appending to parent when tail is full triggers allocation of new block for parent
        self.mgr.append_slot("parent")
        self.assertEqual(self.mgr.num_allocated_blocks, 3)

    def test_free_and_recycle(self):
        self.mgr.allocate(seq_id="seq_4", num_tokens=48)  # 3 blocks
        self.assertEqual(self.mgr.num_free_blocks, 13)

        self.mgr.free("seq_4")
        self.assertEqual(self.mgr.num_free_blocks, 16)
        self.assertEqual(self.mgr.num_allocated_blocks, 0)
        self.assertEqual(self.mgr.get_block_table("seq_4"), [])


if __name__ == "__main__":
    unittest.main()
