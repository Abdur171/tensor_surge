"""
Virtualized PagedAttention Memory Management Architecture
Divides continuous key-value (KV) cache memory into non-contiguous physical pages (blocks),
eliminating internal and external memory fragmentation.
Supports Copy-on-Write (CoW) for parallel beam search and prefix caching.
"""

from typing import Dict, List, Optional, Tuple
import math


class PhysicalBlock:
    """Represents a discrete physical memory block holding block_size token KV vectors."""
    def __init__(self, block_id: int, block_size: int = 16):
        self.block_id = block_id
        self.block_size = block_size
        self.ref_count = 0
        self.num_tokens = 0

    @property
    def is_full(self) -> bool:
        return self.num_tokens >= self.block_size

    def reset(self):
        self.ref_count = 0
        self.num_tokens = 0


class BlockManager:
    """
    Paged memory manager that coordinates virtual-to-physical address translation
    for transformer KV-caches.
    """
    def __init__(self, total_blocks: int = 1024, block_size: int = 16):
        self.total_blocks = total_blocks
        self.block_size = block_size
        self.blocks: List[PhysicalBlock] = [
            PhysicalBlock(block_id=i, block_size=block_size) for i in range(total_blocks)
        ]
        self.free_list: List[int] = list(range(total_blocks))
        # Mapping from sequence_id to list of allocated physical blocks
        self.seq_block_tables: Dict[str, List[PhysicalBlock]] = {}

    def can_allocate(self, num_tokens: int) -> bool:
        """Determines if there are enough free blocks to hold num_tokens."""
        blocks_needed = math.ceil(num_tokens / self.block_size)
        return len(self.free_list) >= blocks_needed

    def allocate(self, seq_id: str, num_tokens: int) -> bool:
        """Allocates non-contiguous physical blocks for a new sequence prompt."""
        if seq_id in self.seq_block_tables:
            raise ValueError(f"Sequence {seq_id} is already allocated.")

        blocks_needed = math.ceil(num_tokens / self.block_size)
        if len(self.free_list) < blocks_needed:
            return False

        allocated_blocks: List[PhysicalBlock] = []
        tokens_remaining = num_tokens

        for _ in range(blocks_needed):
            block_id = self.free_list.pop()
            block = self.blocks[block_id]
            block.ref_count = 1
            block.num_tokens = min(tokens_remaining, self.block_size)
            tokens_remaining -= block.num_tokens
            allocated_blocks.append(block)

        self.seq_block_tables[seq_id] = allocated_blocks
        return True

    def can_append_slot(self, seq_id: str) -> bool:
        """Checks if a sequence can append one more token in decode phase."""
        if seq_id not in self.seq_block_tables:
            return False
        table = self.seq_block_tables[seq_id]
        if not table:
            return len(self.free_list) >= 1
        last_block = table[-1]
        if not last_block.is_full:
            return True
        return len(self.free_list) >= 1

    def append_slot(self, seq_id: str) -> Tuple[int, int]:
        """
        Appends a new token slot for autoregressive decode.
        Handles Copy-on-Write (CoW) if the tail block is shared (ref_count > 1).
        Returns (physical_block_id, slot_offset).
        """
        table = self.seq_block_tables[seq_id]
        last_block = table[-1] if table else None

        if last_block is not None and not last_block.is_full:
            # Check for Copy-on-Write if tail block is shared
            if last_block.ref_count > 1:
                # Fork the block
                if not self.free_list:
                    raise RuntimeError("Out of GPU memory blocks during CoW fork")
                new_block_id = self.free_list.pop()
                new_block = self.blocks[new_block_id]
                new_block.ref_count = 1
                new_block.num_tokens = last_block.num_tokens
                last_block.ref_count -= 1
                table[-1] = new_block
                last_block = new_block

            slot_offset = last_block.num_tokens
            last_block.num_tokens += 1
            return last_block.block_id, slot_offset

        # Tail block is full or table empty: allocate a new physical block
        if not self.free_list:
            raise RuntimeError(f"Out of GPU memory blocks for sequence {seq_id}")
        new_block_id = self.free_list.pop()
        new_block = self.blocks[new_block_id]
        new_block.ref_count = 1
        new_block.num_tokens = 1
        table.append(new_block)
        return new_block.block_id, 0

    def fork(self, parent_seq_id: str, child_seq_id: str):
        """
        Implements Copy-on-Write (CoW) branching for beam search and parallel sampling.
        Shares physical blocks with child sequence and increments ref_counts.
        """
        if parent_seq_id not in self.seq_block_tables:
            raise KeyError(f"Parent sequence {parent_seq_id} does not exist.")
        if child_seq_id in self.seq_block_tables:
            raise ValueError(f"Child sequence {child_seq_id} already exists.")

        parent_table = self.seq_block_tables[parent_seq_id]
        child_table: List[PhysicalBlock] = []

        for block in parent_table:
            block.ref_count += 1
            child_table.append(block)

        self.seq_block_tables[child_seq_id] = child_table

    def free(self, seq_id: str):
        """Releases physical blocks associated with a finished sequence back to the free list."""
        if seq_id not in self.seq_block_tables:
            return

        table = self.seq_block_tables.pop(seq_id)
        for block in table:
            block.ref_count -= 1
            if block.ref_count == 0:
                block.reset()
                self.free_list.append(block.block_id)

    def get_block_table(self, seq_id: str) -> List[int]:
        """Returns the list of physical block IDs for sequence KV cache indexing."""
        if seq_id not in self.seq_block_tables:
            return []
        return [b.block_id for b in self.seq_block_tables[seq_id]]

    @property
    def num_free_blocks(self) -> int:
        return len(self.free_list)

    @property
    def num_allocated_blocks(self) -> int:
        return self.total_blocks - len(self.free_list)

    @property
    def memory_utilization(self) -> float:
        return self.num_allocated_blocks / self.total_blocks
