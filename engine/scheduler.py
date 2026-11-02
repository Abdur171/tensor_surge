"""
Dynamic Continuous Iteration-Level Scheduler (vLLM / Orca Paradigm)
Performs request batching at the fine-grained token iteration boundary rather than
coarse-grained request boundary, achieving >2.5x higher token throughput.
"""

from enum import Enum
from typing import List, Optional, Dict
from collections import deque
import time
from .block_manager import BlockManager


class SequenceStatus(Enum):
    WAITING = "WAITING"
    RUNNING = "RUNNING"
    SWAPPED = "SWAPPED"
    FINISHED = "FINISHED"


class Sequence:
    """Represents an active generation request with prompt and decoded tokens."""
    def __init__(self, seq_id: str, prompt: str, prompt_tokens: List[int], max_tokens: int = 128):
        self.seq_id = seq_id
        self.prompt = prompt
        self.prompt_tokens = prompt_tokens
        self.output_tokens: List[int] = []
        self.max_tokens = max_tokens
        self.status = SequenceStatus.WAITING
        self.arrival_time = time.time()
        self.first_token_time: Optional[float] = None
        self.finish_time: Optional[float] = None

    @property
    def is_finished(self) -> bool:
        return len(self.output_tokens) >= self.max_tokens or self.status == SequenceStatus.FINISHED

    @property
    def num_prompt_tokens(self) -> int:
        return len(self.prompt_tokens)

    @property
    def num_output_tokens(self) -> int:
        return len(self.output_tokens)

    @property
    def total_tokens(self) -> int:
        return self.num_prompt_tokens + self.num_output_tokens


class ScheduledBatch:
    """Contains the batch of sequences scheduled for an engine forward iteration."""
    def __init__(self, sequences: List[Sequence], is_prefill: bool, block_tables: Dict[str, List[int]]):
        self.sequences = sequences
        self.is_prefill = is_prefill
        self.block_tables = block_tables

    def __len__(self) -> int:
        return len(self.sequences)


class ContinuousScheduler:
    """
    Iteration-level continuous batching scheduler.
    Coordinates between arriving sequences and PagedAttention BlockManager.
    """
    def __init__(
        self,
        block_manager: BlockManager,
        max_batch_size: int = 32,
        max_tokens_per_iteration: int = 2048,
    ):
        self.block_manager = block_manager
        self.max_batch_size = max_batch_size
        self.max_tokens_per_iteration = max_tokens_per_iteration

        self.waiting: deque[Sequence] = deque()
        self.running: List[Sequence] = []
        self.swapped: List[Sequence] = []

    def add_request(self, seq_id: str, prompt: str, prompt_tokens: List[int], max_tokens: int = 64) -> Sequence:
        """Enqueues a new generation request into the waiting queue."""
        seq = Sequence(seq_id=seq_id, prompt=prompt, prompt_tokens=prompt_tokens, max_tokens=max_tokens)
        self.waiting.append(seq)
        return seq

    def schedule(self) -> Optional[ScheduledBatch]:
        """
        Schedules next iteration batch:
        1. Decode phase: prioritizes advancing active running sequences.
        2. Prefill phase: admits new waiting requests if memory and batch limits permit.
        """
        # Purge finished sequences
        finished_seqs = [s for s in self.running if s.is_finished]
        for s in finished_seqs:
            s.finish_time = time.time()
            s.status = SequenceStatus.FINISHED
            self.block_manager.free(s.seq_id)
            self.running.remove(s)

        # 1. Schedule Decode Iteration for running sequences
        if self.running:
            scheduled_running: List[Sequence] = []
            block_tables: Dict[str, List[int]] = {}

            for seq in list(self.running):
                # Ensure memory block exists for new token
                if self.block_manager.can_append_slot(seq.seq_id):
                    self.block_manager.append_slot(seq.seq_id)
                    scheduled_running.append(seq)
                    block_tables[seq.seq_id] = self.block_manager.get_block_table(seq.seq_id)
                else:
                    # Preemption: out of memory blocks during decode
                    self._preempt(seq)

            if scheduled_running:
                return ScheduledBatch(sequences=scheduled_running, is_prefill=False, block_tables=block_tables)

        # 2. Schedule Prefill Iteration for waiting sequences
        if self.waiting:
            scheduled_waiting: List[Sequence] = []
            block_tables: Dict[str, List[int]] = {}
            total_tokens = 0

            while self.waiting and len(scheduled_waiting) < self.max_batch_size:
                seq = self.waiting[0]
                if total_tokens + seq.num_prompt_tokens > self.max_tokens_per_iteration:
                    break

                if self.block_manager.can_allocate(seq.num_prompt_tokens):
                    self.waiting.popleft()
                    self.block_manager.allocate(seq.seq_id, seq.num_prompt_tokens)
                    seq.status = SequenceStatus.RUNNING
                    scheduled_waiting.append(seq)
                    self.running.append(seq)
                    block_tables[seq.seq_id] = self.block_manager.get_block_table(seq.seq_id)
                    total_tokens += seq.num_prompt_tokens
                else:
                    break

            if scheduled_waiting:
                return ScheduledBatch(sequences=scheduled_waiting, is_prefill=True, block_tables=block_tables)

        return None

    def _preempt(self, seq: Sequence):
        """Preempts a running sequence due to memory exhaustion."""
        seq.status = SequenceStatus.SWAPPED
        self.running.remove(seq)
        self.block_manager.free(seq.seq_id)
        # Re-queue at the front of waiting for recomputation
        self.waiting.appendleft(seq)

    def has_unfinished_requests(self) -> bool:
        return bool(self.waiting or self.running or self.swapped)
