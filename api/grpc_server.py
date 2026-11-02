"""
Asynchronous Streaming Inference API with TTFT and ITL Telemetry
Delivers token-by-token streaming responses while recording real-time latency metrics:
- Time To First Token (TTFT)
- Inter-Token Latency (ITL)
- Output token throughput (tokens / second)
"""

import sys
import os
import asyncio
import time
import uuid
from typing import AsyncGenerator, Dict, Any, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from engine.block_manager import BlockManager
from engine.scheduler import ContinuousScheduler, Sequence
from engine.model_runner import ModelRunner, ModelRunnerConfig


class InferenceEngine:
    """Core asynchronous serving engine coordinating scheduling, execution, and queues."""
    def __init__(
        self,
        total_blocks: int = 512,
        block_size: int = 16,
        max_batch_size: int = 16,
    ):
        self.block_manager = BlockManager(total_blocks=total_blocks, block_size=block_size)
        self.scheduler = ContinuousScheduler(
            block_manager=self.block_manager,
            max_batch_size=max_batch_size,
        )
        self.model_runner = ModelRunner(ModelRunnerConfig(hidden_dim=256, num_heads=4, head_dim=64))
        self.response_queues: Dict[str, asyncio.Queue] = {}
        self._running = False
        self._engine_task: asyncio.Task = None

    async def start(self):
        """Starts the background continuous batching loop."""
        if not self._running:
            self._running = True
            self._engine_task = asyncio.create_task(self._continuous_loop())

    async def stop(self):
        """Gracefully stops the continuous batching loop."""
        self._running = False
        if self._engine_task:
            self._engine_task.cancel()
            try:
                await self._engine_task
            except asyncio.CancelledError:
                pass

    async def _continuous_loop(self):
        """Main asynchronous event loop driving model iterations."""
        while self._running:
            try:
                batch = self.scheduler.schedule()
                if batch is not None and len(batch) > 0:
                    # Execute one forward iteration
                    loop = asyncio.get_running_loop()
                    step_results = await loop.run_in_executor(None, self.model_runner.step, batch)

                    now = time.time()
                    for seq in batch.sequences:
                        if seq.seq_id in self.response_queues:
                            q = self.response_queues[seq.seq_id]
                            token_id = step_results.get(seq.seq_id, 0)
                            await q.put((token_id, now, seq.is_finished))
                else:
                    # No work ready, yield CPU briefly
                    await asyncio.sleep(0.002)
            except Exception as e:
                await asyncio.sleep(0.005)

    async def generate_stream(
        self,
        prompt: str,
        max_tokens: int = 32,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Asynchronously streams generated tokens with TTFT and ITL telemetry.
        """
        seq_id = str(uuid.uuid4())
        # Naive tokenization for demo/inference: split by space or character
        prompt_tokens = [hash(w) % 32000 for w in prompt.split()]
        if not prompt_tokens:
            prompt_tokens = [1]

        q = asyncio.Queue()
        self.response_queues[seq_id] = q

        req_arrival = time.time()
        seq = self.scheduler.add_request(
            seq_id=seq_id,
            prompt=prompt,
            prompt_tokens=prompt_tokens,
            max_tokens=max_tokens,
        )

        last_token_time = req_arrival
        token_count = 0
        ttft_ms = 0.0

        try:
            while True:
                token_id, token_time, is_finished = await q.get()
                token_count += 1

                if token_count == 1:
                    ttft_ms = (token_time - req_arrival) * 1000.0
                    itl_ms = ttft_ms
                else:
                    itl_ms = (token_time - last_token_time) * 1000.0

                last_token_time = token_time
                elapsed_total_ms = (token_time - req_arrival) * 1000.0
                throughput = token_count / (elapsed_total_ms / 1000.0) if elapsed_total_ms > 0 else 0.0

                yield {
                    "sequence_id": seq_id,
                    "token_id": token_id,
                    "token_text": f"tok_{token_id}",
                    "token_index": token_count,
                    "is_finished": is_finished,
                    "ttft_ms": round(ttft_ms, 2),
                    "itl_ms": round(itl_ms, 2),
                    "total_latency_ms": round(elapsed_total_ms, 2),
                    "throughput_tokens_sec": round(throughput, 1),
                }

                if is_finished:
                    break
        finally:
            self.response_queues.pop(seq_id, None)


async def main_demo():
    """Demonstration of asynchronous streaming server."""
    print("Starting TensorSurge Inference Server...")
    engine = InferenceEngine(total_blocks=256, block_size=16, max_batch_size=8)
    await engine.start()

    print("Submitting concurrent generation requests...")
    tasks = [
        engine.generate_stream(f"Prompt #{i}: What is quantum algorithmic transpilation?", max_tokens=10)
        for i in range(3)
    ]

    async def consume(stream_idx, generator):
        async for chunk in generator:
            print(f"[Req {stream_idx}] Tok #{chunk['token_index']}: TTFT={chunk['ttft_ms']}ms, ITL={chunk['itl_ms']}ms, Finished={chunk['is_finished']}")

    await asyncio.gather(*(consume(i, g) for i, g in enumerate(tasks)))
    await engine.stop()
    print("TensorSurge Inference Server terminated gracefully.")


if __name__ == "__main__":
    asyncio.run(main_demo())
