"""
Unit Tests for Streaming Inference API
Verifies async streaming token delivery, TTFT/ITL metrics, and engine shutdown.
"""

import sys
import os
import unittest
import asyncio

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from api.grpc_server import InferenceEngine


class TestStreamingAPI(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.engine = InferenceEngine(total_blocks=64, block_size=16, max_batch_size=4)
        await self.engine.start()

    async def asyncTearDown(self):
        await self.engine.stop()

    async def test_streaming_generation_telemetry(self):
        chunks = []
        async for chunk in self.engine.generate_stream("Test continuous streaming inference", max_tokens=5):
            chunks.append(chunk)

        self.assertEqual(len(chunks), 5)
        self.assertTrue(chunks[-1]["is_finished"])

        # Check TTFT and ITL are recorded and positive
        first_chunk = chunks[0]
        self.assertGreater(first_chunk["ttft_ms"], 0.0)
        self.assertGreater(first_chunk["itl_ms"], 0.0)

        # Check subsequent token indexes
        for i, chunk in enumerate(chunks, start=1):
            self.assertEqual(chunk["token_index"], i)
            self.assertIn("throughput_tokens_sec", chunk)


if __name__ == "__main__":
    unittest.main()
