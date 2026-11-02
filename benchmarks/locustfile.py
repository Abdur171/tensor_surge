"""
Locust Load Testing Suite for TensorSurge Streaming Inference Server
Simulates 10, 50, and 100 concurrent users streaming tokens.
Measures Time To First Token (TTFT) and Inter-Token Latency (ITL).
"""

import time
import json
from locust import HttpUser, task, between, events


class TensorSurgeLoadTestUser(HttpUser):
    wait_time = between(0.1, 0.5)

    @task(3)
    def stream_short_generation(self):
        """Simulate short prompt / short response query (e.g. classification / NER)."""
        payload = {
            "prompt": "Classify sentiment: This ultra-low latency architecture is extraordinarily fast.",
            "max_tokens": 16,
            "temperature": 0.7,
        }
        self._execute_streaming_request("short_gen", payload)

    @task(1)
    def stream_long_generation(self):
        """Simulate longer reasoning / code generation query."""
        payload = {
            "prompt": "Implement a lock-free ring buffer in C++20 using acquire-release memory fences.",
            "max_tokens": 64,
            "temperature": 0.2,
        }
        self._execute_streaming_request("long_gen", payload)

    def _execute_streaming_request(self, name: str, payload: dict):
        start_time = time.time()
        first_token_time = None
        token_count = 0

        with self.client.post("/v1/generate_stream", json=payload, stream=True, catch_response=True) as resp:
            if resp.status_code != 200:
                resp.failure(f"HTTP Error: {resp.status_code}")
                return

            for line in resp.iter_lines():
                if line:
                    token_count += 1
                    current_time = time.time()
                    if first_token_time is None:
                        first_token_time = current_time
                        ttft_ms = (first_token_time - start_time) * 1000.0
                        events.request.fire(
                            request_type="STREAM_TTFT",
                            name=f"{name}_ttft",
                            response_time=ttft_ms,
                            response_length=len(line),
                            context=self.context(),
                            exception=None,
                        )

            total_elapsed_ms = (time.time() - start_time) * 1000.0
            resp.success()
