"""
Unit Test & Numerical Parity Verification for Fused Attention (FlashAttention)
Verifies: torch.allclose(out_fused, out_sdpa, atol=1e-3, rtol=1e-3)
for both causal and non-causal attention configurations.
"""

import sys
import os
import unittest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from kernels.triton.fused_attention import fused_attention, reference_attention


class TestFusedAttention(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_non_causal_parity(self):
        """Test standard non-causal full attention."""
        batch_size, num_heads, seq_len, head_dim = 2, 4, 128, 64
        q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=self.device, dtype=torch.float32)
        k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=self.device, dtype=torch.float32)
        v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=self.device, dtype=torch.float32)

        out_fused = fused_attention(q, k, v, is_causal=False)
        out_ref = reference_attention(q, k, v, is_causal=False)

        max_diff = torch.max(torch.abs(out_fused - out_ref)).item()
        print(f"Non-causal attention max abs diff: {max_diff:.8e}")

        self.assertTrue(
            torch.allclose(out_fused, out_ref, atol=1e-3, rtol=1e-3),
            f"Non-causal attention parity failed! Max diff: {max_diff}"
        )

    def test_causal_parity(self):
        """Test autoregressive causal masked attention."""
        batch_size, num_heads, seq_len, head_dim = 2, 4, 256, 64
        q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=self.device, dtype=torch.float32)
        k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=self.device, dtype=torch.float32)
        v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=self.device, dtype=torch.float32)

        out_fused = fused_attention(q, k, v, is_causal=True)
        out_ref = reference_attention(q, k, v, is_causal=True)

        max_diff = torch.max(torch.abs(out_fused - out_ref)).item()
        print(f"Causal attention max abs diff: {max_diff:.8e}")

        self.assertTrue(
            torch.allclose(out_fused, out_ref, atol=1e-3, rtol=1e-3),
            f"Causal attention parity failed! Max diff: {max_diff}"
        )

    def test_sequence_length_scaling(self):
        """Test varying sequence lengths from 64 to 512."""
        for seq_len in [64, 128, 256, 512]:
            q = torch.randn(1, 2, seq_len, 64, device=self.device, dtype=torch.float32)
            k = torch.randn(1, 2, seq_len, 64, device=self.device, dtype=torch.float32)
            v = torch.randn(1, 2, seq_len, 64, device=self.device, dtype=torch.float32)

            out_fused = fused_attention(q, k, v, is_causal=True)
            out_ref = reference_attention(q, k, v, is_causal=True)

            self.assertTrue(
                torch.allclose(out_fused, out_ref, atol=1e-3, rtol=1e-3),
                f"Failed for seq_len={seq_len}"
            )


if __name__ == "__main__":
    unittest.main()
