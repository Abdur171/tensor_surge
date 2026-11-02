"""
Unit Test & Numerical Parity Verification for Fused SwiGLU Kernel
Verifies: torch.allclose(out_fused, out_torch, atol=1e-3, rtol=1e-3)
across various tensor geometries and shapes.
"""

import sys
import os
import unittest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from kernels.triton.fused_swiglu import fused_swiglu, reference_swiglu, FusedSwiGLU


class TestFusedSwiGLU(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_forward_numerical_parity_standard(self):
        """Test forward numerical parity for standard hidden dimensions (e.g. LLaMA-7B: 4096 -> 11008)."""
        batch_size, seq_len, dim = 2, 128, 4096
        gate = torch.randn(batch_size, seq_len, dim, device=self.device, dtype=torch.float32)
        up = torch.randn(batch_size, seq_len, dim, device=self.device, dtype=torch.float32)

        out_fused = fused_swiglu(gate, up)
        out_ref = reference_swiglu(gate, up)

        max_diff = torch.max(torch.abs(out_fused - out_ref)).item()
        print(f"SwiGLU standard forward max abs diff: {max_diff:.8e}")

        self.assertTrue(
            torch.allclose(out_fused, out_ref, atol=1e-3, rtol=1e-3),
            f"Numerical parity failed! Max diff: {max_diff}"
        )

    def test_forward_odd_shapes(self):
        """Test irregular / unaligned shapes to verify tile masking."""
        for shape in [(1, 7, 33), (3, 17, 127), (5, 63, 1023), (2, 1, 512)]:
            gate = torch.randn(*shape, device=self.device, dtype=torch.float32)
            up = torch.randn(*shape, device=self.device, dtype=torch.float32)

            out_fused = fused_swiglu(gate, up)
            out_ref = reference_swiglu(gate, up)

            self.assertTrue(
                torch.allclose(out_fused, out_ref, atol=1e-3, rtol=1e-3),
                f"Parity failed for shape {shape}"
            )

    def test_backward_gradients_parity(self):
        """Verify analytical gradients match PyTorch autograd gradients."""
        gate_fused = torch.randn(2, 64, 256, device=self.device, dtype=torch.float32, requires_grad=True)
        up_fused = torch.randn(2, 64, 256, device=self.device, dtype=torch.float32, requires_grad=True)

        gate_ref = gate_fused.detach().clone().requires_grad_(True)
        up_ref = up_fused.detach().clone().requires_grad_(True)

        out_fused = fused_swiglu(gate_fused, up_fused)
        out_ref = reference_swiglu(gate_ref, up_ref)

        grad_output = torch.randn_like(out_fused)

        out_fused.backward(grad_output)
        out_ref.backward(grad_output)

        diff_gate = torch.max(torch.abs(gate_fused.grad - gate_ref.grad)).item()
        diff_up = torch.max(torch.abs(up_fused.grad - up_ref.grad)).item()

        print(f"SwiGLU backward grad gate max diff: {diff_gate:.8e}, up max diff: {diff_up:.8e}")

        self.assertTrue(
            torch.allclose(gate_fused.grad, gate_ref.grad, atol=1e-3, rtol=1e-3),
            f"Gate gradient mismatch! Max diff: {diff_gate}"
        )
        self.assertTrue(
            torch.allclose(up_fused.grad, up_ref.grad, atol=1e-3, rtol=1e-3),
            f"Up gradient mismatch! Max diff: {diff_up}"
        )

    def test_nn_module_integration(self):
        """Verify FusedSwiGLU nn.Module works in a neural network layer."""
        layer = FusedSwiGLU(in_features=128, hidden_features=256).to(self.device)
        x = torch.randn(4, 32, 128, device=self.device)
        out = layer(x)
        self.assertEqual(out.shape, (4, 32, 128))


if __name__ == "__main__":
    unittest.main()
