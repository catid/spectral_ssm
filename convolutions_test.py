import torch
import torch.nn.functional as F
import numpy as np
import time
import unittest

from convolutions import fft_causal_conv, ConvolutionLayer

def ref_causal_convolution(u, k):
    B, D, L = u.shape  # Batch size, Dimension, Length of the input signal
    H, M = k.shape  # Number of filters, Length of each filter

    assert k.dtype == torch.float, "Kernel must be of type float for quality."
    u = u.to(k.dtype)  # Ensure u is the same dtype as k
    
    # Initialize the output tensor with zeros, with a new shape to accommodate H different convolutions
    y = torch.zeros(B, H, D, L, dtype=k.dtype, device=u.device)
    
    # Iterate over each filter in the kernel
    for h in range(H):
        # Apply each filter across all dimensions D of each batch in u
        for d in range(D):
            # Perform the naive causal convolution for each filter
            for i in range(L):
                for j in range(M):
                    if i - j >= 0:  # Ensure causality
                        y[:, h, d, i] += u[:, d, i-j] * k[h, j]
    
    return y


class TestNaiveCausalConvolution(unittest.TestCase):
    def test_convolution(self):
        device = "cuda"
        dtype = torch.float16

        # Note: Usually D is the last dimension but we expect it to be in the middle of the input.
        B, D, L = 2, 3, 16  # Batch size, dimension, sequence length
        H = 8 # Heads (K in the paper)
        k = torch.rand(H, L, dtype=torch.float)  # Kernel
        k = k.to(device)

        # Generate a single, long input sequence
        u = torch.rand(B, D, L, dtype=dtype)
        u = u.to(device)

        naive_out = ref_causal_convolution(u, k)
        self.assertEqual(naive_out.shape, (B, H, D, L))

        # Compute output with the original, unchanged input
        original_out = fft_causal_conv(u, k)
        self.assertEqual(original_out.shape, (B, H, D, L))

        np.testing.assert_allclose(original_out.cpu().numpy(), naive_out.cpu().numpy(), atol=1e-5)

        # Create a modified version of u_variable where data after L is changed
        modified_u_variable = u.clone()
        R = 1
        modified_u_variable[..., (L-R):] = torch.rand(B, D, R, dtype=dtype).to(device)

        # Compute output with the modified input
        t0 = time.time()
        modified_out = fft_causal_conv(modified_u_variable, k)
        t1 = time.time()

        print(f"time={t1 - t0}")

        original_out = original_out[..., :L-R]
        modified_out = modified_out[..., :L-R]

        np.testing.assert_allclose(original_out.cpu().numpy(), modified_out.cpu().numpy(), atol=2e-3)

class TestConvolutionLayer(unittest.TestCase):
    def setUp(self):
        # Example dimensions
        self.D_in = 3
        self.D_out = 11
        self.L = 2048
        self.K = 13
        
        # Initialize the ConvolutionLayer
        self.layer = ConvolutionLayer(self.D_in, self.D_out, self.L, self.K)

    def test_parameter_shapes(self):
        # Verify the shape of the learned projections M
        self.assertEqual(self.layer.M.shape, (self.K, self.D_in, self.D_out),
                         "Shape of learned projections M is incorrect.")

        # Verify the shape of eigenvector_k_f
        expected_eigenvector_shape = (1, self.K, 1, self.L+1)
        self.assertEqual(self.layer.eigenvector_k_f_imag.shape, expected_eigenvector_shape,
                         "Shape of eigenvector_k_f is incorrect.")

    def test_forward_shape(self):
        # Create a dummy input tensor
        u = torch.randn(1, self.D_in, self.L)  # Note: Ensure input shape is [B, D, L]

        # Forward pass
        y = self.layer(u)

        # Expected output shape [B, D_out, L]
        expected_shape = (1, self.D_out, self.L)

        # Check the output shape
        self.assertEqual(y.shape, expected_shape,
                         f"Output shape is incorrect. Expected {expected_shape}, got {y.shape}.")

    # Add more tests as needed, e.g., to verify numerical outputs or layer properties


if __name__ == '__main__':
    unittest.main()
