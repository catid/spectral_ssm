import torch
import torch.nn.functional as F
import numpy as np
import time
import unittest

from convolutions import fft_causal_conv

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

if __name__ == '__main__':
    unittest.main()
