import torch
import torch.nn.functional as F
import numpy as np
import time
import unittest

from convolutions import fft_causal_conv

def naive_causal_convolution(u, k):
    assert k.dtype == torch.float, "Kernel must be of type float for quality."
    
    # Length of input signal and kernel
    L = u.shape[-1]
    M = k.shape[-1]
    
    # Ensure u is the same dtype as k
    u = u.to(k.dtype)
    
    # Prepare output tensor, initialized with zeros
    y = torch.zeros_like(u)
    
    # Perform the naive causal convolution
    for i in range(L):
        for j in range(M):
            if i - j >= 0:
                y[..., i] += u[..., i-j] * k[..., j]
    
    return y


class TestNaiveCausalConvolution(unittest.TestCase):
    def test_convolution(self):
        device = "cuda"
        dtype = torch.float16

        B, H, L = 2, 3, 16  # Batch size, heads, maximum sequence length
        k = torch.rand(H, L, dtype=torch.float)  # Kernel
        k = k.to(device)

        # Generate a single, long input sequence
        u = torch.rand(B, H, L, dtype=dtype)
        u = u.to(device)

        # Compute output with the original, unchanged input
        original_out = fft_causal_conv(u, k)

        naive_out = naive_causal_convolution(u, k)

        np.testing.assert_allclose(original_out.cpu().numpy(), naive_out.cpu().numpy(), atol=1e-5)

        # Create a modified version of u_variable where data after L is changed
        modified_u_variable = u.clone()
        R = 1
        modified_u_variable[..., (L-R):] = torch.rand(B, H, R)

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
