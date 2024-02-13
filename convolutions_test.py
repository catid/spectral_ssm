import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import random
import time
import unittest

def ref_causal_conv(input_tensor, weight, bias=None, padding=0, groups=1):
    B, D, L = input_tensor.shape

    output = torch.zeros(B, D, L)

    for b in range(B):
        for d in range(D):
            for l in range(L):
                for d_ in range(D):
                    for l_ in range(L):
                        if d_ == d and l_ - l + L - 1 >= 0 and l_ - l + L - 1 < L:
                            output[b, d, l] += input_tensor[b, d_, l_] * weight[d, l_ - l + L - 1]

    return output

def conv1d_causal_conv(input_tensor, weight):
    B, D, L = input_tensor.shape

    return F.conv1d(input_tensor, weight.unsqueeze(1), bias=None, padding=L-1, groups=D)[..., :L]

def fft_conv1d_causal_conv(input_tensor, weight):
    B, C, L = input_tensor.shape  # Batch size, Channels, Length
    _, W_L = weight.shape  # Assuming weight shape is [Output Channels, Filter Length] for simplicity

    # Step 1: Pad the input on the left to maintain causality
    pad_len = W_L - 1
    input_padded = F.pad(input_tensor, (pad_len, 0), "constant", 0)  # Left padding to maintain causality

    # Step 2: Determine the length after padding for efficient FFT
    # No need to double the size as we are not avoiding circular convolution here
    total_length_fft = input_padded.shape[-1]

    # Prepare weight for FFT: Assume single output channel for simplicity
    # Pad weight to match the FFT length
    weight_padded_fft = F.pad(weight, (0, total_length_fft - W_L), "constant", 0)
    weight_padded_fft = weight_padded_fft.expand(B, -1, -1)  # Expand weight dimensions to match input batch size

    # Perform FFT on padded input and weights
    input_fft = torch.fft.rfft(input_padded, dim=2)
    weight_fft = torch.fft.rfft(weight_padded_fft, dim=2)

    # Element-wise multiplication in the frequency domain
    result_fft = input_fft * weight_fft

    # Inverse FFT to convert back to the time domain
    result_ifft = torch.fft.irfft(result_fft, n=total_length_fft, dim=2)

    # Trim the output to match the original input length
    result_trimmed = result_ifft[..., pad_len:]

    return result_trimmed



class TestCausalConv(unittest.TestCase):
    def test_output_shape(self):
        B, D, L = 2, 16, 128

        input_tensor = torch.randn(B, D, L)
        weight = torch.randn(D, L)

        method1 = ref_causal_conv(input_tensor, weight)
        method2 = conv1d_causal_conv(input_tensor, weight)
        method3 = fft_conv1d_causal_conv(input_tensor, weight)

        self.assertEqual(method1.shape, method2.shape)
        self.assertEqual(method1.shape, method3.shape)

    def test_output_values(self):
        B, D, L = 2, 16, 128

        input_tensor = torch.randn(B, D, L)
        weight = torch.randn(D, L)

        t0 = time.time()
        method1 = ref_causal_conv(input_tensor, weight)
        t1 = time.time()
        method2 = conv1d_causal_conv(input_tensor, weight)
        t2 = time.time()
        method3 = fft_conv1d_causal_conv(input_tensor, weight)
        t3 = time.time()

        print(f"method1 time: {t1 - t0}")
        print(f"method2 time: {t2 - t1}")
        print(f"method3 time: {t3 - t2}")

        print(f"method2: {method2}")
        print(f"method3: {method3}")

        np.testing.assert_allclose(method1.numpy(), method2.numpy(), atol=1e-5)
        np.testing.assert_allclose(method1.numpy(), method3.numpy(), atol=1e-5)

def seed_random(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

if __name__ == '__main__':
    seed_random(42)
    unittest.main()
