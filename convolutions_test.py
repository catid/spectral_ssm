import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import random

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

class TestCausalConv(unittest.TestCase):
    def test_output_shape(self):
        B, D, L = 2, 16, 128

        input_tensor = torch.randn(B, D, L)
        weight = torch.randn(D, L)

        method1 = ref_causal_conv(input_tensor, weight)
        method2 = conv1d_causal_conv(input_tensor, weight)
        
        self.assertEqual(method1.shape, method2.shape)

    def test_output_values(self):
        B, D, L = 2, 16, 128

        input_tensor = torch.randn(B, D, L)
        weight = torch.randn(D, L)

        method1 = ref_causal_conv(input_tensor, weight)
        method2 = conv1d_causal_conv(input_tensor, weight)
        
        np.testing.assert_allclose(method1.numpy(), method2.numpy(), atol=1e-5)

def seed_random(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

if __name__ == '__main__':
    seed_random(42)
    unittest.main()
