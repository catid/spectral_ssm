import torch
import torch.nn as nn
import torch.nn.functional as F

from convolutions import ConvolutionLayer
from autoregressive import AutoRegressiveCausalLayer

class AR_STULayer(nn.Module):
    def __init__(self, d_in, d_out, L, k=16, alpha=0.9):
        super(AR_STULayer, self).__init__()
        self.L = L
        self.k = k
        self.d_in = d_in
        self.d_out = d_out

        # Autoregressive sum for u input from Eq. 6
        self.autoregressive_u = AutoRegressiveCausalLayer(d_in, d_out, K=3)

        # Spectral component from Eq. 6
        self.convolution_layer = ConvolutionLayer(d_in, d_out, L, k)

    def forward(self, u):
        y = self.autoregressive_u(u)
        # FIXME: Add k_y AR components
        y = y + self.convolution_layer(u)
        return y
