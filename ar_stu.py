import torch
import torch.nn as nn
import torch.nn.functional as F

from convolutions import ConvolutionLayer
from autoregressive import AutoRegressiveCausalLayer

class AR_STULayer(nn.Module):
    def __init__(self, D_in, D_out, L, K=16, alpha=0.9):
        super(AR_STULayer, self).__init__()
        self.L = L
        self.K = K
        self.D_in = D_in
        self.D_out = D_out

        # Autoregressive sum for u input from Eq. 6
        self.autoregressive_u = AutoRegressiveCausalLayer(D_in, D_out, sum_count=3)

        # Spectral component from Eq. 6
        self.convolution_layer = ConvolutionLayer(D_in, D_out, L, K)

    def forward(self, u):
        u = u.permute(0, 2, 1) # Convert to [B, D, L]

        y = self.autoregressive_u(u)
        # FIXME: Add k_y AR components
        y = y + self.convolution_layer(u)

        y = y.permute(0, 2, 1) # Convert to [B, L, D]
        return y
