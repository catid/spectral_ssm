import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math

from convolutions import ConvolutionLayer
from autoregressive import AutoRegressiveCausalInput, AutoRegressiveCausalOutput

class AR_STULayer(nn.Module):
    def __init__(self, D_in, D_out, L, K=16, Ku=3, Ky=2, alpha=0.9):
        super(AR_STULayer, self).__init__()

        #self.bn = nn.BatchNorm1d(D_in)

        # Autoregressive sum for u (input) from Eq. 6
        self.autoregressive_u = AutoRegressiveCausalInput(D_in, D_out, Ku=Ku)

        # Autoregressive sum for y (output) from Eq. 6
        self.autoregressive_y = AutoRegressiveCausalOutput(D_out, Ky=Ky, alpha=alpha)

        # Spectral component from Eq. 6
        self.convolution_layer = ConvolutionLayer(D_in, D_out, L, K)

    def forward(self, u):
        u = u.permute(0, 2, 1) # Convert to [B, D, L]

        #u = self.bn(u)

        # Spectral channelization and mixing via convolution
        y = self.convolution_layer(u)

        # Causal auto-regressive process for inputs
        y = y + self.autoregressive_u(u)

        # Causal auto-regressive process for outputs
        y = self.autoregressive_y(y)

        y = y.permute(0, 2, 1) # Convert to [B, L, D]
        return y
