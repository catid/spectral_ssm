import torch
import torch.nn as nn
import torch.nn.functional as F

from convolutions import ConvolutionLayer

class AR_STULayer(nn.Module):
    def __init__(self, d_in, d_out, L, k=16, alpha=0.9):
        super(AR_STULayer, self).__init__()
        self.L = L
        self.k = k
        self.d_in = d_in
        self.d_out = d_out

        # Implementing fixed ky=2
        self.My_n1 = nn.Linear(d_out, d_out, bias=False)
        self.My_n2 = nn.Linear(d_out, d_out, bias=False)

        self.Mu_0 = nn.Linear(d_in, d_out, bias=False)
        self.Mu_1 = nn.Linear(d_in, d_out, bias=False)
        self.Mu_2 = nn.Linear(d_in, d_out, bias=False)

        # Initialize to STU model by default
        self.My_n2.weight = nn.Parameter(torch.eye(d_out) * alpha)
        for proj in [self.My_n1, self.Mu_0, self.Mu_1, self.Mu_2]:
            nn.init.zeros_(proj.weight)

        self.convolution_layer = ConvolutionLayer(d_in, d_out, L, k)

    def forward(self, u, u_n1=None, u_n2=None, y_n1=None, y_n2=None):
        # This implements Eq (6) that defines AR-STU
        result = self.Mu_0(u)
        if u_n1 is not None:
            result = result + self.Mu_1(u_n1)
        if u_n2 is not None:
            result = result + self.Mu_2(u_n2)
        if y_n1 is not None:
            result = result + self.My_n1(y_n1)
        if y_n2 is not None:
            result = result + self.My_n2(y_n2)
        result = result + self.convolution_layer(u)
        return result
