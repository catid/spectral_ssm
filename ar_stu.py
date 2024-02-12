import torch
import torch.nn as nn
import torch.nn.functional as F

from convolutions import ConvolutionLayer

# FIXME: This is not implementing efficient causal batch mode.
# It can only produce one output at a time so we expect B=1
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

        self.reset()

    def reset(self):
        self.u_n1 = None
        self.u_n2 = None
        self.y_n1 = None
        self.y_n2 = None

    def remember(self, u, y):
        device = u.device

        self.u_n2 = self.u_n1
        self.u_n1 = u.detach().to(device)

        self.y_n2 = self.y_n1
        self.y_n1 = y.detach().to(device)

    def forward(self, u):
        assert u.shape[0] == 1, "FIXME: Batch not supported yet"
        # This implements Eq (6) that defines AR-STU
        y = self.Mu_0(u)
        if self.u_n1 is not None:
            y = y + self.Mu_1(self.u_n1)
        if self.u_n2 is not None:
            y = y + self.Mu_2(self.u_n2)
        if self.y_n1 is not None:
            y = y + self.My_n1(self.y_n1)
        if self.y_n2 is not None:
            y = y + self.My_n2(self.y_n2)
        y = y + self.convolution_layer(u)
        self.remember(u, y)
        return y
