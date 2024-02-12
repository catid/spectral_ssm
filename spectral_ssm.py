import torch
import torch.nn as nn
import torch.nn.functional as F

from ar_stu import AR_STULayer

class FeedForward(nn.Module):
    def __init__(self, d_in, d_out, mult=4):
        super(FeedForward, self).__init__()

        hidden_size = d_in * mult
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, d_out)
        )

    def forward(self, x):
        return self.net(x)

class SpectralSSM(nn.Module):
    def __init__(self, d_in, d_hidden, d_out, L, num_layers=2, k=16, alpha=0.9):
        super(SpectralSSM, self).__init__()
        self.d_in = d_in
        self.d_hidden = d_hidden
        self.d_out = d_out

        self.proj_in = nn.Linear(d_in, d_hidden, bias=False)
        self.proj_out = nn.Linear(d_hidden, d_out, bias=False)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = nn.Sequential(
                AR_STULayer(d_hidden, d_hidden, L, k, alpha),
                FeedForward(d_hidden, d_hidden)  # Assuming FeedForward is a defined class
            )
            self.layers.append(layer)

    def reset(self):
        for layer in self.layers:
            layer.reset()

    def forward(self, u):
        assert u.dim() == 3, "Input shape must be [B, L, D]"
        B, L, D = u.shape

        y = self.proj_in(u)
        for layer in self.layers:
            y = layer(y)

        yt = y.transpose(1, 2) # [B, D, L]
        avg_pooled = F.avg_pool1d(yt, kernel_size=L)
        avg_pooled = avg_pooled.transpose(1, 2) # [1, 1, D]

        y = self.proj_out(avg_pooled) # [1, 1, d_out]
        return y
