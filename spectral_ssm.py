import torch
import torch.nn as nn
import torch.nn.functional as F

from ar_stu import AR_STULayer

# Basic MLP feed-forward network like in transformers
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

# Causal average pooling: Output at time T is the average of prior values.
class CausalAveragePooling(nn.Module):
    def __init__(self):
        super(CausalAveragePooling, self).__init__()

    def forward(self, x):
        B, L, D = x.shape
        cumulative_sum = x.cumsum(dim=1)
        timesteps = torch.arange(1, L + 1, device=x.device).view(1, L, 1).expand(B, L, D)
        causal_average = cumulative_sum / timesteps
        return causal_average # [B, L, D]

class SpectralSSM(nn.Module):
    def __init__(self, d_in, d_hidden, d_out, L, num_layers=2):
        super(SpectralSSM, self).__init__()
        self.d_in = d_in
        self.d_hidden = d_hidden
        self.d_out = d_out

        # Fig.5: Embedding Layer
        self.proj_in = nn.Linear(d_in, d_hidden, bias=False)

        # Fig.5: Dense (output) Layer [B, L, d_out]
        self.proj_out = nn.Linear(d_hidden, d_out, bias=False)

        # Repeat num_layers times:
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = nn.Sequential(
                AR_STULayer(d_hidden, d_hidden, L), # Fig.5: STU
                FeedForward(d_hidden, d_hidden) # Fig.5: MLP+Non-LIN
            )
            self.layers.append(layer)

        # Fig.5: Time Pool
        self.time_pool = CausalAveragePooling()

    def reset(self):
        for layer in self.layers:
            layer.reset()

    def forward(self, u):
        assert u.dim() == 3, "Input shape must be [B, L, D]"
        B, L, D = u.shape

        y = self.proj_in(u)
        for layer in self.layers:
            y = layer(y)

        y = self.time_pool(y)

        y = self.proj_out(y) # [B, L, d_out]

        return y
