import torch
import torch.nn as nn
import torch.nn.init as init

class AutoRegressiveCausalModel(nn.Module):
    def __init__(self, D_in, D_out, K=3):
        super(AutoRegressiveCausalModel, self).__init__()

        # Assuming A, B, and C are equivalent to convolutional filters
        self.K = K
        self.conv = nn.Conv1d(D_in, D_out, kernel_size=K, padding=K-1, bias=False)

        init.xavier_uniform_(self.conv.weight)

    def forward(self, x):
        # x is expected to have shape [B, D, L], but Conv1d expects [B, C, L]
        # No need to permute if x is already in the correct shape
        y = self.conv(x)  # No need for extra slicing if padding is adjusted correctly
        y = y[:, :, :-(self.K-1)] if self.K > 1 else y
        return y
