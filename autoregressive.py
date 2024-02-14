import torch.nn as nn

class AutoRegressiveCausalLayer(nn.Module):
    def __init__(self, D_in, D_out, sum_count=3):
        super(AutoRegressiveCausalLayer, self).__init__()

        self.sum_count = sum_count
        self.conv = nn.Conv1d(D_in, D_out, kernel_size=sum_count, padding=sum_count-1, bias=False)

        nn.init.zeros_(self.conv.weight)

    def forward(self, x):
        y = self.conv(x)
        y = y[:, :, :-(self.sum_count-1)] if self.sum_count > 1 else y
        return y
