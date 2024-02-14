import torch
import torch.nn as nn
import torch.nn.init as init

# Expected input shape [B, D, L]
# Auto-regressive sum for input:
# y[t] = w0 * x[t] + w1 * x[t-1] + w2 * x[t-2] + ... where sum_count = number of w terms.
class AutoRegressiveCausalInput(nn.Module):
    def __init__(self, D_in, D_out, Ku=3):
        super(AutoRegressiveCausalInput, self).__init__()

        self.Ku = Ku
        self.conv = nn.Conv1d(D_in, D_out, kernel_size=Ku, padding=Ku-1, bias=False)

        init.xavier_uniform_(self.conv.weight)
        #nn.init.zeros_(self.conv.weight)

    def forward(self, x):
        y = self.conv(x)
        y = y[:, :, :-(self.Ku-1)] if self.Ku > 1 else y
        return y

# Expected input shape [B, D, L]
# Auto-regressive sum for output:
# y[t] = y[t] + w0 * y[t-1] + w1 * y[t-2] + ... where sum_count = number of w terms.
class AutoRegressiveCausalOutput(nn.Module):
    def __init__(self, D, Ky=2, alpha=0.9):
        super(AutoRegressiveCausalOutput, self).__init__()

        self.Ky = Ky
        self.conv = nn.Conv1d(D, D, kernel_size=Ky, padding=Ky-1, bias=False)

        # Initialization as per section 5
        nn.init.zeros_(self.conv.weight)
        self.conv.weight.data[:, :, -2] = torch.eye(D) * alpha

    def forward(self, x):
        # Shift x forward in time one, removing the top entry.
        x_shift = torch.cat((torch.zeros_like(x[:, :, :1]), x[:, :, :-1]), dim=2)

        y = self.conv(x_shift)
        y = y[:, :, :-(self.Ky-1)] if self.Ky > 1 else y

        # Sum with original input
        y = y + x

        return y
