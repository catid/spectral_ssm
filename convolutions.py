import torch
import torch.nn as nn

from hankel_spectra import load_or_compute_eigen_data

# Input shapes: u [B, D, L], k [H, L]
# y = u * k (see convolutions_test.py for naive reference version)
# k.dtype should be float
# FP16 does not support odd sizes, has terrible accuracy, and it is also not any faster.
def fft_causal_conv(u, k):
    L = u.shape[-1]
    assert k.dtype == torch.float, "Too much quality loss with fp16"
    fft_size = 2*L
    k_f = torch.fft.rfft(k, n=fft_size) / fft_size
    u_f = torch.fft.rfft(u.to(k.dtype), n=fft_size)

    # Match up dimensions so we will produce [B, H, D, L]
    # We should end up with D*H sequences from D sequences of length L
    k_f = k_f.unsqueeze(0).unsqueeze(2)
    u_f = u_f.unsqueeze(1)

    prod = u_f * k_f

    y = torch.fft.irfft(prod, n=fft_size, norm="forward")[..., :L]
    return y

class ConvolutionLayer(nn.Module):
    def __init__(self, d_in, d_out, L, k):
        super(ConvolutionLayer, self).__init__()

        self.d_in = d_in
        self.d_out = d_out
        self.L = L
        self.k = k

        # Note the 2024 version only has a positive part to compute, so this code only works with that
        eigenvalues, eigenvectors = load_or_compute_eigen_data(L=L, k=k, matrix_version="2024")

        # Store k eigenvalues^^(1/4)
        self.eigenvalues = nn.Parameter(torch.tensor(eigenvalues, dtype=torch.float).pow(0.25), requires_grad=False)

        eigenvectors = torch.tensor(eigenvectors, dtype=torch.float)  # [k, L]
        eigenvector_ffts = torch.fft.rfft(eigenvectors)  # [k, L//2+1]
        eigenvector_expanded = eigenvector_ffts.unsqueeze(0).unsqueeze(2)  # [1, k, 1, L//2+1]
        self.eigenvector_ffts_expanded = nn.Parameter(eigenvector_expanded, requires_grad=False)

        # K parallel d_in -> d_out learned projections
        self.M = nn.Parameter(torch.Tensor(k, d_in, d_out))
        nn.init.xavier_uniform_(self.M)

    def forward(self, u):
        # Ensure input shape is [B, L, D]
        assert u.dim() == 3, "Input shape must be [B, L, D]"
        B, L, D = u.shape
        assert self.d_in == D, "Unexpected input dimension"
        assert self.L == L, "Sequence length change is unsupported"

        # [B, L, D] -> [B, D, L]
        u_transposed = u.transpose(1, 2)

        convolutions = fft_causal_conv(u, self.eigenvector_ffts_expanded)

        # Combined operations:
        # 1. Scale by k eigenvalues
        # 2. Permute from [B, k, D, L] to [B, k, L, D]
        # 3. Apply M_k to project from D to d_out with K different projection matrices
        # 4. Sum across k to get [B, L, d_out]
        return torch.einsum('bkdl,k,kdp->blp', convolutions, self.eigenvalues, self.M)
