import torch
import torch.nn as nn

from hankel_spectra import load_or_compute_eigen_data

def precompute_k_f(k):
    L = k.shape[-1]
    fft_size = 2*L
    assert k.dtype != torch.float16 and k.dtype != torch.bfloat16, "Too much quality loss with fp16"
    k_f = torch.fft.rfft(k, n=fft_size) / fft_size
    k_f = k_f.unsqueeze(0).unsqueeze(2)
    return k_f

# This version has a precomputed k_f from precompute_k_f()
def fft_causal_conv_fftk(u, k_f):
    L = u.shape[-1]
    fft_size = 2*L

    u_f = torch.fft.rfft(u.to(torch.float), n=fft_size)
    u_f = u_f.unsqueeze(1)

    assert k_f.shape[-1] == fft_size//2+1, "fft_causal_conv_fftk: fft_k shape should be L//2+1"

    prod = u_f * k_f

    y = torch.fft.irfft(prod, n=fft_size, norm="forward")[..., :L]
    return y

# Input shapes: u [B, D, L], k [H, L]
# y = u * k (see convolutions_test.py for naive reference version)
# k.dtype should be float
# FP16 does not support odd sizes, has terrible accuracy, and it is also not any faster.
def fft_causal_conv(u, k):
    return fft_causal_conv_fftk(u, precompute_k_f(k))

class ConvolutionLayer(nn.Module):
    def __init__(self, D_in, D_out, L, K):
        super(ConvolutionLayer, self).__init__()

        self.D_in = D_in
        self.D_out = D_out
        self.L = L
        self.K = K

        # Note the 2024 version only has a positive part to compute, so this code only works with that
        eigenvalues, eigenvectors = load_or_compute_eigen_data(L=L, K=K, matrix_version="2024")

        # Store k eigenvalues^^(1/4)
        self.eigenvalues = nn.Parameter(torch.tensor(eigenvalues, dtype=torch.float).pow(0.25), requires_grad=False)

        eigenvectors = torch.tensor(eigenvectors, dtype=torch.float) # [k, L]
        self.eigenvector_k_f = nn.Parameter(precompute_k_f(eigenvectors), requires_grad=False)

        # K parallel D_in -> D_out learned projections
        self.M = nn.Parameter(torch.Tensor(K, D_in, D_out))
        nn.init.xavier_uniform_(self.M)

    def forward(self, u):
        # Ensure input shape is [B, D, L]
        assert u.dim() == 3, "Input shape must be [B, D, L]"
        B, D_in, L = u.shape
        assert self.D_in == D_in, "Unexpected input dimension"
        assert self.L == L, "Sequence length change is unsupported"

        convolutions = fft_causal_conv_fftk(u, self.eigenvector_k_f)

        print(f"B={B} D_in={D_in} D_out={self.D_out} L={L} convolutions.shape={convolutions.shape}")

        # Combined operations:
        # 1. Scale by k eigenvalues
        # 2. Permute from [B, k, D_in, L] to [B, k, L, D_in]
        # 3. Apply M_k to project from D_in to D_out with K different projection matrices
        # 4. Sum across k to get [B, L, D_out]
        # 5. Permute from [B, L, D] to [B, D, L]
        result = torch.einsum('bkdl,k,kdp->bpl', convolutions, self.eigenvalues, self.M)

        print(f"result.shape={result.shape}")

        return result
