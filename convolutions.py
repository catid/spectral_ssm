import torch
import torch.nn as nn

from hankel_spectra import load_or_compute_eigen_data

class ConvolutionLayer(nn.Module):
    def __init__(self, dim, L, k):
        super(ConvolutionLayer, self).__init__()

        self.L = L
        self.k = k

        # Note the 2024 version only has a positive part to compute, so this code only works with that
        eigenvalues, eigenvectors = load_or_compute_eigen_data(L=L, k=k, matrix_version="2024")

        # Store k eigenvalues^^(1/4)
        self.eigenvalues = nn.Parameter(torch.tensor(eigenvalues, dtype=torch.float).pow(0.25), requires_grad=False)

        eigenvectors = torch.tensor(eigenvectors, dtype=torch.float)  # [k, L]
        self.eigenvector_ffts = nn.Parameter(torch.fft.rfft(eigenvectors), requires_grad=False)  # [k, L//2+1]

        self.M = nn.Parameter(torch.Tensor(k, L//2+1))
        nn.init.xavier_uniform_(self.M)

    def forward(self, u):
        L = u.size(-1)
        assert self.L == L, "FIXME: Currently only works for fixed-length sequences"

        u_fft = torch.fft.rfft(u) # [B, L//2+1]

        # Multiply u_fft by each eigenvector producing [B, k, L//2+1]
        p_fft = torch.einsum('bi,kj->bkj', u_fft, self.eigenvector_ffts)

        convolutions = torch.fft.irfft(p_fft, n=L)  # [B, k, L]

        # Compute the sum of convolutions*eigenvalues^^(1/4)
        result = torch.einsum('bkl,k->bl', convolutions, self.eigenvalues)

        return result
