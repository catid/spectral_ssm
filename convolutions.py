import torch
import torch.nn as nn

from hankel_spectra import load_or_compute_eigen_data

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
        eigenvector_ffts = nn.Parameter(torch.fft.rfft(eigenvectors), requires_grad=False)  # [k, L//2+1]
        self.eigenvector_ffts_expanded = eigenvector_ffts.unsqueeze(0).unsqueeze(2)  # [1, k, 1, L//2+1]

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

        # Perform FFT on each sequence in the batch for all features [B, D, L//2+1]
        u_fft = torch.fft.rfft(u_transposed)

        # u_fft is [B, D, L//2+1], so we want to introduce the 'k' dimension
        u_fft_expanded = u_fft.unsqueeze(1)  # [B, 1, D, L//2+1]

        p_fft = u_fft_expanded * self.eigenvector_ffts_expanded  # [B, k, D, L//2+1]

        # Apply IFFT to convert back to time domain, resulting in [B, k, D, L]
        convolutions = torch.fft.irfft(p_fft)

        # Combined operations:
        # 1. Scale by k eigenvalues
        # 2. Permute from [B, k, D, L] to [B, k, L, D]
        # 3. Apply M_k to project from D to d_out with K different projection matrices
        # 4. Sum across k to get [B, L, d_out]
        return torch.einsum('bkdl,k,kdp->blp', convolutions, self.eigenvalues, self.M)
