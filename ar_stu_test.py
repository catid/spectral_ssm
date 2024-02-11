import torch
import torch.nn as nn
import torch.nn.functional as F

from convolutions import ConvolutionLayer

class ARSTULayer(nn.Module):
    def __init__(self, dim, L, k, ky, d_in, d_out):
        super(ARSTULayer, self).__init__()
        self.dim = dim
        self.L = L
        self.k = k
        self.ky = ky
        self.d_in = d_in
        self.d_out = d_out

        # Parameters for the auto-regressive component
        self.My = nn.ParameterList([nn.Parameter(torch.Tensor(d_out, d_out)) for _ in range(ky)])
        self.Mu = nn.ParameterList([nn.Parameter(torch.Tensor(d_in, d_out)) for _ in range(3)])

        # Initialize parameters
        for p in self.My:
            nn.init.xavier_uniform_(p)
        for p in self.Mu:
            nn.init.xavier_uniform_(p)
        
        # Assuming ConvolutionLayer processes the spectral component
        self.convolution_layer = ConvolutionLayer(d_in, d_out, L, k)

    def forward(self, u):
        # u shape: [B, L, d_in]

        # Auto-regressive component
        y_autoreg = torch.zeros(u.shape[0], self.L, self.d_out, device=u.device)
        for i in range(1, self.ky + 1):
            if i <= u.shape[1]:  # Check to avoid index error
                y_autoreg[:, i:, :] += F.linear(y_autoreg[:, i-1:-1, :], self.My[i-1])
        
        # Process input u for the Mu component
        for i in range(3):
            if i < u.shape[1]:
                y_autoreg[:, i+1:, :] += F.linear(u[:, i:-1-i, :], self.Mu[i])

        # Spectral component (processed by ConvolutionLayer)
        y_spectral = self.convolution_layer(u)

        # Combine auto-regressive and spectral components
        y = y_autoreg + y_spectral

        return y
