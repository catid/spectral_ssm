import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from spectral_ssm import SpectralSSM

class SimpleRNNModel(nn.Module):
    def __init__(self, d_in, d_hidden, d_out, num_layers=2):
        super(SimpleRNNModel, self).__init__()
        self.d_in = d_in
        self.d_hidden = d_hidden
        self.d_out = d_out
        
        # Embedding Layer
        self.proj_in = nn.Linear(d_in, d_hidden, bias=False)
        
        # RNN Layer
        self.rnn = nn.RNN(input_size=d_hidden, hidden_size=d_hidden, num_layers=num_layers, batch_first=True)
        
        # Dense (output) Layer
        self.proj_out = nn.Linear(d_hidden, d_out, bias=False)
    
    def forward(self, u):
        # u shape: [B, L, D_in]
        B, L, D = u.shape
        
        # Project inputs to hidden dimension
        u_proj = self.proj_in(u)  # [B, L, d_hidden]
        
        # RNN forward pass
        rnn_out, _ = self.rnn(u_proj)  # [B, L, d_hidden]
        
        # Project RNN outputs to desired output dimension
        y = self.proj_out(rnn_out)  # [B, L, d_out]
        
        return y

# Toy dataset: sine wave prediction
def generate_sine_wave(B, L):
    L += 1

    # Generate random frequencies in some range, e.g., 0.1 to 1.0
    frequencies = torch.rand(B, 1) * 0.9 + 0.1  # [B, 1] frequencies in range [0.1, 1.0]
    
    # Generate random phase shifts in the range [0, 2π]
    phases = torch.rand(B, 1) * 2 * np.pi  # [B, 1] phases in range [0, 2π]
    
    # Generate a time sequence from 0 to 2π with L samples
    t = torch.linspace(0, 2 * np.pi, L).unsqueeze(0).repeat(B, 1)  # [B, L]
    
    # Generate sine waves with different frequencies and phase shifts
    data = torch.sin(frequencies * t + phases).unsqueeze(-1)  # [B, L, D]

    data = data.permute(0, 2, 1) # Convert to [B, D, L]

    # Return input and target: use all but last as input, and all but first as target
    return data[:, :, :-1], data[:, :, 1:]

# Hyperparameters
d_in = 1
d_hidden = 64
d_out = 1
B = 100
L = 256
learning_rate = 0.001
num_epochs = 1000

# Initialize the model
model = SpectralSSM(d_in, d_hidden, d_out, L, num_layers=1)
#model = SimpleRNNModel(d_in, d_hidden, d_out)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

# Generate toy data
x, y = generate_sine_wave(B, L)

x = x.permute(0, 2, 1) # Convert to [B, L, D]
y = y.permute(0, 2, 1) # Convert to [B, L, D]

print(f"x.shape={x.shape}")
print(f"y.shape={y.shape}")

# Split data into training and validation
split_idx = int(B * 0.8)
x_train, y_train = x[:split_idx], y[:split_idx]
x_val, y_val = x[split_idx:], y[split_idx:]

print(f"x_train.shape={x_train.shape}")
print(f"y_train.shape={y_train.shape}")

print(f"x_val.shape={x_val.shape}")
print(f"y_val.shape={y_val.shape}")

# Training loop
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(x_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    # Validation
    model.eval()
    with torch.no_grad():
        val_outputs = model(x_val)
        val_loss = criterion(val_outputs, y_val)

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}, Val Loss: {val_loss.item()}")
