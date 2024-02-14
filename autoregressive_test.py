import unittest
import torch
import torch.nn.init as init

from autoregressive import AutoRegressiveCausalLayer

def naive_auto_regressive_causal_model(x, D_out, weights, K=3):
    B, D_in, L = x.shape
    y = torch.zeros(B, D_out, L, dtype=x.dtype, device=x.device)

    # Apply the weights to simulate the convolution operation
    for t in range(L):
        for d_out in range(D_out):
            for b in range(B):
                weighted_sum = 0.0
                for k in range(K):
                    idx = K - 1 - k
                    if t-k >= 0:
                        # Multiply the input by the corresponding weight
                        weighted_sum += (x[b, :, t-k] * weights[d_out, :, idx]).sum()
                
                y[b, d_out, t] = weighted_sum

    return y

class TestAutoRegressiveCausalLayer(unittest.TestCase):
    def test_model_output(self):
        B, D_in, L = 1, 2, 16
        D_out = 4
        K=3

        model = AutoRegressiveCausalLayer(D_in, D_out, K)
        init.xavier_uniform_(model.conv.weight)

        # Should be [D_out, D_in, K]
        print(f"B={B} D_in={D_in} D_out={D_out} K={K} L={L} model.conv.weight.shape = {model.conv.weight.shape}")
        expected_shape = (D_out, D_in, K)
        actual_shape = model.conv.weight.shape
        self.assertEqual(actual_shape, expected_shape,
                         msg=f"Convolutional layer weight shape is incorrect, expected {expected_shape}, got {actual_shape}.")

        # Create a random tensor for x
        x = torch.rand(B, D_in, L)

        # Generate model output
        y = model(x)

        # Generate naive implementation output
        z = naive_auto_regressive_causal_model(x, D_out, model.conv.weight, K)

        # Test if y and z are close within tolerance
        self.assertTrue(torch.allclose(y, z, rtol=1e-05, atol=1e-06),
                        msg="The outputs of the AutoRegressiveCausalLayer and the naive implementation do not match within tolerance.")

    def test_causality(self):
        B, D_in, L = 1, 2, 16
        D_out = 4
        K = 3

        model = AutoRegressiveCausalLayer(D_in, D_out, K)
        init.xavier_uniform_(model.conv.weight)

        # Create a base input tensor filled with zeros
        x_base = torch.zeros(B, D_in, L)
        
        # Copy the base tensor and modify the last element
        x_modified = x_base.clone()
        x_modified[:, :, -1] = 1  # Change only the last element
        
        # Get the output from the model for both the base and modified inputs
        y_base = model(x_base)
        y_modified = model(x_modified)
        
        # Check that the outputs are identical except for the last position
        # This verifies that the model is causal
        difference = (y_base != y_modified).float()
        expected_difference = torch.zeros_like(difference)
        expected_difference[:, :, -1] = 1  # Expect difference only at the last position
        
        # Assert that the difference matches the expected pattern
        self.assertTrue(torch.equal(difference, expected_difference),
                        msg="The model is not causal. Output changes are observed in positions other than the last.")

if __name__ == '__main__':
    unittest.main()
