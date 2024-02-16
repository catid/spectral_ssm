import unittest
import torch
import torch.nn.init as init

from autoregressive import AutoRegressiveCausalInput, AutoRegressiveCausalOutput

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

        model = AutoRegressiveCausalInput(D_in, D_out, K)

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
                        msg="The outputs of the AutoRegressiveCausalInput and the naive implementation do not match within tolerance.")

    def test_causality(self):
        B, D_in, L = 1, 7, 2048
        D_out = 13
        K = 3

        model = AutoRegressiveCausalInput(D_in, D_out, K)

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

class TestAutoRegressiveCausalOutput(unittest.TestCase):
    def setUp(self):
        # Initialize module parameters
        self.D = 3  # Dimensionality of the input
        self.L = 10  # Length of the sequence
        self.B = 2  # Batch size
        self.Ky = 2  # Kernel size
        self.alpha = 0.9  # Weight for the autoregressive term

        # Create a test input tensor
        self.x = torch.rand(self.B, self.D, self.L)

        # Initialize the module
        self.module = AutoRegressiveCausalOutput(D=self.D, Ky=self.Ky, alpha=self.alpha)

    def test_initialization(self):
        # Verify that the last convolutional weight is initialized correctly
        expected_last_weight = torch.eye(self.D) * self.alpha
        actual_last_weight = self.module.conv.weight.data[:, :, -2]
        self.assertTrue(torch.allclose(expected_last_weight, actual_last_weight), "Initialization does not match expected values")

    def test_output_shape(self):
        # Verify the output shape is as expected
        y = self.module(self.x)
        expected_shape = (self.B, self.D, self.L)
        self.assertEqual(y.shape, expected_shape, "Output shape is incorrect")

    def test_causality(self):
        # Verify causality by ensuring output at time t does not depend on inputs at time t+1, t+2, ...
        x_modified = self.x.clone()
        x_modified[:, :, -1] = torch.zeros(self.B, self.D)  # Modify the last time step

        y_original = self.module(self.x)
        y_modified = self.module(x_modified)

        # The outputs should be identical except for the last step if the module is causal
        self.assertTrue(torch.allclose(y_original[:, :, :-1], y_modified[:, :, :-1]), "Module is not causal")

    def test_weight_application(self):
        D = 2  # Number of features
        L = 5  # Sequence length
        B = 1  # Batch size
        Ky = 3  # Kernel size, implying the auto-regressive sum includes the current and two previous terms
        alpha = 0.9  # Weight for the auto-regressive term

        # Initialize the module with the correct Ky
        module = AutoRegressiveCausalOutput(D, Ky, alpha)

        # Create a sample input tensor [B, D, L]
        x = torch.randn(B, D, L)

        # Compute the output
        y = module(x)

        # Manually compute expected output for comparison, considering Ky
        expected_y = x.clone()  # Start with x and then add the auto-regressive components
        for t in range(1, L):
            for k in range(1, min(Ky, t+1)):  # Ensure we don't go beyond the current time step
                if k == Ky - 1:  # Apply the alpha weight only to the term at Ky-1 (as per initialization)
                    expected_y[:, :, t] += alpha * x[:, :, t-k]

        # Check if the computed output matches the expected output
        self.assertTrue(torch.allclose(y, expected_y, atol=1e-5), "The auto-regressive output does not match the expected output.")

if __name__ == '__main__':
    unittest.main()
