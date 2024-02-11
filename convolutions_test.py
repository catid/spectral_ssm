import torch
import torch.nn as nn
import numpy as np

from convolutions import ConvolutionLayer


import unittest

# Unit test class
class TestARSTULayer(unittest.TestCase):
    def test_forward_shape(self):
        # Parameters for the test
        dim = 1  # Dimension of the input, not used in this test but required for compatibility
        L = 16  # Length of the input sequence
        k = 4  # Number of eigenvalues/eigenvectors
        batch_size = 2  # Number of sequences in a batch

        # Initialize the layer
        arstu_layer = ConvolutionLayer(dim, L, k)

        # Create a dummy input tensor of shape [batch_size, L]
        input_tensor = torch.randn(batch_size, L)

        # Perform the forward pass
        output_tensor = arstu_layer(input_tensor)

        print(f"output_tensor = {output_tensor}")

        has_nan = torch.isnan(output_tensor).any()
        self.assertEqual(has_nan, False, f"Output is NaN")

        # Check the output shape
        expected_shape = (batch_size, L)
        self.assertEqual(output_tensor.shape, expected_shape, f"Output tensor shape should be {expected_shape}")

    # Additional tests can be added here to test other aspects of the ARSTULayer

torch.manual_seed(42)

# Run the tests
if __name__ == '__main__':
    unittest.main()
