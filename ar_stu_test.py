import unittest
import torch
import torch.nn as nn

from ar_stu import AR_STULayer  # Adjust import as necessary

class TestAR_STULayer(unittest.TestCase):
    def setUp(self):
        # Example dimensions
        self.D_in = 2
        self.D_out = 4
        self.L = 10
        self.K = 16
        self.alpha = 0.9

        # Initialize the AR_STULayer
        self.layer = AR_STULayer(self.D_in, self.D_out, self.L, K=self.K, alpha=self.alpha)

    def test_subcomponents_initialized(self):
        # Check if subcomponents are instances of the expected classes
        self.assertIsInstance(self.layer.autoregressive_u, nn.Module,
                              "AutoRegressiveCausalLayer is not initialized correctly.")
        self.assertIsInstance(self.layer.convolution_layer, nn.Module,
                              "ConvolutionLayer is not initialized correctly.")

    def test_output_shape(self):
        # Create a dummy input tensor
        u = torch.randn(1, self.D_in, self.L)  # Batch size of 1 for simplicity

        # Forward pass
        y = self.layer(u)

        # Expected output shape
        expected_shape = (1, self.D_out, self.L)

        # Check the output shape
        self.assertEqual(y.shape, expected_shape,
                         f"Output shape is incorrect. Expected {expected_shape}, got {y.shape}.")

    # Add more tests as needed, for example, to verify the behavior of the layer
    # under specific input conditions or to check the properties of the layer's parameters.

if __name__ == '__main__':
    unittest.main()
