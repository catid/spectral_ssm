import torch
import torch.nn as nn
import torch.nn.functional as F

from ar_stu import AR_STULayer

import unittest

class TestARSTULayer(unittest.TestCase):

    def setUp(self):
        self.d_in = 10  # Input dimension
        self.d_out = 5  # Output dimension
        self.L = 32  # Sequence length
        self.k = 16  # Number of eigenvectors/eigenvalues
        self.alpha = 0.9  # Scaling factor for My_2
        self.layer = AR_STULayer(self.d_in, self.d_out, self.L, self.k, self.alpha)

    def test_output_shape(self):
        """Test if the output shape is correct."""
        batch_size = 4
        input_tensor = torch.randn(batch_size, self.L, self.d_in)
        output = self.layer(input_tensor)
        expected_shape = (batch_size, self.L, self.d_out)
        self.assertEqual(output.shape, expected_shape, "Output shape is incorrect.")

    def test_output_with_previous_inputs(self):
        """Test the layer with previous inputs and outputs provided."""
        batch_size = 4
        u = torch.randn(batch_size, self.L, self.d_in)
        u_n1 = torch.randn(batch_size, self.L, self.d_in)
        u_n2 = torch.randn(batch_size, self.L, self.d_in)
        y_n1 = torch.randn(batch_size, self.L, self.d_out)
        y_n2 = torch.randn(batch_size, self.L, self.d_out)

        # Call the forward method with previous inputs and outputs
        output = self.layer(u, u_n1, u_n2, y_n1, y_n2)
        self.assertEqual(output.shape, (batch_size, self.L, self.d_out), "Output shape with previous inputs is incorrect.")

    def test_no_nan_in_output(self):
        """Ensure that the output does not contain NaN values."""
        batch_size = 4
        input_tensor = torch.randn(batch_size, self.L, self.d_in)
        output = self.layer(input_tensor)
        self.assertFalse(torch.isnan(output).any(), "Output contains NaN values.")

if __name__ == '__main__':
    unittest.main()
