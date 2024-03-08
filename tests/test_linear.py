import unittest
import torch
from src.network import LinearTransformer


class TestMyLinearModule(unittest.TestCase):
    def test_forward_pass(self):
        input_size = 5
        output_size = 5
        batch_size = 10

        # Create an instance of the custom module
        model = LinearTransformer(input_size, output_size)

        # Create random input data
        x = torch.randn(batch_size, input_size)

        # Perform the forward pass
        y = model(x)

        # Check the output shape
        self.assertEqual(y.shape, (batch_size, output_size))


if __name__ == "__main__":
    unittest.main()
