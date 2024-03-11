import torch
import unittest

class TestMps(unittest.TestCase):
    def test_mps(self):
        input_size = 5
        output_size = 5
        batch_size = 10

        # Create an instance of the custom module
        x = torch.tensor([1, 2, 3], device='mps')

        print(x)
        # Check the output shape
        self.assertEqual(x.shape[0], 3)


if __name__ == "__main__":
    unittest.main()
