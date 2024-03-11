from torch import Tensor


def custom_reshape(x: Tensor):
    # Assuming input tensor shape is (C, H, W)
    # Reshape to (C, H*W)
    return x.view(x.size(0), -1)
