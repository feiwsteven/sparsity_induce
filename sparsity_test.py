import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch import Tensor
from torch.nn.modules import Linear


class ClipReLu(nn.Module):
    def __init__(self, tau: float, m: float) -> None:
        super(ClipReLu, self).__init__()
        self.tau = tau
        self.m = m

    def forward(self, x: Tensor) -> Tensor:
        crelu = torch.zeros_like(x)
        crelu[self.tau >= x] = 0
        crelu[(x > self.tau) & (x <= self.tau + self.m)] = (
            x[(x > self.tau) & (x <= self.tau + self.m)] - self.tau
        )

        crelu[x > self.tau + self.m] = self.m
        return crelu


class DeepNet(nn.Module):
    def __init__(self, input_size: int, output_size: int, tau: float, m: float) -> None:
        super(DeepNet, self).__init__()
        CReLU = ClipReLu(tau, m)
        self.layers = nn.Sequential(nn.Linear(input_size, 50), CReLU)
        # Add 48 hidden layers
        for _ in range(48):
            self.layers.add_module("linear", nn.Linear(50, 50))
            self.layers.add_module("cliprelu", CReLU)

        # Last layer (hidden to output)
        self.layers.add_module("output", nn.Linear(50, output_size))

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


if __name__ == "__main__":

    # create custom dataset
    x = torch.linspace(-5, 5, 100)
    k = ClipReLu(1, 2)
    y = k(x)

    # plot the softplus function graph
    plt.plot(x, y)
    plt.grid(True)
    plt.title("Softplus Function")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

    model = DeepNet(100, 100, 0.6, 1.2)

    c_y = model(x)
    # plot the softplus function graph
    plt.plot(x, c_y.detach().numpy())
    plt.grid(True)
    plt.title("Softplus Function")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

