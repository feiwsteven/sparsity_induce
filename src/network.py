import torch
import torch.nn as nn

from torch import Tensor


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


class ResNet(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        tau: float,
        m: float,
        residual_net=False,
    ) -> None:
        super(ResNet, self).__init__()
        Crelu = ClipReLu(tau, m)
        self.layers = nn.Sequential(nn.Linear(input_size, output_size), Crelu)
        self.x_layer = nn.Linear(input_size, output_size, bias=False)
        self.residual_net = residual_net

    def forward(self, x: Tensor):
        if self.residual_net:
            return self.layers(x) + x
        else:
            return self.layers(x)


class DeepNet(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        middle_layer_size: int,
        n_middle_layers: int,
        tau: float,
        m: float,
        self_attention=False,
        normalization=False,
        residual_net=False,
    ) -> None:
        super(DeepNet, self).__init__()
        crelu = ClipReLu(tau, m)
        self.input_layers = nn.Sequential(
            nn.Linear(input_size, middle_layer_size), crelu
        )
        self.self_attention = self_attention
        self.residual_net = residual_net
        self.attn_layers = nn.MultiheadAttention(input_size, 2)
        self.layers = nn.ModuleList()
        resnet_layer = ResNet(
            middle_layer_size, middle_layer_size, tau, m, residual_net=residual_net
        )
        # Add 48 hidden layers
        for _ in range(n_middle_layers):
            if normalization:
                self.layers.append(nn.LayerNorm(middle_layer_size))
            self.layers.append(resnet_layer)
            self.layers.append(crelu)

        # Last layer (hidden to output)
        self.layers.append(nn.Linear(middle_layer_size, output_size))

    def forward(self, x: Tensor) -> Tensor:
        # return self.layers(x.view(x.size(0), -1))
        # return self.layers(x.reshape(x.size(0), -1))

        temp = x.reshape(x.size(0), -1)
        if self.self_attention:
            x, _ = self.attn_layers(temp, temp, temp)
            x = self.input_layers(x)
        else:
            x = self.input_layers(temp)
        for layer in self.layers:
            x = layer(x)

        return x
