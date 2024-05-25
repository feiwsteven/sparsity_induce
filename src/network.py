import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch import Tensor

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def attention(p_mat, q_mat, z_mat, activation=None):
    B = z_mat.shape[0]
    N = z_mat.shape[1] - 1
    d = z_mat.shape[2] - 1
    p_full = torch.cat([p_mat, torch.zeros(1, d).to(device)], dim=0)
    p_full = torch.cat([p_full, torch.zeros(d + 1, 1).to(device)], dim=1)
    p_full[d, d] = 1
    q_full = torch.cat([q_mat, torch.zeros(1, d).to(device)], dim=0)
    q_full = torch.cat([q_full, torch.zeros(d + 1, 1).to(device)], dim=1)
    A = torch.eye(N + 1).to(device)
    A[N, N] = 0
    attn = torch.einsum("BNi, ij, BMj -> BNM", (z_mat, q_full, z_mat))
    if activation is not None:
        attn = activation(attn)
    key = torch.einsum("ij, BNj -> BNi", (p_full, z_mat))
    output = torch.einsum("BNM,ML, BLi -> BNi", (attn, A, key))
    return output / N


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
        crelu = ClipReLu(tau, m)
        self.layers = nn.Sequential(nn.Linear(input_size, output_size), crelu)
        self.residual_net = residual_net

    def forward(self, x: Tensor):
        out = self.layers(x)
        if self.residual_net:
            out += x
        return out


class LinearTransformer(nn.Module):
    def __init__(self, input_size: int, n_heads=1) -> None:
        super(LinearTransformer, self).__init__()
        self.n_heads = n_heads
        self.q_mat = nn.Parameter(torch.randn(input_size, input_size))
        self.p_mat = nn.Parameter(torch.randn(input_size, input_size))
        nn.init.xavier_uniform_(self.q_mat)
        nn.init.xavier_uniform_(self.p_mat)
 
 
    def forward(self, x: Tensor):
        n = x.shape[0]
        # attn = n by n
        x_norm = torch.norm(x, p=2, dim=2, keepdim=True)
        attn2 = torch.einsum("nqd,dj->nqj", [x/x_norm, self.q_mat])
        attn2 = torch.einsum("nqj,mqj->nm", [attn2, x/x_norm])

        attn = torch.matmul(
            torch.matmul(x / x_norm, self.q_mat), torch.transpose(x / x_norm, 0, 1)
        )
        # attn is a n by n matrix
        attn = torch.matmul(torch.matmul(attn, x), self.p_mat) / n + x
        return attn

class SelfLinearAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfLinearAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values_weight = nn.Parameter(torch.Tensor(self.head_dim, self.head_dim))
        self.keys_weight = nn.Parameter(torch.Tensor(self.head_dim, self.head_dim))
        self.queries_weight = nn.Parameter(torch.Tensor(self.head_dim, self.head_dim))
        self.fc_weight = nn.Parameter(torch.Tensor(embed_size, heads * self.head_dim))

        nn.init.xavier_uniform_(self.values_weight)
        nn.init.xavier_uniform_(self.keys_weight)
        nn.init.xavier_uniform_(self.queries_weight)
        nn.init.xavier_uniform_(self.fc_weight)

    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        query = query.reshape(N, query_len, self.heads, self.head_dim)

        values = torch.einsum("nlhd,df->nlhf", [values, self.values_weight])
        keys = torch.einsum("nlhd,df->nlhf", [keys, self.keys_weight])
        queries = torch.einsum("nlhd,df->nlhf", [query, self.queries_weight])

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )

        out = torch.einsum("nqd,df->nqf", [out, self.fc_weight])
        return out

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
        # self.attn_layers = nn.MultiheadAttention(input_size, 2)
        self.attn_linear_layers = LinearTransformer(input_size)
        self.attn_heads_layers = SelfLinearAttention(input_size, heads=2)
        self.layers = nn.ModuleList()
        resnet_layer = ResNet(
            middle_layer_size, middle_layer_size, tau, m, residual_net=residual_net
        )
        # Add 48 hidden layers
        for _ in range(n_middle_layers):
            if normalization:
                self.layers.append(nn.LayerNorm(middle_layer_size))
            self.layers.append(resnet_layer)
            # resnet_layer alreadys has a crelu activation.

        # Last layer (hidden to output)
        self.layers.append(nn.Linear(middle_layer_size, output_size))

    def forward(self, x: Tensor) -> Tensor:
        # return self.layers(x.view(x.size(0), -1))
        # return self.layers(x.reshape(x.size(0), -1))

        temp = x.reshape(x.size(0), -1)
        if self.self_attention:
            # x, _ = self.attn_layers(x, x, x)
            x = self.attn_linear_layers(x)
            x = self.attn_heads_layers(x, x, x, mask=None)
            # x = self.attn_layers(temp)
            x = self.input_layers(x)

        else:
            x = self.input_layers(temp)
        for layer in self.layers:
            x = layer(x)

        return x
