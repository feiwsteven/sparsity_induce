import torch

if __name__ == '__main'
a = torch.Tensor([[[1, 1, 1], [1, 1, 1]], [[2, 2, 2], [2, 2, 2]]])
b = torch.Tensor([[[1, 1, 1], [1, 1, 1]], [[2, 2, 2], [2, 2, 2]]])

print(a.shape)
print(torch.einsum('ijk, ifk->ijf', [a, b]))
