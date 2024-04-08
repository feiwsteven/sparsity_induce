import torch

if __name__ == '__main__':

    a = torch.Tensor([[[1, 1, 1]], [[2, 2, 2]]])
    b = torch.Tensor([[[1, 1, 1]], [[2, 2, 2]]])

    print(a)
    print(a.shape)
    print(torch.einsum('ijk, ifk->ijf', [a, b]))
    print(torch.einsum('ijk, qjk->iq', [a, b]))

