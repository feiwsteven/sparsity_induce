import torch
import torch.nn as nn
import matplotlib.pyplot as plt 
 
from torch import Tensor



class ClipRelu(nn.Module):
    def __init__(self, tau: float, m: float) -> None:
        super(ClipRelu, self).__init__()
        self.tau = tau
        self.m = m

    def forward(self, x: Tensor) -> Tensor:
        crelu = torch.zeros_like(x)
        crelu[torch.ge(self.tau, x)] = 0
        crelu[torch.gt(x, self.tau) and torch.ge(self.tau + self.m, x)] = x -self.tau

        elif x >= self.tau and x < self.tau + self.m:
            crelu = x - self.tau
        else:
            crelu = self.m
        return crelu

if __name__ == '__main__':
    
 
    # create custom dataset 
    x = torch.linspace(-5, 5, 100) 
    k = ClipRelu(1, 2) 
    y = k(x) 
    
    # plot the softplus function graph 
    plt.plot(x, y) 
    plt.grid(True) 
    plt.title('Softplus Function') 
    plt.xlabel('x') 
    plt.ylabel('y') 
    plt.show()