import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch import Tensor
from torch.nn.modules import Linear
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms


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
    def __init__(self, input_size: int, output_size: int, n_middle_layers: int, tau: float, m: float) -> None:
        super(DeepNet, self).__init__()
        CReLU = ClipReLu(tau, m)
        self.layers = nn.Sequential(nn.Linear(input_size, 50), CReLU)
        # Add 48 hidden layers
        for _ in range(n_middle_layers):
            self.layers.add_module("linear", nn.Linear(50, 50))
            self.layers.add_module("cliprelu", CReLU)

        # Last layer (hidden to output)
        self.layers.add_module("output", nn.Linear(50, output_size))

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x.view(x.size(0), -1))


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

    # model = DeepNet(100, 100, 0.6, 1.2)

    # c_y = model(x)
    # # plot the softplus function graph
    # plt.plot(x, c_y.detach().numpy())
    # plt.grid(True)
    # plt.title("Softplus Function")
    # plt.xlabel("x")
    # plt.ylabel("y")
    # plt.show()

    # Check if GPU is available
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f'Using device: {device}')

    # Load the MNIST dataset
    batch_size = 50
    learning_rate = 1e-4
    num_epochs = 10
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    train_dataset = torchvision.datasets.MNIST(
        root="./data", train=True, transform=transform, download=True
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )

    test_dataset = torchvision.datasets.MNIST(
        root="./data", train=False, transform=transform, download=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )

    # Initialize the model, loss function, and optimizer
    model = DeepNet(28*28, 10, 10, 0.0, 100000)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)  # Move the data to the GPU
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')

    # Test the model
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)  # Move the data to the GPU
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f'Accuracy of the model on the 10000 test images: {100 * correct / total}%')