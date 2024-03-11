import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
import os
from src.utility import custom_reshape

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


if __name__ == "__main__":
    print(f"Using device: {device}")

    # Load the MNIST dataset
    # Change the dimension of data from (*, 1, 28 * 28) to (*, 28 * 28)
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
            transforms.Lambda(custom_reshape),
        ]
    )

    train_dataset = torchvision.datasets.MNIST(
        root="./data", train=True, transform=transform, download=True
    )

    test_dataset = torchvision.datasets.MNIST(
        root="./data", train=False, transform=transform, download=True
    )

    torch.save(train_dataset, os.path.join("./data/MNIST", "train.pt"))
    torch.save(test_dataset, os.path.join("./data/MNIST", "test.pt"))

I
