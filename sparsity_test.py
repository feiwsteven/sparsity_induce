import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim

from torchsummary import summary
from src.network import ClipReLu, DeepNet
from src.utility import custom_reshape


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

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
    print(f"Using device: {device}")

    # Load the MNIST dataset
    torch.manual_seed(1234)
    batch_size = 50
    learning_rate = 1e-4
    num_epochs = 10

    train_dataset = torch.load("./data/MNIST/train.pt")
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=5
    )

    test_dataset = torch.load("./data/MNIST/test.pt")
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=5
    )

    # Initialize the model, loss function, and optimizer
    model = DeepNet(
        28 * 28,
        10,
        16 * 25 * 8,
        2,
        0.05,
        10,
        self_attention=True,
        normalization=False,
        residual_net=False,
    )
    summary(model, (1, 28 * 28))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.view(images.size(0), -1)
            labels = labels
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}"
                )

    # Test the model
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            # images, labels = images.to(device), labels.to(device)  # Move the data to the GPU
            outputs = model(images)
            _, predicted = torch.max(outputs.view(outputs.size(0), -1), 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(
            f"Accuracy of the model on the 10000 test images: {100 * correct / total}%"
        )
