import torch
import torchvision
import torchvision.transforms as transforms

# Download the MNIST dataset
transform = transforms.Compose([transforms.ToTensor()])
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

import torch.nn as nn
import torch.nn.functional as F

# Define the neural network class
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Input layer to hidden layer
        self.fc1 = nn.Linear(28 * 28, 128)  # 28x28 is the size of the MNIST images
        # Hidden layer to output layer
        self.fc2 = nn.Linear(128, 10)  # 10 output classes (digits 0-9)

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten the 28x28 images
        x = F.relu(self.fc1(x))   # Apply ReLU activation
        x = self.fc2(x)           # Output layer
        return x

# Instantiate the model
net = Net()


import torch.optim as optim

# Define a loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)


# Training loop
for epoch in range(2):  # Loop over the dataset twice
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = net(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Print statistics
        running_loss += loss.item()
        if i % 100 == 99:  # Print every 100 mini-batches
            print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 100}")
            running_loss = 0.0

print("Finished Training")

self.fc3 = nn.Linear(128, 64)  # Adding another layer

