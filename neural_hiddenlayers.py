import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the neural network class with 2 hidden layers
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Input layer to first hidden layer (128 neurons)
        self.fc1 = nn.Linear(28 * 28, 128)  # 784 inputs (28x28 image) -> 128 neurons
        # First hidden layer to second hidden layer (64 neurons)
        self.fc2 = nn.Linear(128, 64)  # 128 neurons -> 64 neurons
        # Second hidden layer to output layer (10 neurons for 10 digit classes)
        self.fc3 = nn.Linear(64, 10)  # 64 neurons -> 10 neurons (for digits 0-9)

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten the 28x28 images into a 1D vector
        x = F.relu(self.fc1(x))  # Apply ReLU activation to the first hidden layer
        x = F.relu(self.fc2(x))  # Apply ReLU activation to the second hidden layer
        x = self.fc3(x)          # Output layer (no activation, as it's handled by loss function)
        return x

