import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the neural network class
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Input layer to first hidden layer
        self.fc1 = nn.Linear(28 * 28, 128)  # First layer (input to hidden)
        # First hidden layer to second hidden layer
        self.fc2 = nn.Linear(128, 64)  # Adding another hidden layer (128 to 64)
        # Second hidden layer to output layer
        self.fc3 = nn.Linear(64, 10)  # Output layer (64 to 10 classes for digits)

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten the 28x28 images
        x = F.relu(self.fc1(x))   # Pass through first layer with ReLU activation
        x = F.relu(self.fc2(x))   # Pass through second hidden layer with ReLU
        x = self.fc3(x)           # Output layer, no activation (CrossEntropyLoss handles this)
        return x

# Instantiate the model
net = Net()

print(net)  # You can print the model architecture to verify the changes

