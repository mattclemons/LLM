import torch

# Create a tensor
x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
print(x)

# Add tensors
y = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
z = x + y
print(z)
