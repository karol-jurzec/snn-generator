import torch
import torch.nn as nn
import numpy as np

def read_input(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    data = [float(line.strip()) for line in lines[1:]]  # Skip the 'Layer' text
    return torch.tensor(data, dtype=torch.float32)

def read_weights(filename, in_channels, out_channels, kernel_size):
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    data = [float(line.strip()) for line in lines[1:]]  # Skip 'Layer' header
    weight_size = out_channels * in_channels * kernel_size * kernel_size
    weights = torch.tensor(data[:weight_size], dtype=torch.float32)
   # biases = torch.tensor(data[weight_size:weight_size + out_channels], dtype=torch.float32)
    biases = torch.randn(out_channels) * 0.1  # Initialize biases with a normal distribution, scaled by 0.1

    weights = weights.view(out_channels, in_channels, kernel_size, kernel_size)
    return weights, biases

def save_output(filename, tensor):
    with open(filename, 'w') as f:
        f.write("Layer\n")
        for value in tensor.view(-1):
            f.write(f"{value.item():.6f}\n")

class ConvModel(nn.Module):
    def __init__(self, weights, biases):
        super(ConvModel, self).__init__()
        self.conv = nn.Conv2d(12, 32, kernel_size=(5, 5), stride=(1, 1))
        self.conv.weight = nn.Parameter(weights)
        self.conv.bias = nn.Parameter(biases)
        
    def forward(self, x):
        return self.conv(x)

# Load input
input_file = "/Users/karol/Documents/repos/snn_generator/tests/inputs_171.txt"  # Change if necessary
input_data = read_input(input_file)

print(f"Data shape {input_data.shape}")
print("Input data was read...")

# Reshape input to (batch_size=1, channels=12, height=15, width=15)
input_data = input_data.view(1, 12, 15, 15)

# Load weights and biases
weights_file = "/Users/karol/Documents/repos/snn_generator/tests/weights_171.txt"
weights, biases = read_weights(weights_file, in_channels=12, out_channels=32, kernel_size=5)

print("Wiehgts data was read")


# Initialize model with loaded weights
model = ConvModel(weights, biases)

print("Model is initialized")

# Forward pass
output = model(input_data)

print("Values were predicted...")

# Save output
output_file = "output.txt"
save_output(output_file, output)
