# SpikeEdge Runtime

A C-based runtime for inference of Spiking Neural Networks (SNNs) trained in SNNTorch. SpikeEdge enables deployment of trained SNN models on resource-constrained embedded devices.

## Overview

SpikeEdge provides a C-based runtime that bridges the gap between Python-based SNN training and embedded system deployment. While PyTorch and SNNTorch are excellent for training, they introduce significant overhead for deployment on resource-constrained devices. SpikeEdge addresses this by:

- Exporting trained models from SNNTorch to JSON format
- Reconstructing and executing models in efficient C code
- Providing optimization strategies like spiking activity-based pruning

## Deployment Pipeline

The workflow separates training and inference:

1. **Training Phase**: Models are designed and trained in Python using PyTorch/SNNTorch
2. **Export Phase**: Trained models (architecture + weights) are exported to JSON
3. **Inference Phase**: JSON files are loaded and executed by the C runtime

This design combines the flexibility of Python research environments with the efficiency of low-level C execution.

## Runtime Architecture

SpikeEdge consists of three main components:

### Network Loader
- Parses JSON files containing network architecture and weights
- Dynamically allocates memory for layers
- Validates layer compatibility
- Reconstructs the network structure in C

### Dataset Loader
- Loads event-based data from binary or MATLAB files
- Performs optional denoising and stabilization
- Converts events to frames (temporal aggregation)
- Prepares data for network input

### Execution Pipeline
- Processes input frames sequentially through network layers
- Supports multiple layer types: Conv2d, Linear, MaxPool2d, Flatten, Spiking
- Implements Leaky Integrate-and-Fire (LIF) neuron model
- Accumulates spike counts for classification

## Requirements

### System Requirements
- GCC compiler (C99 or later)
- JSON-C library
- MATIO library (for .mat file support)
- POSIX threads (pthread)

### Installation

#### Windows (MSYS2/MinGW64)

1. Download and install MSYS2 from [https://www.msys2.org](https://www.msys2.org)
2. Open the **MSYS2 MSYS** terminal and update packages:
    ```bash
    pacman -Syu
    ```
3. Close and reopen the terminal, then update again if needed.
4. Open the **MSYS2 MinGW 64-bit** terminal and install MinGW-w64 toolchain and libraries:
    ```bash
    pacman -S --needed base-devel mingw-w64-x86_64-toolchain \
                   mingw-w64-x86_64-json-c \
                   mingw-w64-x86_64-matio
    ```
5. Verify installation:
    ```bash
    gcc --version
    make --version
    ```

#### Linux (Ubuntu/Debian)
```bash
sudo apt-get update
sudo apt-get install gcc libjson-c-dev libmatio-dev build-essential
```

#### MacOs
```
brew install json-c matio
```

### Building

Clone the repository and build:
```
git clone <repository-url>
cd snn-generator
make release
```

### Clean build artifacts:
```
make clean
```
### Usage
Basic Execution

Sample run:
```
./snn_generator snn_nmnist_architecture.json snn_nmnist_weights_bs_32.json input/nmnist/1.bin
```
*Note: In a real-world embedded implementation, the input data typically comes from a live event-based sensor. In this case, for demonstration purposes, a static .bin or .mat file with raw events is used.

### Optimization Strategies
- **Channel Pruning**: Removes less active channels in convolutional layers
- **Bidirectional Pruning**: Combines forward and backward pruning strategies

### References
- SNNTorch: Spiking Neural Networks in PyTorch
- PyTorch: Deep learning framework
