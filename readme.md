# SpikeEdge Runtime

A lightweight C-based runtime for efficient inference of Spiking Neural Networks (SNNs) trained in SNNTorch. SpikeEdge enables deployment of trained SNN models on resource-constrained embedded devices.

## Overview

SpikeEdge provides a C-based runtime that bridges the gap between Python-based SNN training and embedded system deployment. While PyTorch and SNNTorch are excellent for training, they introduce significant overhead for deployment on resource-constrained devices. SpikeEdge addresses this by:

- Exporting trained models from SNNTorch to JSON format
- Reconstructing and executing models in efficient C code
- Supporting multiple event-based datasets (NMNIST, STMNIST, N-CARS, DVS Gesture)
- Providing optimization strategies like spiking activity-based pruning

## Deployment Pipeline

The workflow separates training and inference:

1. **Training Phase**: Models are designed and trained in Python using PyTorch/SNNTorch
2. **Export Phase**: Trained models (architecture + weights) are exported to JSON
3. **Inference Phase**: JSON files are loaded and executed by the C runtime on target hardware

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

#### Windows (MSYS2/MinGW)
```bash
pacman -S mingw-w64-x86_64-gcc \
          mingw-w64-x86_64-json-c \
          mingw-w64-x86_64-matio
```

#### Linux (Ubuntu/Debian)
```bash
sudo apt-get update
sudo apt-get install gcc libjson-c-dev libmatio-dev build-essential
```

#### macOS
```bash
brew install json-c matio
```

## Building

Clone the repository and build:

```bash
git clone <repository-url>
cd snn-generator
make
```

For optimized release build:
```bash
make release
```

Clean build artifacts:
```bash
make clean
```

## Usage

### Basic Execution

Run the default test:
```bash
./snn_generator
```

### Performance Profiling

Enable performance profiling:
```bash
./snn_generator --perf
```

### Configuring Dataset Paths

Set environment variables for dataset paths:

**Windows (PowerShell):**
```powershell
$env:NMNIST_TEST_PATH = "C:/path/to/NMNIST/Test"
$env:STMNIST_TEST_PATH = "C:/path/to/STMNIST/data_submission"
```

**Linux/macOS:**
```bash
export NMNIST_TEST_PATH="/path/to/NMNIST/Test"
export STMNIST_TEST_PATH="/path/to/STMNIST/data_submission"
```

### Model Files

Place your model files in the project root:
- Architecture files: `*_architecture.json`
- Weight files: `*_weights_*.json`

Example files included:
- `snn_nmnist_architecture.json` / `snn_nmnist_weights_bs_32.json`
- `scnn_stmnist_architecture.json` / `scnn_stmnist_weights_bs_64.json`

## Supported Datasets

- **NMNIST**: Neuromorphic MNIST dataset
- **STMNIST**: Spiking Temporal MNIST dataset

## Project Structure
