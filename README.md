# PyTorch MNIST Project

A clean, optimized implementation for MNIST dataset handling using PyTorch.

## 🚀 Features

- **Simple API**: Single class handles everything
- **Lazy Loading**: Datasets and loaders created on-demand
- **Property-based**: Clean access via properties
- **Minimal Dependencies**: Only essential imports
- **Ready for ML**: Perfect foundation for neural networks

## 📁 Project Structure

```
pytorch_learn/
├── README.md              # Project documentation
├── src/
│   └── mnist_handler.py   # Optimized MNIST handler
├── data/                  # MNIST dataset (auto-downloaded)
└── .gitignore            # Project gitignore
```

## 🛠️ Installation

```bash
# Create conda environment
conda create -n pytorch_learn_env python=3.8
conda activate pytorch_learn_env

# Install dependencies
conda install pytorch torchvision torchaudio cpuonly -c pytorch
conda install matplotlib
```

## 🎯 Usage

### Quick Start

```bash
python src/mnist_handler.py
```

### Programmatic Usage

```python
from src.mnist_handler import MNIST

# Initialize handler
mnist = MNIST(batch_size=64)

# Access datasets and loaders
train_loader = mnist.train_loader
test_loader = mnist.test_loader

# Get sample batch
images, labels = mnist.sample_batch()

# Show dataset info
info = mnist.info()
```

## 📊 What It Provides

- **Dataset Loading**: Automatic MNIST download and preprocessing
- **Data Loaders**: Efficient batched loading for training
- **Visualization**: Sample image grid generation  
- **Info**: Quick dataset statistics

## 🔧 Key Features

### Simple API
```python
mnist = MNIST()
train_loader = mnist.train_loader  # Lazy loaded
images, labels = mnist.sample_batch()
```

### Efficient Design
- Lazy loading (only loads when needed)
- Property-based access
- Minimal memory footprint
- Standard preprocessing pipeline

### Ready for Neural Networks
The handler provides exactly what you need for training:
- Normalized tensors (-1 to 1 range)
- Proper batch dimensions
- Shuffled training data
- Consistent preprocessing

## 📈 Next Steps

This foundation enables:
- Building neural networks
- Training on MNIST
- Experimenting with architectures
- Adding custom preprocessing
