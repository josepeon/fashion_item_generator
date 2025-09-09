# PyTorch MNIST Project

A clean, production-ready implementation for MNIST dataset handling, analysis, and preprocessing using PyTorch.

## 🚀 Features

- **Modular Design**: Clean class-based architecture
- **Type Hints**: Full type annotation support
- **Logging**: Professional logging implementation
- **Performance Monitoring**: Built-in benchmarking
- **Visualization**: Automated sample image generation
- **Error Handling**: Robust exception management

## 📁 Project Structure

```
pytorch_learn/
├── README.md                    # Project documentation
├── src/
│   └── mnist_data_handler.py    # Main MNIST data handling module
├── data/                        # MNIST dataset (auto-downloaded)
└── .gitignore                  # Project gitignore
```

## �️ Installation

```bash
# Create conda environment
conda create -n pytorch_learn_env python=3.8
conda activate pytorch_learn_env

# Install dependencies
conda install pytorch torchvision torchaudio cpuonly -c pytorch
conda install matplotlib
```

## 🎯 Usage

### Basic Usage

```bash
# Run the complete MNIST analysis
python src/mnist_data_handler.py
```

### Programmatic Usage

```python
from src.mnist_data_handler import MNISTDataLoader, DatasetAnalyzer

# Initialize data loader
loader = MNISTDataLoader(batch_size=64)

# Load datasets
train_dataset, test_dataset = loader.load_datasets()

# Create data loaders
train_loader, test_loader = loader.create_data_loaders()

# Analyze dataset
analysis = DatasetAnalyzer.analyze_dataset(train_dataset)
print(f"Dataset contains {analysis['total_samples']} samples")
```

## 📊 What It Does

### ✅ **System Verification**
- PyTorch installation check
- CUDA availability detection
- Version information display

### ✅ **Dataset Management**
- Automatic MNIST download
- Configurable preprocessing pipelines
- Efficient data loader creation

### ✅ **Data Analysis**
- Comprehensive dataset statistics
- Tensor property analysis
- Performance benchmarking

### ✅ **Visualization**
- Sample image grid generation
- Automated plot saving
- Clean matplotlib integration

## 🔧 Technical Details

### Classes

- **`MNISTDataLoader`**: Handles dataset loading and data loader creation
- **`DatasetAnalyzer`**: Provides analysis and visualization utilities
- **`SystemInfo`**: System and environment information

### Key Features

- **Type Safety**: Full type hints for better IDE support
- **Logging**: Structured logging with timestamps
- **Modularity**: Reusable components for other projects
- **Performance**: Optimized data loading with benchmarking

## � Performance

- **Data Loading**: ~27,000 images/second
- **Memory Efficient**: Lazy loading with configurable batch sizes
- **CPU Optimized**: Works efficiently without GPU

## 🎓 Next Steps

This foundation enables:
- Neural network development
- Custom model training
- Experiment tracking
- Advanced preprocessing pipelines
