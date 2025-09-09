# PyTorch Learning Project

A comprehensive, hands-on tutorial for learning PyTorch fundamentals using the MNIST dataset.

## 🎯 What You'll Learn

- PyTorch installation and setup verification
- Loading and exploring datasets (MNIST)
- Data visualization and preprocessing
- Creating efficient data loaders
- Understanding tensor operations
- Building neural networks (coming next!)

## 📁 Project Structure

```
pytorch_learn/
├── README.md                 # This file
├── src/
│   └── mnist_tutorial.py     # Complete MNIST tutorial
├── data/                     # MNIST dataset (auto-downloaded)
└── .gitignore               # Ignores data files and outputs
```

## 🚀 Getting Started

### 1. Environment Setup

```bash
# Create and activate conda environment
conda create -n pytorch_learn_env python=3.8
conda activate pytorch_learn_env

# Install PyTorch and dependencies
conda install pytorch torchvision torchaudio cpuonly -c pytorch
conda install matplotlib
```

### 2. Run the Tutorial

```bash
# Make sure you're in the project directory
cd pytorch_learn

# Activate environment
conda activate pytorch_learn_env

# Run the comprehensive tutorial
python src/mnist_tutorial.py
```

## 📊 What the Tutorial Covers

### ✅ **Step 1: Setup Verification**
- Check PyTorch installation
- Verify CUDA availability
- Display version information

### ✅ **Step 2: Dataset Loading**
- Download MNIST dataset automatically
- Understand data transformations
- Explore dataset properties

### ✅ **Step 3: Data Visualization**
- Display sample digit images
- Understand image preprocessing
- Save visualization outputs

### ✅ **Step 4: Data Loaders**
- Create efficient batched data loading
- Configure training vs test loaders
- Understand batch processing

### ✅ **Step 5: Tensor Operations**
- Explore tensor shapes and properties
- Learn basic tensor manipulations
- Understand memory and data types

## 🎓 Learning Path

1. **Start Here**: Run `mnist_tutorial.py` to understand the basics
2. **Next**: Build your first neural network (coming soon!)
3. **Then**: Train and evaluate models
4. **Advanced**: Explore different architectures and datasets

## 🔧 Requirements

- Python 3.8+
- PyTorch 2.0+
- Matplotlib for visualizations
- ~100MB disk space for MNIST data

## 📈 Next Steps

After completing this tutorial, you'll be ready to:
- Build feedforward neural networks
- Create convolutional neural networks (CNNs)
- Train models on MNIST and other datasets
- Implement custom training loops
- Experiment with different optimizers and loss functions
