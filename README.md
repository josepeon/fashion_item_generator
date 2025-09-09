# PyTorch MNIST CNN Learning Project

A clean, educational PyTorch implementation for MNIST digit classification using Convolutional Neural Networks.

## 🎯 Project Overview

This project demonstrates fundamental PyTorch concepts through a well-structured CNN that achieves **99.37% accuracy** on MNIST digit recognition.

## 📁 Project Structure

```
pytorch_learn/
├── README.md                 # This file
├── .gitignore               # Git ignore rules
├── src/
│   ├── mnist_handler.py     # Optimized MNIST data loading
│   └── mnist_cnn.py         # Complete CNN implementation
├── data/
│   └── MNIST/               # Dataset (auto-downloaded)
├── mnist_cnn.pth           # Trained model weights
└── training_history.png    # Training visualization
```

## 🚀 Quick Start

1. **Setup Environment**
   ```bash
   conda create -n pytorch_learn_env python=3.8
   conda activate pytorch_learn_env
   conda install pytorch torchvision matplotlib
   ```

2. **Train the Model**
   ```bash
   python src/mnist_cnn.py
   ```

3. **Results**
   - Model trains for 10 epochs (~6 minutes on CPU)
   - Achieves 99.37% test accuracy
   - Saves trained model as `mnist_cnn.pth`
   - Generates training history visualization

## 🧠 What You'll Learn

- **CNN Architecture**: 3 convolutional layers + 2 fully connected layers
- **PyTorch Fundamentals**: Tensors, autograd, nn.Module, DataLoader
- **Training Loop**: Forward pass, loss calculation, backpropagation
- **Model Evaluation**: Accuracy metrics and visualization
- **Data Handling**: Efficient dataset loading and preprocessing

## 📊 Model Performance

- **Test Accuracy**: 99.37%
- **Parameters**: 688,138
- **Architecture**: Conv2D → ReLU → MaxPool → Dropout → FC
- **Training Time**: ~6 minutes (CPU)

## 🔧 Technical Details

### CNN Architecture
```
Input (28x28) → Conv(32) → Conv(64) → Conv(128) → FC(512) → FC(10)
```

### Key Features
- Dropout regularization (25% and 50%)
- Adam optimizer with learning rate 0.001
- Cross-entropy loss function
- Automatic device detection (CPU/GPU)

## 📈 Training Progress

The model shows consistent improvement:
- Epoch 1: 98.45% → Epoch 10: 99.37%
- Training loss decreases steadily
- No overfitting observed

## 🎓 Learning Outcomes

This project covers essential PyTorch concepts:
- Building custom neural networks with `nn.Module`
- Implementing training and evaluation loops
- Using `DataLoader` for efficient batch processing
- Visualizing training progress with matplotlib
- Saving and loading model checkpoints

Perfect for beginners learning deep learning with PyTorch!