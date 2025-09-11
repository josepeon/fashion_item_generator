# PyTorch MNIST: Classification + Generation

A clean, complete PyTorch project demonstrating both **classification** and **generation** of handwritten digits.

## Project Overview

1. **Classifies** handwritten digits with **99.50% accuracy**
2. **Generates** new realistic digit images with **90.5% quality**
3. **Evaluates** both models with comprehensive metrics

## Project Structure

```text
pytorch_learn/
├── README.md                    # This file
├── run_demo.py                  # Complete pipeline demo
├── src/                         # Source code
│   ├── mnist_handler.py         # MNIST data loading
│   ├── mnist_cnn.py            # Enhanced CNN classifier (99.50% accuracy)
│   ├── quick_generator.py      # VAE generator
│   └── evaluate_models.py      # Model evaluation & comparison
├── models/                      # Trained models
│   ├── best_mnist_cnn.pth      # Best CNN model (99.50%)
│   ├── mnist_cnn_final_99.5pct.pth # Final CNN checkpoint
│   └── quick_generator.pth     # Trained VAE generator
├── results/                     # Training results & visualizations
├── checkpoints/                 # Training checkpoints (every 10 epochs)
└── data/MNIST/                 # Dataset (auto-downloaded)
```

## Quick Start

**Option 1: Run complete pipeline demo:**

```bash
# Setup environment (Python 3.12 + latest PyTorch)
conda env create -f environment.yml
conda activate pytorch_learn_env

# Run everything at once
python run_demo.py
```

**Alternative setup:**

```bash
# Manual environment setup
conda create -n pytorch_learn_env python=3.12
conda activate pytorch_learn_env
conda install pytorch torchvision torchaudio cpuonly -c pytorch
conda install matplotlib pandas scikit-learn seaborn

# Run everything at once
python run_demo.py
```

**Option 2: Step by step training:**

```bash
# 1. Enhanced CNN training (up to 200 epochs, early stopping)
python src/mnist_cnn.py
# → models/best_mnist_cnn.pth (99.50% accuracy)

# 2. VAE generator training
python src/quick_generator.py  
# → models/quick_generator.pth (digit generation)

# 3. Comprehensive evaluation
python src/evaluate_models.py
# → results/ (quality metrics, comparison images)
```

## Performance Results

| Model | Task | Performance | Quality |
|-------|------|-------------|---------|
| **CNN Classifier** | Digit Recognition | **99.50% accuracy** | Excellent |
| **VAE Generator** | Digit Creation | **90.5% CNN confidence** | High Quality |

### Key Metrics

- **Enhanced Classification**: 99.50% test accuracy (improved from 99.37%)
- **Enhanced Generation**: 90.5% quality rating (90%+ high confidence >0.7)
- **Parameters**: 688K (CNN) + 801K (VAE) = 1.49M total
- **Training Features**: Early stopping, learning rate scheduling, checkpointing
- **Training Time**: CNN ~10 minutes, VAE ~10 minutes (both up to 200 epochs)

## Technical Implementation

### Core PyTorch Concepts

- Building neural networks with `nn.Module`
- Training loops with backpropagation
- Data loading with `DataLoader`
- Model saving and loading

### Advanced Techniques

- **CNN Architecture**: Convolutional layers, pooling, dropout
- **VAE Generation**: Latent space, encoder-decoder, reparameterization
- **Model Evaluation**: Using one model to evaluate another

## Architecture Details

### CNN Classifier Architecture

```text
Input (28×28) → Conv(32) → Conv(64) → Conv(128) → FC(512) → FC(10)
```

- 688,138 parameters
- Dropout regularization
- Adam optimizer

### VAE Generator Architecture

```text
Encoder: 784 → 400 → 200 → latent(20)
Decoder: latent(20) → 200 → 400 → 784
```

- 801,224 parameters
- Latent space dimension: 20
- Generates from random noise

## Generated Examples

The VAE successfully generates digit-like images that:

- **75% achieve high confidence** (>0.7) from our CNN classifier
- **93.8% are recognizable** as valid digits
- **Show clear digit patterns** across all 10 classes (0-9)

## 🎓 Educational Value

This project demonstrates:

1. **Discriminative AI**: Learning to classify existing data
2. **Generative AI**: Learning to create new data
3. **Model Evaluation**: Using one AI to judge another
4. **Complete Pipeline**: From data loading to final evaluation

Perfect for understanding both **recognition** and **creation** in deep learning!

## Environment & Dependencies

This project uses the latest stable versions:

- **Python**: 3.12.11 (latest stable)
- **PyTorch**: 2.2.2 (with CPU optimization)
- **NumPy**: 1.26.4 (compatible with PyTorch)
- **Matplotlib**: 3.10.5
- **Pandas**: 2.3.2
- **Scikit-learn**: 1.7.1

### Environment Setup

The project includes a complete `environment.yml` file for easy reproduction:

```bash
conda env create -f environment.yml
conda activate pytorch_learn_env
```

This ensures identical dependencies and versions across different machines.

## Next Steps

- Experiment with different architectures
- Try conditional generation (specify which digit to generate)
- Implement more advanced GANs
- Apply to other datasets (CIFAR-10, Fashion-MNIST)
- Upgrade to newer PyTorch versions for latest features
