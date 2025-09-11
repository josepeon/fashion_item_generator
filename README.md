# PyTorch MNIST: Classification + Generation

A clean, complete PyTorch project demonstrating both **classification** and **generation** of handwritten digits.

## 🎯 What This Project Does

1. **🔍 Classifies** handwritten digits with **99.37% accuracy**
2. **🎨 Generates** new realistic digit images with **85% quality**
3. **📊 Evaluates** both models with comprehensive metrics

## 📁 Project Structure

```
pytorch_learn/
├── README.md                    # This file
├── src/
│   ├── mnist_handler.py         # MNIST data loading
│   ├── mnist_cnn.py            # CNN classifier (99.37% accuracy)
│   ├── quick_generator.py      # VAE generator (85% quality)
│   └── evaluate_models.py      # Model evaluation & comparison
├── data/MNIST/                 # Dataset (auto-downloaded)
├── mnist_cnn.pth              # Trained classifier
├── quick_generator.pth        # Trained generator
├── training_history.png       # CNN training progress
├── vae_quality_test.png       # Generated digits evaluation
└── quality_comparison_final.png # Real vs Generated comparison
```

## 🚀 Quick Start

### 1. Setup Environment
```bash
conda create -n pytorch_learn_env python=3.8
conda activate pytorch_learn_env
conda install pytorch torchvision matplotlib
```

### 2. Train CNN Classifier
```bash
python src/mnist_cnn.py
# Output: 99.37% accuracy, saves mnist_cnn.pth
```

### 3. Train VAE Generator  
```bash
python src/quick_generator.py
# Output: Generates new digits, saves quick_generator.pth
```

### 4. Evaluate Both Models
```bash
python src/evaluate_models.py
# Output: Quality metrics, comparison images
```

## 📊 Performance Results

| Model | Task | Performance | Quality |
|-------|------|-------------|---------|
| **CNN Classifier** | Digit Recognition | **99.37% accuracy** | Excellent |
| **VAE Generator** | Digit Creation | **85% CNN confidence** | High Quality |

### 🎯 Key Metrics
- **Classification**: 99.37% test accuracy on 10,000 MNIST images
- **Generation**: 85.2% average confidence when classifying generated digits
- **Quality Ratio**: Generated images are 82% as recognizable as real ones

## 🧠 What You'll Learn

### Core PyTorch Concepts
- Building neural networks with `nn.Module`
- Training loops with backpropagation
- Data loading with `DataLoader`
- Model saving and loading

### Advanced Techniques
- **CNN Architecture**: Convolutional layers, pooling, dropout
- **VAE Generation**: Latent space, encoder-decoder, reparameterization
- **Model Evaluation**: Using one model to evaluate another

## 🔧 Technical Details

### CNN Classifier Architecture
```
Input (28×28) → Conv(32) → Conv(64) → Conv(128) → FC(512) → FC(10)
```
- 688,138 parameters
- Dropout regularization
- Adam optimizer

### VAE Generator Architecture  
```
Encoder: 784 → 400 → 200 → latent(20)
Decoder: latent(20) → 200 → 400 → 784
```
- 801,224 parameters
- Latent space dimension: 20
- Generates from random noise

## 🎨 Generated Examples

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

## 🔄 Next Steps

- Experiment with different architectures
- Try conditional generation (specify which digit to generate)
- Implement more advanced GANs
- Apply to other datasets (CIFAR-10, Fashion-MNIST)