# Fashion-MNIST Generator

A machine learning project for Fashion-MNIST classification and generation using CNN and VAE models.

## Quick Start

```bash
# Clone and setup
git clone https://github.com/josepeon/fashion_item_generator.git
cd fashion_item_generator
conda env create -f environment.yml
conda activate fashion_mnist_env

# Run demos
python src/complete_demo.py         # CNN classification
python src/simple_generator.py     # VAE generation
```

## Features

- **Classification**: CNN model for fashion item recognition
- **Generation**: VAE models for creating new fashion items
- **Complete Pipeline**: End-to-end prediction and generation

## Project Structure

```
fashion_item_generator/
├── src/                    # Source code
│   ├── fashion_handler.py  # Data loading
│   ├── fashion_cnn.py      # CNN models
│   ├── enhanced_fashion_cnn.py  # Advanced CNN
│   ├── enhanced_vae.py     # Conditional VAE
│   ├── simple_generator.py # Basic VAE
│   └── complete_demo.py    # Demo script
├── models/                 # Trained models
├── data/                   # Dataset
└── results/               # Outputs
```

## Models

### CNN Classification
- **Enhanced CNN**: Advanced architecture with attention mechanism
- **Performance**: 94.50% accuracy on Fashion-MNIST test set
- **Features**: Batch normalization, dropout, MPS acceleration

### VAE Generation
- **Simple VAE**: Basic unconditional generation (653K parameters)
- **Enhanced VAE**: Conditional class-specific generation (3.48M parameters)
- **Capabilities**: Generate fashion items for all 10 categories

## Fashion Categories

The models work with all 10 Fashion-MNIST categories:
- T-shirt/top, Trouser, Pullover, Dress, Coat
- Sandal, Shirt, Sneaker, Bag, Ankle boot

## Requirements

- Python 3.12+
- PyTorch 2.5.1+
- See `environment.yml` for complete dependencies

## Usage

### Classification
```python
from fashion_cnn import FashionNet
import torch

model = FashionNet()
model.load_state_dict(torch.load('models/enhanced_fashion_cnn_200epochs.pth'))
# Use model for predictions
```

### Generation
```python
from enhanced_vae import EnhancedVAE

model = EnhancedVAE(latent_dim=32, conditional=True)
model.load_state_dict(torch.load('models/enhanced_vae_superior.pth'))
# Generate fashion items
samples = model.generate(num_samples=16)
```

## License

MIT License - see LICENSE file for details.
