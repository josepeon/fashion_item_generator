# Fashion Item Generator

Fashion-MNIST classification and generation with PyTorch.

## Models

| Model | Params | Performance |
|-------|--------|-------------|
| CNN   | 6.9M   | 95.3% accuracy |
| VAE   | 33.5M  | A+ generation |

## Setup

```bash
conda env create -f environment.yml
conda activate fashion_mnist_env
```

## Usage

```bash
# Demo both models
python src/demo.py

# Evaluate CNN
python src/evaluate_cnn.py

# Evaluate VAE
python src/evaluate_vae.py

# Train (optional)
python src/train_cnn.py
python src/train_vae.py
```

## Structure

```
fashion_item_generator/
├── src/
│   ├── models/          # Model architectures
│   │   ├── cnn.py       # FashionCNN
│   │   └── vae.py       # FashionVAE
│   ├── train_cnn.py     # CNN training
│   ├── train_vae.py     # VAE training
│   ├── evaluate_cnn.py  # CNN evaluation
│   ├── evaluate_vae.py  # VAE evaluation
│   └── demo.py          # Quick demo
├── weights/             # Trained model weights
│   ├── cnn.pth
│   └── vae.pth
├── data/                # Fashion-MNIST (auto-download)
├── results/             # Generated outputs
├── environment.yml
└── requirements.txt
```

## Requirements

- Python 3.12+
- PyTorch 2.0+
- Apple Silicon (MPS) / CUDA supported
