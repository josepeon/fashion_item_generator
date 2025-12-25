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

Or with pip:

```bash
pip install -r requirements.txt
```

## Usage

```bash
python src/demo.py          # Demo both models
python src/evaluate_cnn.py  # Evaluate classifier
python src/evaluate_vae.py  # Evaluate generator
```

Training (optional):

```bash
python src/train_cnn.py
python src/train_vae.py
```

## Project Structure

```
├── src/
│   ├── models/         # Model architectures
│   ├── train_cnn.py    # Train classifier
│   ├── train_vae.py    # Train generator
│   ├── evaluate_cnn.py
│   ├── evaluate_vae.py
│   └── demo.py
├── weights/            # Model weights (.pth)
├── data/               # Fashion-MNIST (auto-download)
└── results/            # Generated outputs
```

## Requirements

- Python 3.10+
- PyTorch 2.0+
- MPS (Apple Silicon) / CUDA supported
