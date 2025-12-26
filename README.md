# Fashion Item Generator

Fashion-MNIST classification and generation with PyTorch.

## Models

| Model | Params | Performance |
|-------|--------|-------------|
| CNN   | 1.2M   | 95.1% accuracy |
| VAE   | 3.7M   | A+ (0.93 correlation) |

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

### API Server

```bash
python src/api.py           # Start FastAPI server on :8000
```

Endpoints:
- `POST /classify` - Upload image → get classification
- `POST /generate` - Class ID → generated images (base64)
- `GET /health` - Service status
- `GET /classes` - List all classes

Docs at http://localhost:8000/docs

### Streamlit Demo

```bash
streamlit run src/app.py    # Interactive web demo
```

Features:
- Upload image → classification + variations
- Generate samples by class

Training:

```bash
python src/train_cnn.py     # ~150 epochs, reaches 95%+
python src/train_vae.py     # ~150 epochs
```

## Project Structure

```
├── src/
│   ├── models/
│   │   ├── cnn.py          # FashionCNN classifier
│   │   └── vae.py          # FashionVAE generator
│   ├── train_cnn.py
│   ├── train_vae.py
│   ├── evaluate_cnn.py
│   ├── evaluate_vae.py
│   ├── demo.py
│   ├── api.py              # FastAPI server
│   └── app.py              # Streamlit demo
├── weights/                # Trained weights (gitignored)
├── data/                   # Fashion-MNIST (auto-download)
└── results/                # Evaluation outputs
```

## Architecture

### CNN Classifier
- 3 conv blocks (64→128→256 channels)
- Global average pooling
- Single FC layer (256→10)
- TTA (test-time augmentation) for evaluation

### VAE Generator
- Conditional on class labels (one-hot)
- Encoder: 784→512→256→32 (latent)
- Decoder: 32→256→512→784
- Residual blocks with LayerNorm

## Results

**CNN per-class accuracy:**
```
T-shirt/top: 91.2%  |  Sandal:     99.1%
Trouser:     99.5%  |  Shirt:      83.7%
Pullover:    92.4%  |  Sneaker:    98.2%
Dress:       95.3%  |  Bag:        99.7%
Coat:        94.7%  |  Ankle boot: 97.0%
```

**VAE metrics:**
- MSE: 0.040
- Correlation: 0.934
- Diversity: 20.8

## Requirements

- Python 3.10+
- PyTorch 2.0+
- MPS / CUDA supported
