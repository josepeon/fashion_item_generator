#!/usr/bin/env python3
"""Quick demo of Fashion-MNIST models."""

from pathlib import Path

import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

from models import FashionCNN, FashionVAE

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = PROJECT_ROOT / "weights"
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def demo_cnn():
    """Demo CNN classifier."""
    device = get_device()
    model = FashionCNN().to(device)
    model.load_state_dict(torch.load(WEIGHTS_DIR / "cnn.pth", map_location=device, weights_only=True))
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),
    ])
    dataset = datasets.FashionMNIST(root=DATA_DIR, train=False, transform=transform)

    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    indices = torch.randperm(len(dataset))[:10]

    for i, idx in enumerate(indices):
        img, label = dataset[idx]
        with torch.no_grad():
            output = model(img.unsqueeze(0).to(device))
            probs = torch.softmax(output, dim=1)
            pred = output.argmax(1).item()
            conf = probs[0, pred].item()

        ax = axes[i // 5, i % 5]
        ax.imshow(img.squeeze() * 0.353 + 0.286, cmap="gray")
        ax.set_title(f"{CLASS_NAMES[pred]}\n({conf:.0%})", fontsize=9)
        color = "green" if pred == label else "red"
        ax.spines[:].set_color(color)
        ax.spines[:].set_linewidth(2)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle("CNN Classifier Demo", fontsize=12)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "demo_cnn.png", dpi=150)
    plt.close()
    print(f"Saved: {RESULTS_DIR / 'demo_cnn.png'}")


def demo_vae():
    """Demo VAE generator."""
    device = get_device()
    model = FashionVAE(latent_dim=32, conditional=True).to(device)
    model.load_state_dict(torch.load(WEIGHTS_DIR / "vae.pth", map_location=device, weights_only=True))
    model.eval()

    fig, axes = plt.subplots(2, 5, figsize=(10, 4))

    with torch.no_grad():
        for i in range(10):
            sample = model.generate_class(i, 1, device=device)
            ax = axes[i // 5, i % 5]
            ax.imshow(sample.cpu().squeeze(), cmap="gray")
            ax.set_title(CLASS_NAMES[i], fontsize=9)
            ax.axis("off")

    plt.suptitle("VAE Generator Demo", fontsize=12)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "demo_vae.png", dpi=150)
    plt.close()
    print(f"Saved: {RESULTS_DIR / 'demo_vae.png'}")


if __name__ == "__main__":
    print("Fashion-MNIST Demo\n")
    demo_cnn()
    demo_vae()
