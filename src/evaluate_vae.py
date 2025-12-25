#!/usr/bin/env python3
"""Evaluate Fashion-MNIST VAE generator."""

import json
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from models import FashionVAE

CLASS_NAMES = [
    "T-shirt", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(path: str = "weights/vae.pth") -> FashionVAE:
    device = get_device()
    model = FashionVAE(latent_dim=64, conditional=True).to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.eval()
    return model


def reconstruction_quality(model, data_loader, device):
    """Measure reconstruction quality."""
    model.eval()
    total_mse = 0.0
    total_corr = 0.0
    n_batches = 0

    with torch.no_grad():
        for data, labels in data_loader:
            data = data.view(data.size(0), -1).to(device)
            labels = labels.to(device)
            recon, _, _ = model(data, labels)

            total_mse += F.mse_loss(recon, data).item()

            # Correlation
            x_flat = data - data.mean(dim=1, keepdim=True)
            y_flat = recon - recon.mean(dim=1, keepdim=True)
            num = (x_flat * y_flat).sum(dim=1)
            denom = torch.sqrt((x_flat**2).sum(dim=1) * (y_flat**2).sum(dim=1) + 1e-8)
            total_corr += (num / denom).mean().item()
            n_batches += 1

    return total_mse / n_batches, total_corr / n_batches


def generation_diversity(model, device, n_samples=100):
    """Measure diversity of generated samples."""
    model.eval()
    with torch.no_grad():
        samples = model.generate(n_samples, device=device).view(n_samples, -1)

    distances = []
    for i in range(min(50, n_samples)):
        for j in range(i + 1, min(50, n_samples)):
            dist = torch.sqrt(((samples[i] - samples[j]) ** 2).sum()).item()
            distances.append(dist)

    return np.mean(distances), np.std(distances)


def save_reconstruction_samples(model, data_loader, device, path):
    """Save reconstruction visualization."""
    model.eval()
    data, labels = next(iter(data_loader))
    data = data[:8].to(device)
    labels = labels[:8].to(device)

    with torch.no_grad():
        recon, _, _ = model(data.view(8, -1), labels)

    fig, axes = plt.subplots(2, 8, figsize=(12, 3))
    for i in range(8):
        axes[0, i].imshow(data[i].cpu().squeeze(), cmap="gray")
        axes[0, i].axis("off")
        axes[1, i].imshow(recon[i].cpu().view(28, 28), cmap="gray")
        axes[1, i].axis("off")
    axes[0, 0].set_ylabel("Original")
    axes[1, 0].set_ylabel("Recon")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_generation_samples(model, device, path):
    """Save generation visualization."""
    model.eval()
    fig, axes = plt.subplots(2, 10, figsize=(12, 3))

    with torch.no_grad():
        for i in range(10):
            samples = model.generate_class(i, 2, device=device)
            for j in range(2):
                axes[j, i].imshow(samples[j].cpu().squeeze(), cmap="gray")
                axes[j, i].axis("off")
            axes[0, i].set_title(CLASS_NAMES[i], fontsize=8)

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main():
    device = get_device()
    print(f"Device: {device}\n")

    model = load_model()
    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}\n")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    test_dataset = datasets.FashionMNIST(root="./data", train=False, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    # Metrics
    mse, corr = reconstruction_quality(model, test_loader, device)
    div_mean, div_std = generation_diversity(model, device)

    print("Reconstruction:")
    print(f"  MSE:         {mse:.4f}")
    print(f"  Correlation: {corr:.4f}")
    print("\nGeneration:")
    print(f"  Diversity:   {div_mean:.2f} ± {div_std:.2f}")

    # Grade
    score = corr - 0.1 * mse + 0.01 * div_mean
    if score > 0.85:
        grade = "A+"
    elif score > 0.75:
        grade = "A"
    elif score > 0.65:
        grade = "B"
    else:
        grade = "C"
    print(f"\nGrade: {grade} (score: {score:.3f})")

    # Save visualizations
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_reconstruction_samples(model, test_loader, device, f"results/vae_recon_{timestamp}.png")
    save_generation_samples(model, device, f"results/vae_gen_{timestamp}.png")

    # Save results
    results = {
        "timestamp": timestamp,
        "parameters": params,
        "mse": mse,
        "correlation": corr,
        "diversity_mean": div_mean,
        "diversity_std": div_std,
        "score": score,
        "grade": grade,
    }
    with open(f"results/vae_eval_{timestamp}.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to results/")


if __name__ == "__main__":
    main()
