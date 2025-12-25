#!/usr/bin/env python3
"""Train Fashion-MNIST VAE generator."""

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from models import FashionVAE


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train(epochs: int = 150, latent_dim: int = 32, batch_size: int = 128, lr: float = 1e-3):
    device = get_device()
    print(f"Training on: {device}")

    model = FashionVAE(latent_dim=latent_dim).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    train_data = datasets.FashionMNIST("./data", train=True, download=True, transform=transform)
    test_data = datasets.FashionMNIST("./data", train=False, transform=transform)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_data, batch_size=256, shuffle=False, num_workers=4)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = OneCycleLR(optimizer, max_lr=lr, epochs=epochs, steps_per_epoch=len(train_loader))

    best_loss = float("inf")
    for epoch in range(epochs):
        model.train()
        for data, labels in train_loader:
            data = data.view(data.size(0), -1).to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            recon, mu, logvar = model(data, labels)
            recon_loss = F.mse_loss(recon, data, reduction="sum") / data.size(0)
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / data.size(0)
            beta = min(1.0, epoch / 30)  # KL warmup
            loss = recon_loss + beta * kl_loss
            loss.backward()
            optimizer.step()
            scheduler.step()

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data, labels in test_loader:
                data = data.view(data.size(0), -1).to(device)
                labels = labels.to(device)
                recon, mu, logvar = model(data, labels)
                recon_loss = F.mse_loss(recon, data, reduction="sum") / data.size(0)
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / data.size(0)
                val_loss += (recon_loss + kl_loss).item()
        val_loss /= len(test_loader)

        if epoch % 30 == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} | Loss: {val_loss:.3f} | Best: {best_loss:.3f}")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), "weights/vae.pth")

    print(f"\nBest loss: {best_loss:.3f}")
    return model


if __name__ == "__main__":
    train()
