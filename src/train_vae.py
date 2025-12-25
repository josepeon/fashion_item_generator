#!/usr/bin/env python3
"""Train Fashion-MNIST VAE generator."""

import time
import warnings

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from models import FashionVAE


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def compute_loss(recon_x, x, mu, logvar, beta=1.0):
    batch_size = x.size(0)
    recon_loss = F.mse_loss(recon_x, x, reduction="sum") / batch_size
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size
    return recon_loss + beta * kl_loss, recon_loss.item(), kl_loss.item()


def train(
    epochs: int = 500,
    latent_dim: int = 64,
    batch_size: int = 256,
    lr: float = 2e-3,
    beta_start: float = 0.1,
    beta_end: float = 2.0,
):
    warnings.filterwarnings("ignore")
    device = get_device()
    print(f"Training on: {device}")

    model = FashionVAE(latent_dim=latent_dim, conditional=True).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    train_dataset = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
    val_dataset = datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = OneCycleLR(
        optimizer, max_lr=lr, epochs=epochs, steps_per_epoch=len(train_loader), pct_start=0.1
    )

    best_loss = float("inf")
    start_time = time.time()

    for epoch in range(epochs):
        # Progressive beta schedule
        if epoch < epochs * 0.3:
            beta = beta_start
        else:
            progress = (epoch - epochs * 0.3) / (epochs * 0.7)
            beta = beta_start + (beta_end - beta_start) * progress

        # Train
        model.train()
        train_loss = 0.0
        for data, labels in train_loader:
            data = data.view(data.size(0), -1).to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            recon, mu, logvar = model(data, labels)
            loss, _, _ = compute_loss(recon, data, mu, logvar, beta)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data, labels in val_loader:
                data = data.view(data.size(0), -1).to(device)
                labels = labels.to(device)
                recon, mu, logvar = model(data, labels)
                loss, _, _ = compute_loss(recon, data, mu, logvar, beta)
                val_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        if epoch % 20 == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} | Train: {train_loss:.3f} | Val: {val_loss:.3f} | β: {beta:.2f}")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), "weights/vae.pth")

    elapsed = time.time() - start_time
    print(f"\nTraining complete in {elapsed/60:.1f} min")
    print(f"Best validation loss: {best_loss:.4f}")
    return model


if __name__ == "__main__":
    train()
