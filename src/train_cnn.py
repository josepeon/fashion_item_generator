#!/usr/bin/env python3
"""Train Fashion-MNIST CNN classifier."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from models import FashionCNN


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_dataloaders(batch_size: int = 256):
    train_transform = transforms.Compose([
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.1)),
        transforms.Normalize((0.2860,), (0.3530,)),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),
    ])

    train_dataset = datasets.FashionMNIST(
        root="./data", train=True, download=True, transform=train_transform
    )
    test_dataset = datasets.FashionMNIST(
        root="./data", train=False, download=True, transform=test_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    return train_loader, test_loader


def train(epochs: int = 200, batch_size: int = 256, lr: float = 0.003):
    device = get_device()
    print(f"Training on: {device}")

    model = FashionCNN().to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    train_loader, test_loader = get_dataloaders(batch_size)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=lr,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
    )

    best_acc = 0.0

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()

        # Evaluate
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

        acc = 100.0 * correct / total
        if epoch % 10 == 0 or acc > best_acc:
            print(f"Epoch {epoch+1:3d}/{epochs} | Loss: {train_loss/len(train_loader):.4f} | Acc: {acc:.2f}%")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "weights/cnn.pth")

    print(f"\nBest accuracy: {best_acc:.2f}%")
    return model


if __name__ == "__main__":
    train()
