#!/usr/bin/env python3
"""Train Fashion-MNIST CNN classifier."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from models import FashionCNN


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train(epochs: int = 150, batch_size: int = 64, lr: float = 1e-3):
    device = get_device()
    print(f"Training on: {device}")

    model = FashionCNN().to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    train_transform = transforms.Compose([
        transforms.RandomRotation(15),
        transforms.RandomAffine(0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),
        transforms.Normalize((0.2860,), (0.3530,)),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),
    ])

    train_data = datasets.FashionMNIST("./data", train=True, download=True, transform=train_transform)
    test_data = datasets.FashionMNIST("./data", train=False, transform=test_transform)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    test_loader = DataLoader(test_data, batch_size=256, shuffle=False, num_workers=4)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            loss = criterion(model(data), target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        # Evaluate with TTA
        model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                preds = [
                    model(data),
                    model(torch.flip(data, dims=[3])),
                    model(transforms.functional.rotate(data, 5)),
                    model(transforms.functional.rotate(data, -5)),
                ]
                output = torch.stack(preds).mean(dim=0)
                correct += output.argmax(1).eq(target).sum().item()

        acc = 100.0 * correct / len(test_data)
        if epoch % 10 == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} | Acc: {acc:.2f}% | Best: {best_acc:.2f}%")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "weights/cnn.pth")

    print(f"\nBest accuracy: {best_acc:.2f}%")
    return model


if __name__ == "__main__":
    train()
