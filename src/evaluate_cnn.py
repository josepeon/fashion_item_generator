#!/usr/bin/env python3
"""Evaluate Fashion-MNIST CNN classifier."""

from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from models import FashionCNN

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = PROJECT_ROOT / "weights"
DATA_DIR = PROJECT_ROOT / "data"

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


def load_model(path: Path = None) -> FashionCNN:
    if path is None:
        path = WEIGHTS_DIR / "cnn.pth"
    device = get_device()
    model = FashionCNN().to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.eval()
    return model


def evaluate_with_tta(model: FashionCNN, test_loader: DataLoader, device: torch.device):
    """Evaluate with test-time augmentation."""
    class_correct = [0] * 10
    class_total = [0] * 10

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            # TTA: original + horizontal flip + rotations
            preds = [
                model(data),
                model(torch.flip(data, dims=[3])),
                model(transforms.functional.rotate(data, 2)),
                model(transforms.functional.rotate(data, -2)),
                model(transforms.functional.rotate(data, 5)),
                model(transforms.functional.rotate(data, -5)),
            ]
            output = torch.stack(preds).mean(dim=0)
            _, predicted = output.max(1)

            for i in range(target.size(0)):
                label = target[i].item()
                class_total[label] += 1
                if predicted[i] == label:
                    class_correct[label] += 1

    return class_correct, class_total


def main():
    device = get_device()
    print(f"Device: {device}\n")

    model = load_model()
    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}\n")

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),
    ])
    test_dataset = datasets.FashionMNIST(
        root=DATA_DIR, train=False, download=True, transform=test_transform
    )
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False, num_workers=4)

    class_correct, class_total = evaluate_with_tta(model, test_loader, device)

    print("Per-class accuracy:")
    print("-" * 35)
    for i in range(10):
        acc = 100 * class_correct[i] / class_total[i]
        print(f"  {CLASS_NAMES[i]:12s}: {acc:5.2f}%")

    total_acc = 100 * sum(class_correct) / sum(class_total)
    print("-" * 35)
    print(f"  Overall:      {total_acc:5.2f}%")


if __name__ == "__main__":
    main()
