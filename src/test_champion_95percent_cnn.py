#!/usr/bin/env python3
"""
Test Expert CNN - Load and evaluate the trained 95%+ accuracy model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from datetime import datetime
import os


class ExpertFashionCNN(nn.Module):
    """Simplified expert CNN optimized specifically for Fashion-MNIST."""
    
    def __init__(self, num_classes=10):
        super(ExpertFashionCNN, self).__init__()
        
        # Feature extractor - optimized for 28x28 images
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 96, kernel_size=5, padding=2),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.1),
            
            # Block 2
            nn.Conv2d(96, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2),
            
            # Block 3
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            # Block 4
            nn.Conv2d(384, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        
        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def test_expert_cnn():
    """Test the trained Expert CNN model."""
    print("TESTING EXPERT CNN MODEL")
    print("=" * 50)
    
    # Setup device
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Testing on: {device}")
    
    # Load model
    model = ExpertFashionCNN().to(device)
    
    # Check if model exists
    model_path = 'models/champion_95percent_cnn.pth'
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        print("Available models:")
        if os.path.exists('models'):
            for file in os.listdir('models'):
                if file.endswith('.pth'):
                    print(f"    {file}")
        return
    
    # Load trained weights
    print(f"📂 Loading model from: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f" Model parameters: {total_params:,}")
    
    # Load test data
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])
    
    test_dataset = datasets.FashionMNIST(
        root='./data', train=False, download=True, transform=test_transform
    )
    test_loader = DataLoader(
        test_dataset, batch_size=512, shuffle=False,
        num_workers=4, pin_memory=True
    )
    
    print(f" Test samples: {len(test_dataset):,}")
    
    # Enhanced test-time augmentation evaluation
    print("\n Running enhanced evaluation with TTA...")
    
    class_correct = [0] * 10
    class_total = [0] * 10
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            # Enhanced test-time augmentation
            predictions = []
            
            # Multiple augmentations
            predictions.append(model(data))
            predictions.append(model(torch.flip(data, dims=[3])))
            predictions.append(model(transforms.functional.rotate(data, 2)))
            predictions.append(model(transforms.functional.rotate(data, -2)))
            predictions.append(model(transforms.functional.rotate(data, 5)))
            predictions.append(model(transforms.functional.rotate(data, -5)))
            
            # Average all predictions
            outputs_avg = torch.stack(predictions).mean(dim=0)
            _, predicted = torch.max(outputs_avg, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            
            for i in range(target.size(0)):
                label = target[i].item()
                class_total[label] += 1
                if predicted[i] == target[i]:
                    class_correct[label] += 1
    
    # Calculate overall accuracy
    total_correct = sum(class_correct)
    total_samples = sum(class_total)
    overall_accuracy = 100 * total_correct / total_samples
    
    print(f"\n EXPERT CNN TEST RESULTS")
    print("=" * 50)
    print(" Per-Class Accuracy (Enhanced TTA):")
    print("-" * 50)
    
    worst_class_acc = 100.0
    classes_above_95 = 0
    classes_above_90 = 0
    
    for i in range(10):
        if class_total[i] > 0:
            class_acc = 100 * class_correct[i] / class_total[i]
            worst_class_acc = min(worst_class_acc, class_acc)
            if class_acc >= 95:
                classes_above_95 += 1
                status = ""
            elif class_acc >= 90:
                classes_above_90 += 1
                status = ""
            else:
                status = ""
            print(f"  {status} {class_names[i]:12}: {class_acc:6.2f}% ({class_correct[i]:3d}/{class_total[i]:3d})")
    
    print(f"\n OVERALL PERFORMANCE:")
    print(f"   Overall Accuracy: {overall_accuracy:.2f}%")
    print(f"   Worst Class: {worst_class_acc:.2f}%")
    print(f"   Classes ≥95%: {classes_above_95}/10")
    print(f"   Classes ≥90%: {classes_above_90 + classes_above_95}/10")
    
    if overall_accuracy >= 95:
        grade = "A++ OUTSTANDING"
        status = " TARGET ACHIEVED! 95%+"
        emoji = "⭐"
    elif overall_accuracy >= 94:
        grade = "A+ EXCELLENT"
        status = " SO CLOSE! Almost there!"
        emoji = "⭐"
    elif overall_accuracy >= 93:
        grade = "A VERY GOOD"
        status = " Great progress!"
        emoji = "💪"
    else:
        grade = "B+ GOOD"
        status = " Good work!"
        emoji = "💪👍"
    
    print(f"   Grade: {grade}")
    print(f"   Status: {status}")
    print(f"   {emoji}")
    
    # Show some sample predictions
    print(f"\n SAMPLE PREDICTIONS:")
    print("-" * 30)
    
    # Get a few samples for visualization
    sample_data, sample_targets = next(iter(test_loader))
    sample_data = sample_data[:8].to(device)  # First 8 samples
    sample_targets = sample_targets[:8]
    
    with torch.no_grad():
        # TTA for samples
        predictions = []
        predictions.append(model(sample_data))
        predictions.append(model(torch.flip(sample_data, dims=[3])))
        outputs_avg = torch.stack(predictions).mean(dim=0)
        _, predicted = torch.max(outputs_avg, 1)
        confidences = F.softmax(outputs_avg, dim=1).max(dim=1)[0]
    
    for i in range(8):
        actual = class_names[sample_targets[i]]
        pred = class_names[predicted[i]]
        conf = confidences[i].item()
        status = "" if sample_targets[i] == predicted[i] else ""
        print(f"  {status} Actual: {actual:12} | Predicted: {pred:12} | Confidence: {conf:.3f}")
    
    print(f"\n Expert CNN model evaluation complete!")
    print(f" Model is ready for deployment and inference!")
    
    return {
        'overall_accuracy': overall_accuracy,
        'class_accuracies': [100 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0 for i in range(10)],
        'worst_class_acc': worst_class_acc,
        'classes_above_95': classes_above_95,
        'grade': grade
    }


if __name__ == "__main__":
    results = test_expert_cnn()