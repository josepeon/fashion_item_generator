#!/usr/bin/env python3
"""
Quick Visual Demo - Show Expert CNN predictions on a few key examples
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os


class ExpertFashionCNN(nn.Module):
    """Expert CNN architecture"""
    def __init__(self, num_classes=10):
        super(ExpertFashionCNN, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=5, padding=2),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.1),
            
            nn.Conv2d(96, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2),
            
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            nn.Conv2d(384, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        
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


def quick_demo():
    """Quick demo of Expert CNN on Fashion-MNIST images."""
    print(" QUICK DEMO - Expert CNN on Fashion Images")
    print("=" * 50)
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
    # Load model
    model = ExpertFashionCNN().to(device)
    model.load_state_dict(torch.load('models/champion_95percent_cnn.pth', map_location=device, weights_only=True))
    model.eval()
    
    # Load data
    display_transform = transforms.Compose([transforms.ToTensor()])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])
    
    display_dataset = datasets.FashionMNIST(root='./data', train=False, transform=display_transform)
    test_dataset = datasets.FashionMNIST(root='./data', train=False, transform=test_transform)
    
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    # Get one example from each class
    class_examples = {}
    for idx, (_, label) in enumerate(test_dataset):
        if label not in class_examples:
            class_examples[label] = idx
        if len(class_examples) == 10:
            break
    
    # Create simple visualization
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig.suptitle('Expert CNN Predictions - One Example per Class\n95.30% Accuracy Model', 
                 fontsize=14, fontweight='bold')
    
    correct_count = 0
    
    with torch.no_grad():
        for i, (class_idx, data_idx) in enumerate(class_examples.items()):
            row = i // 5
            col = i % 5
            
            # Get images
            display_img, true_label = display_dataset[data_idx]
            test_img, _ = test_dataset[data_idx]
            
            # Make prediction
            test_img = test_img.unsqueeze(0).to(device)
            outputs = model(test_img)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            predicted_class = predicted.item()
            confidence_score = confidence.item()
            is_correct = predicted_class == true_label
            
            if is_correct:
                correct_count += 1
            
            # Plot
            ax = axes[row, col]
            ax.imshow(display_img.squeeze(), cmap='gray')
            ax.axis('off')
            
            # Title with prediction
            status = "✓" if is_correct else "✗"
            color = 'green' if is_correct else 'red'
            
            title = f"{status} True: {class_names[true_label]}\nPred: {class_names[predicted_class]} ({confidence_score:.3f})"
            ax.set_title(title, fontsize=10, color=color)
    
    plt.tight_layout()
    plt.savefig('results/expert_cnn_quick_demo.png', dpi=200, bbox_inches='tight')
    plt.show()
    
    print(f"\n Quick Demo Results:")
    print(f"   Sample Accuracy: {correct_count}/10 = {100*correct_count/10:.0f}%")
    print(f"   Overall Model: 95.30% accuracy")
    print(f"   Saved: results/expert_cnn_quick_demo.png")
    
    # Show detailed predictions for a few samples
    print(f"\n Detailed Analysis:")
    test_loader = DataLoader(test_dataset, batch_size=3, shuffle=True)
    data, targets = next(iter(test_loader))
    data = data.to(device)
    
    with torch.no_grad():
        outputs = model(data)
        probabilities = F.softmax(outputs, dim=1)
        
        for i in range(3):
            true_class = class_names[targets[i]]
            
            # Get top 3 predictions
            top3_probs, top3_indices = torch.topk(probabilities[i], 3)
            
            print(f"\n  Sample {i+1} - True: {true_class}")
            for j, (prob, idx) in enumerate(zip(top3_probs, top3_indices)):
                pred_class = class_names[idx]
                confidence = prob.item()
                rank = ["", "", ""][j]
                status = "← CORRECT" if idx == targets[i] else ""
                print(f"    {rank} {pred_class}: {confidence:.3f} {status}")


if __name__ == "__main__":
    quick_demo()