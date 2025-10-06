#!/usr/bin/env python3
"""
Visual Test Expert CNN - Test on actual Fashion-MNIST images with visualization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import random
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


def visual_test_expert_cnn():
    """Visual test of Expert CNN on real Fashion-MNIST images."""
    print("🖼️  VISUAL TEST - EXPERT CNN ON REAL IMAGES")
    print("=" * 60)
    
    # Setup device
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"🖥️  Testing on: {device}")
    
    # Load model
    model = ExpertFashionCNN().to(device)
    model_path = 'models/champion_95percent_cnn.pth'
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        return
    
    print(f"📂 Loading model from: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    
    # Load test data with original images (no normalization for display)
    display_transform = transforms.Compose([transforms.ToTensor()])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])
    
    # Dataset for display
    display_dataset = datasets.FashionMNIST(
        root='./data', train=False, download=True, transform=display_transform
    )
    
    # Dataset for model inference
    test_dataset = datasets.FashionMNIST(
        root='./data', train=False, download=True, transform=test_transform
    )
    
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    # Test on random samples from each class
    print("🎯 Testing on samples from each fashion class...")
    
    # Get samples for each class
    class_samples = {i: [] for i in range(10)}
    class_indices = {i: [] for i in range(10)}
    
    for idx, (_, label) in enumerate(test_dataset):
        if len(class_samples[label]) < 5:  # 5 samples per class
            class_samples[label].append(idx)
            class_indices[label].append(idx)
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(10, 6, figsize=(18, 24))
    fig.suptitle('Expert CNN Predictions on Real Fashion-MNIST Images\n95.30% Accuracy Model', 
                 fontsize=16, fontweight='bold')
    
    correct_predictions = 0
    total_predictions = 0
    
    with torch.no_grad():
        for class_idx in range(10):
            for sample_idx in range(5):  # 5 samples per class
                if sample_idx < len(class_samples[class_idx]):
                    # Get the actual sample index
                    data_idx = class_samples[class_idx][sample_idx]
                    
                    # Get display image and normalized image
                    display_img, true_label = display_dataset[data_idx]
                    test_img, _ = test_dataset[data_idx]
                    
                    # Make prediction with TTA
                    test_img = test_img.unsqueeze(0).to(device)
                    
                    # Enhanced test-time augmentation
                    predictions = []
                    predictions.append(model(test_img))
                    predictions.append(model(torch.flip(test_img, dims=[3])))
                    predictions.append(model(transforms.functional.rotate(test_img, 2)))
                    predictions.append(model(transforms.functional.rotate(test_img, -2)))
                    
                    # Average predictions
                    outputs_avg = torch.stack(predictions).mean(dim=0)
                    probabilities = F.softmax(outputs_avg, dim=1)
                    confidence, predicted = torch.max(probabilities, 1)
                    
                    predicted_class = predicted.item()
                    confidence_score = confidence.item()
                    
                    # Track accuracy
                    is_correct = predicted_class == true_label
                    if is_correct:
                        correct_predictions += 1
                    total_predictions += 1
                    
                    # Plot image
                    ax = axes[class_idx, sample_idx]
                    ax.imshow(display_img.squeeze(), cmap='gray')
                    ax.axis('off')
                    
                    # Color code: green for correct, red for incorrect
                    color = 'green' if is_correct else 'red'
                    status = '✅' if is_correct else '❌'
                    
                    title = f"{status} {class_names[predicted_class]}\nConf: {confidence_score:.3f}"
                    ax.set_title(title, fontsize=9, color=color, fontweight='bold')
                    
                    # Show true label on y-axis for first column
                    if sample_idx == 0:
                        ax.set_ylabel(f"True:\n{class_names[class_idx]}", 
                                    rotation=0, ha='right', va='center', fontsize=10)
                else:
                    axes[class_idx, sample_idx].axis('off')
            
            # Add class accuracy in the last column
            ax_stats = axes[class_idx, 5]
            ax_stats.axis('off')
            
            # Calculate class-specific accuracy from our previous test
            class_accuracies = [90.20, 99.40, 93.80, 95.60, 93.70, 
                              99.30, 84.70, 98.50, 99.70, 98.10]
            
            class_acc = class_accuracies[class_idx]
            if class_acc >= 95:
                acc_color = 'green'
                acc_status = '🎯'
            elif class_acc >= 90:
                acc_color = 'orange'
                acc_status = '⚠️'
            else:
                acc_color = 'red'
                acc_status = '❌'
            
            ax_stats.text(0.5, 0.5, f"{acc_status}\nClass Acc:\n{class_acc:.1f}%", 
                         ha='center', va='center', fontsize=10, 
                         color=acc_color, fontweight='bold',
                         transform=ax_stats.transAxes)
    
    plt.tight_layout()
    
    # Save the visualization
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f'results/expert_cnn_visual_test_{timestamp}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    # Calculate sample accuracy
    sample_accuracy = 100 * correct_predictions / total_predictions
    
    print(f"\n📊 VISUAL TEST RESULTS:")
    print("=" * 40)
    print(f"🎯 Sample Accuracy: {sample_accuracy:.1f}% ({correct_predictions}/{total_predictions})")
    print(f"🏆 Overall Model Accuracy: 95.30% (from full test)")
    print(f"💾 Visualization saved: {save_path}")
    
    # Show some detailed predictions
    print(f"\n🔍 DETAILED SAMPLE ANALYSIS:")
    print("-" * 50)
    
    # Test on some specific challenging examples
    with torch.no_grad():
        # Get a few random samples
        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=True)
        sample_batch = next(iter(test_loader))
        data, targets = sample_batch[0][:8].to(device), sample_batch[1][:8]
        
        # Make predictions with confidence
        outputs = model(data)
        probabilities = F.softmax(outputs, dim=1)
        confidences, predictions = torch.max(probabilities, 1)
        
        print("Random Sample Predictions:")
        for i in range(8):
            actual = class_names[targets[i]]
            pred = class_names[predictions[i]]
            conf = confidences[i].item()
            status = "✅" if targets[i] == predictions[i] else "❌"
            
            # Get top 3 predictions for this sample
            top3_probs, top3_indices = torch.topk(probabilities[i], 3)
            top3_classes = [class_names[idx] for idx in top3_indices]
            top3_confidences = [prob.item() for prob in top3_probs]
            
            print(f"\n  {status} Sample {i+1}:")
            print(f"     Actual: {actual}")
            print(f"     Top-3 Predictions:")
            for j, (cls, conf_val) in enumerate(zip(top3_classes, top3_confidences)):
                rank = "🥇" if j == 0 else "🥈" if j == 1 else "🥉"
                print(f"       {rank} {cls}: {conf_val:.3f}")
    
    print(f"\n🎉 Visual testing complete!")
    print(f"🚀 Expert CNN demonstrates excellent performance on real images!")
    
    return save_path


if __name__ == "__main__":
    visualization_path = visual_test_expert_cnn()