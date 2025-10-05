#!/usr/bin/env python3
"""
Improved Fashion-MNIST CNN Training Script
Implements improvements identified from performance analysis.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt
from datetime import datetime

from fashion_handler import FashionMNIST
from test_cnn_performance import SimpleCNNTester

# Import the improved model
from analyze_cnn_improvements import ImprovedFashionCNN, TrainingImprovements

def train_improved_model():
    """Train the improved CNN model."""
    print("🚀 TRAINING IMPROVED FASHION-MNIST CNN")
    print("=" * 60)
    
    # Setup device
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Training on: {device}")
    
    # Load data with augmentation
    fashion_data = FashionMNIST(batch_size=128)
    train_loader = fashion_data.get_train_loader()
    test_loader = fashion_data.get_test_loader()
    
    # Create improved model
    model = ImprovedFashionCNN().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Setup training with improvements
    class_weights = TrainingImprovements.get_class_weights().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
    
    # Training loop
    model.train()
    for epoch in range(100):  # More epochs for better convergence
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(data)
            
            # Use focal loss for hard examples
            loss = TrainingImprovements.focal_loss(outputs, targets)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch+1}/100, Batch {batch_idx}, '
                      f'Loss: {loss.item():.4f}, '
                      f'Acc: {100.*correct/total:.2f}%')
        
        scheduler.step()
        
        # Save checkpoint every 25 epochs
        if (epoch + 1) % 25 == 0:
            torch.save(model.state_dict(), f'models/improved_cnn_epoch_{epoch+1}.pth')
            print(f"Checkpoint saved at epoch {epoch+1}")
    
    # Save final model
    torch.save(model.state_dict(), 'models/improved_fashion_cnn.pth')
    print("✅ Training completed!")
    
    # Test the improved model
    print("\n🧪 Testing improved model...")
    tester = SimpleCNNTester('models/improved_fashion_cnn.pth')
    results = tester.run_comprehensive_test()
    
    return model, results

if __name__ == "__main__":
    train_improved_model()
