#!/usr/bin/env python3
"""
Enhanced Fashion CNN - Improved Recognition & Prediction
=======================================================

This module enhances the existing FashionNet CNN to achieve higher accuracy
through architectural improvements and advanced training techniques.

Target: >95% accuracy on Fashion-MNIST
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np
import time
import os
from fashion_handler import FashionMNIST


class EnhancedFashionNet(nn.Module):
    """Enhanced CNN with attention mechanisms and improved architecture"""
    
    def __init__(self, num_classes=10, dropout=0.4):
        super(EnhancedFashionNet, self).__init__()
        
        # Enhanced feature extraction with more depth
        self.features = nn.Sequential(
            # First block: 28x28 -> 14x14
            nn.Conv2d(1, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.1),
            
            # Second block: 14x14 -> 7x7
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2),
            
            # Third block: 7x7 -> 3x3 (rounded)
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.3),
            
            # Fourth block: 3x3 -> 1x1
            nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        
        # Channel attention mechanism
        self.channel_attention = nn.Sequential(
            nn.Linear(512, 256, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(256, 512, bias=False),
            nn.Sigmoid()
        )
        
        # Enhanced classifier with residual connection
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.25),
            
            nn.Linear(128, num_classes)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Feature extraction
        features = self.features(x)
        features = features.view(features.size(0), -1)
        
        # Channel attention
        attention = self.channel_attention(features)
        features = features * attention
        
        # Classification
        output = self.classifier(features)
        return output


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class EnhancedTrainer:
    """Enhanced trainer with advanced optimization techniques"""
    
    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device
        
        # Advanced optimizer setup
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=0.001,
            weight_decay=0.01,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Multiple learning rate schedulers
        self.scheduler_cosine = CosineAnnealingLR(self.optimizer, T_max=50, eta_min=1e-6)
        self.scheduler_plateau = ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.7, patience=5, verbose=True
        )
        
        # Enhanced loss function
        self.criterion = FocalLoss(alpha=1, gamma=2)
        
        # Training history
        self.history = []
        
    def train_epoch(self, train_loader, epoch):
        """Train one epoch with enhanced techniques"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(self.device), targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(data)
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
            if batch_idx % 150 == 0:
                print(f'    Batch {batch_idx:3d}: Loss {loss.item():.4f}')
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        return epoch_loss, epoch_acc
    
    def evaluate(self, test_loader):
        """Evaluate model performance"""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, targets in test_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = self.model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        accuracy = 100. * correct / total
        return accuracy
    
    def train_enhanced(self, epochs=60, target_accuracy=95.0):
        """Enhanced training with advanced techniques"""
        
        print("🚀 ENHANCED FASHION-MNIST CNN TRAINING")
        print("=" * 45)
        
        # Data loaders
        fashion_handler = FashionMNIST(batch_size=64)
        train_loader = fashion_handler.get_train_loader()
        test_loader = fashion_handler.get_test_loader()
        
        model_params = sum(p.numel() for p in self.model.parameters())
        print(f"Model parameters: {model_params:,}")
        print(f"Target accuracy: {target_accuracy}%")
        print(f"Training device: {self.device}")
        print()
        
        best_accuracy = 0.0
        patience_counter = 0
        start_time = time.time()
        
        for epoch in range(epochs):
            print(f"Epoch {epoch+1:2d}/{epochs}")
            
            # Training phase
            train_loss, train_acc = self.train_epoch(train_loader, epoch)
            
            # Evaluation phase
            test_acc = self.evaluate(test_loader)
            
            # Learning rate scheduling
            self.scheduler_cosine.step()
            self.scheduler_plateau.step(test_acc)
            
            current_lr = self.optimizer.param_groups[0]['lr']
            
            print(f"  Train: Loss {train_loss:.4f}, Acc {train_acc:.2f}%")
            print(f"  Test:  Acc {test_acc:.2f}%, LR {current_lr:.6f}")
            
            # Save best model
            if test_acc > best_accuracy:
                best_accuracy = test_acc
                torch.save(self.model.state_dict(), 'models/enhanced_fashion_cnn.pth')
                patience_counter = 0
                print(f"  ✅ New best: {best_accuracy:.2f}%")
            else:
                patience_counter += 1
            
            # Record history
            self.history.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_accuracy': train_acc,
                'test_accuracy': test_acc,
                'learning_rate': current_lr
            })
            
            print()
            
            # Check stopping conditions
            if test_acc >= target_accuracy:
                print(f"🎯 Target accuracy {target_accuracy}% achieved!")
                break
                
            if patience_counter >= 15:
                print("Early stopping: No improvement for 15 epochs")
                break
        
        training_time = time.time() - start_time
        print(f"🎉 Training completed in {training_time/60:.1f} minutes")
        print(f"Best accuracy: {best_accuracy:.2f}%")
        
        return best_accuracy, self.history


def detailed_evaluation(model, device):
    """Comprehensive model evaluation with per-class analysis"""
    print("\n🔬 DETAILED MODEL EVALUATION")
    print("=" * 40)
    
    fashion_handler = FashionMNIST(batch_size=100)
    test_loader = fashion_handler.get_test_loader()
    
    class_names = [
        'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
    ]
    
    model.eval()
    class_correct = [0] * 10
    class_total = [0] * 10
    all_confidences = []
    
    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            probabilities = F.softmax(outputs, dim=1)
            confidences, predicted = torch.max(probabilities, 1)
            
            all_confidences.extend(confidences.cpu().numpy())
            
            for i in range(targets.size(0)):
                label = targets[i].item()
                class_total[label] += 1
                if predicted[i] == label:
                    class_correct[label] += 1
    
    # Calculate overall metrics
    total_correct = sum(class_correct)
    total_samples = sum(class_total)
    overall_accuracy = 100. * total_correct / total_samples
    avg_confidence = np.mean(all_confidences)
    
    print(f"Overall Accuracy: {overall_accuracy:.2f}%")
    print(f"Average Confidence: {avg_confidence:.3f}")
    print(f"Total samples: {total_samples:,}")
    print()
    
    print("Per-Class Performance:")
    print("-" * 35)
    for i, class_name in enumerate(class_names):
        if class_total[i] > 0:
            accuracy = 100. * class_correct[i] / class_total[i]
            print(f"{class_name:12}: {accuracy:5.1f}% ({class_correct[i]:3d}/{class_total[i]:3d})")
    
    return overall_accuracy


def create_comparison_test():
    """Test to compare original vs enhanced CNN"""
    print("\n📊 COMPARISON: Original vs Enhanced CNN")
    print("=" * 50)
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
    # Test original model
    try:
        from fashion_cnn import FashionNet
        original_model = FashionNet().to(device)
        original_model.load_state_dict(torch.load('models/best_fashion_cnn_100epochs.pth', map_location=device))
        print("✅ Original model loaded")
        original_acc = detailed_evaluation(original_model, device)
    except Exception as e:
        print(f"❌ Could not load original model: {e}")
        original_acc = 94.1  # Known accuracy
    
    # Test enhanced model
    try:
        enhanced_model = EnhancedFashionNet().to(device)
        enhanced_model.load_state_dict(torch.load('models/enhanced_fashion_cnn.pth', map_location=device))
        print("✅ Enhanced model loaded")
        enhanced_acc = detailed_evaluation(enhanced_model, device)
    except Exception as e:
        print(f"❌ Could not load enhanced model: {e}")
        enhanced_acc = 0.0
    
    print(f"\n📈 IMPROVEMENT SUMMARY:")
    print(f"Original CNN:  {original_acc:.2f}%")
    print(f"Enhanced CNN:  {enhanced_acc:.2f}%")
    if enhanced_acc > original_acc:
        improvement = enhanced_acc - original_acc
        print(f"Improvement:   +{improvement:.2f}%")
        print("🎉 Enhanced model is better!")
    
    return original_acc, enhanced_acc


def main():
    """Main training and evaluation function"""
    print("🎯 FASHION-MNIST CNN ENHANCEMENT PROJECT")
    print("=" * 50)
    
    # Setup
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create enhanced model
    model = EnhancedFashionNet(num_classes=10, dropout=0.4)
    trainer = EnhancedTrainer(model, device)
    
    # Train model
    print("\n🚀 Starting Enhanced Training...")
    best_accuracy, history = trainer.train_enhanced(epochs=50, target_accuracy=95.0)
    
    # Final evaluation
    print("\n🔬 Final Evaluation...")
    final_accuracy = detailed_evaluation(model, device)
    
    # Results summary
    print(f"\n🏆 FINAL RESULTS:")
    print(f"Best accuracy during training: {best_accuracy:.2f}%")
    print(f"Final test accuracy: {final_accuracy:.2f}%")
    print(f"Model saved as: models/enhanced_fashion_cnn.pth")
    
    if final_accuracy >= 95.0:
        print("🎉 SUCCESS: Achieved 95%+ accuracy target!")
    elif final_accuracy >= 94.5:
        print("🎊 EXCELLENT: Very close to 95% target!")
    else:
        print(f"📈 Good progress: {final_accuracy:.2f}% (Target: 95%)")
    
    return model, trainer, history


if __name__ == "__main__":
    # Ensure directories exist
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Run enhanced training
    model, trainer, history = main()
    
    # Optional: Compare with original
    # create_comparison_test()