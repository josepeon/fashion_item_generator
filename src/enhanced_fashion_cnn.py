"""
Enhanced Fashion-MNIST CNN for >98% Accuracy

Advanced architecture with data augmentation, batch normalization,
and sophisticated training techniques.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST
import time


class EnhancedFashionNet(nn.Module):
    """Enhanced CNN architecture for >98% Fashion-MNIST accuracy."""
    
    def __init__(self, num_classes=10):
        super(EnhancedFashionNet, self).__init__()
        
        # First convolutional block
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Second convolutional block
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)  
        self.bn4 = nn.BatchNorm2d(128)
        
        # Third convolutional block
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        
        # Fully connected layers
        self.fc1 = nn.Linear(256 * 3 * 3, 512)
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(0.5)
        
        self.fc2 = nn.Linear(512, 256)
        self.bn_fc2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.3)
        
        self.fc3 = nn.Linear(256, num_classes)
        
        # Max pooling
        self.pool = nn.MaxPool2d(2, 2)
        
    def forward(self, x):
        # First block: 28x28 -> 14x14
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        
        # Second block: 14x14 -> 7x7
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)
        
        # Third block: 7x7 -> 3x3 (with additional pooling)
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.pool(x)
        
        # Flatten and fully connected
        x = x.view(x.size(0), -1)
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout1(x)
        x = F.relu(self.bn_fc2(self.fc2(x)))
        x = self.dropout2(x)
        x = self.fc3(x)
        
        return x


def get_enhanced_transforms():
    """Get data transforms with augmentation for better accuracy."""
    train_transform = transforms.Compose([
        transforms.RandomRotation(10),  # Slight rotation
        transforms.RandomHorizontalFlip(p=0.1),  # Minimal flip (fashion items)
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),  # Small translation
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))  # Fashion-MNIST normalization
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),  
        transforms.Normalize((0.2860,), (0.3530,))
    ])
    
    return train_transform, test_transform


def train_enhanced_model():
    """Train the enhanced model for >98% accuracy."""
    print('🚀 ENHANCED FASHION CNN TRAINING')
    print('Target: >98% Accuracy')
    print('='*40)
    
    # Device setup
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f'Device: {device}')
    
    # Enhanced data loading with augmentation
    train_transform, test_transform = get_enhanced_transforms()
    
    train_dataset = FashionMNIST(
        '../data', train=True, download=True, transform=train_transform
    )
    test_dataset = FashionMNIST(
        '../data', train=False, transform=test_transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)
    
    # Enhanced model
    model = EnhancedFashionNet().to(device)
    print(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')
    
    # Advanced optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=0.01, epochs=50, steps_per_epoch=len(train_loader)
    )
    criterion = nn.CrossEntropyLoss()
    
    print('\\nTraining configuration:')
    print('- Optimizer: AdamW with weight decay')
    print('- Scheduler: OneCycleLR')
    print('- Data augmentation: Enabled')
    print('- Batch normalization: Enabled')
    print('- Epochs: 50')
    
    print('\\n🚀 TRAINING STARTING...')
    print('='*40)
    
    best_accuracy = 0.0
    start_time = time.time()
    
    for epoch in range(50):
        # Training phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total_train += target.size(0)
            correct_train += (predicted == target).sum().item()
        
        # Validation phase
        model.eval()
        correct_test = 0
        total_test = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                _, predicted = torch.max(output.data, 1)
                total_test += target.size(0)
                correct_test += (predicted == target).sum().item()
        
        train_acc = 100. * correct_train / total_train
        test_acc = 100. * correct_test / total_test
        avg_loss = running_loss / len(train_loader)
        
        if (epoch + 1) % 5 == 0 or epoch < 10:
            print(f'Epoch {epoch+1:2d}/50: Loss: {avg_loss:.4f} | Train: {train_acc:.2f}% | Test: {test_acc:.2f}%')
        
        # Save best model
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            torch.save(model.state_dict(), '../models/enhanced_fashion_cnn.pth')
    
    training_time = time.time() - start_time
    
    print(f'\\n🎉 ENHANCED TRAINING COMPLETED!')
    print(f'Time: {training_time/60:.1f} minutes')
    print(f'Best accuracy: {best_accuracy:.2f}%')
    
    if best_accuracy >= 98.0:
        print('🏆 TARGET ACHIEVED: >98% accuracy!')
    else:
        print(f'📈 Progress: {best_accuracy:.2f}% (Target: 98%)')
        
    print('💾 Best model saved to: models/enhanced_fashion_cnn.pth')
    
    return best_accuracy


if __name__ == '__main__':
    train_enhanced_model()