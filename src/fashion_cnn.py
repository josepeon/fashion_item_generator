"""
Fashion-MNIST CNN Model - Convolutional Neural Network

A clean, efficient CNN implementation for Fashion-MNIST item classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from typing import Tuple, List
import time

from fashion_handler import FashionMNIST


class FashionNet(nn.Module):
    """Convolutional Neural Network for Fashion-MNIST classification."""
    
    def __init__(self):
        super(FashionNet, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)    # 28x28 -> 28x28
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)   # 14x14 -> 14x14
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # 7x7 -> 7x7
        
        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)  # Reduces size by half
        
        # Dropout for regularization
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 3 * 3, 512)  # 128 channels * 3x3 after pooling
        self.fc2 = nn.Linear(512, 10)           # 10 output classes (fashion items)
    
    def forward(self, x):
        """Forward pass through the network."""
        # First conv block: Conv -> ReLU -> Pool
        x = self.pool(F.relu(self.conv1(x)))    # [batch, 32, 14, 14]
        
        # Second conv block: Conv -> ReLU -> Pool  
        x = self.pool(F.relu(self.conv2(x)))    # [batch, 64, 7, 7]
        
        # Third conv block: Conv -> ReLU -> Pool
        x = self.pool(F.relu(self.conv3(x)))    # [batch, 128, 3, 3]
        
        # Flatten for fully connected layers
        x = x.view(-1, 128 * 3 * 3)             # [batch, 1152]
        x = self.dropout1(x)
        
        # First fully connected layer
        x = F.relu(self.fc1(x))                 # [batch, 512]
        x = self.dropout2(x)
        
        # Output layer (no activation - CrossEntropyLoss handles it)
        x = self.fc2(x)                         # [batch, 10]
        
        return x


class CNNTrainer:
    """Handles training and evaluation of the CNN."""
    
    def __init__(self, model: nn.Module, device: str = None):
        self.model = model
        # Optimal device selection for performance
        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = 'mps'  # Apple Silicon GPU
        else:
            self.device = 'cpu'
        self.model.to(self.device)
        
        # Training components
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Training history
        self.train_losses = []
        self.train_accuracies = []
        self.test_accuracies = []
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        
        return epoch_loss, epoch_acc
    
    def evaluate(self, test_loader: DataLoader) -> float:
        """Evaluate the model on test data."""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        return accuracy
    
    def train(self, train_loader: DataLoader, test_loader: DataLoader, epochs: int = 10):
        """Train the model for specified epochs."""
        print(f"Training on {self.device}")
        print(f"Model has {sum(p.numel() for p in self.model.parameters()):,} parameters")
        
        start_time = time.time()
        
        for epoch in range(epochs):
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Evaluate
            test_acc = self.evaluate(test_loader)
            
            # Store history
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            self.test_accuracies.append(test_acc)
            
            # Print progress
            print(f"Epoch {epoch+1}/{epochs}: "
                  f"Loss: {train_loss:.4f} | "
                  f"Train Acc: {train_acc:.2f}% | "
                  f"Test Acc: {test_acc:.2f}%")
        
        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time:.2f} seconds")
        print(f"Final test accuracy: {self.test_accuracies[-1]:.2f}%")
    
    def plot_training_history(self, save_path: str = None):
        """Plot training history."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Loss plot
        ax1.plot(self.train_losses, 'b-', label='Training Loss')
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plot
        ax2.plot(self.train_accuracies, 'b-', label='Training Accuracy')
        ax2.plot(self.test_accuracies, 'r-', label='Test Accuracy')
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Training history saved to {save_path}")
        
        plt.show()
        plt.close()
    
    def predict_samples(self, test_loader: DataLoader, num_samples: int = 8):
        """Show predictions on sample images."""
        self.model.eval()
        images, labels = next(iter(test_loader))
        images, labels = images.to(self.device), labels.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(images)
            _, predicted = torch.max(outputs, 1)
        
        # Get class names for better visualization
        fashion = FashionMNIST()
        class_names = fashion.CLASS_NAMES
        
        # Plot predictions
        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        fig.suptitle('CNN Predictions on Fashion Items', fontweight='bold')
        
        for i in range(num_samples):
            image = images[i].cpu().squeeze()
            true_label = labels[i].cpu().item()
            pred_label = predicted[i].cpu().item()
            
            row, col = i // 4, i % 4
            axes[row, col].imshow(image, cmap='gray')
            
            # Color code: green if correct, red if wrong
            color = 'green' if true_label == pred_label else 'red'
            axes[row, col].set_title(f'True: {class_names[true_label]}\nPred: {class_names[pred_label]}', 
                                   color=color, fontsize=9)
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.show()
        plt.close()


def main():
    """Main training pipeline."""
    print("Fashion-MNIST CNN Training Pipeline")
    print("=" * 40)
    
    # Initialize data
    fashion = FashionMNIST(batch_size=128)  # Larger batch size for CNN
    train_loader = fashion.get_train_loader()
    test_loader = fashion.get_test_loader()
    
    print(f"Dataset loaded: {fashion.info()}")
    
    # Create model
    model = FashionNet()
    trainer = CNNTrainer(model)
    
    # Train model
    print("\nStarting training...")
    trainer.train(train_loader, test_loader, epochs=10)
    
    # Plot results
    trainer.plot_training_history('training_history.png')
    
    # Show predictions
    trainer.predict_samples(test_loader)
    
    # Save model
    torch.save(model.state_dict(), 'fashion_cnn.pth')
    print("Model saved to fashion_cnn.pth")


if __name__ == "__main__":
    main()