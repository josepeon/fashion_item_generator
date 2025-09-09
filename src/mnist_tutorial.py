"""
PyTorch MNIST Tutorial - Complete Beginner's Guide

This comprehensive script covers:
1. PyTorch installation verification
2. Dataset loading and exploration
3. Data visualization
4. Data loaders for efficient training
5. Tensor operations and understanding

Follow along to understand the fundamentals before building neural networks.
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
import matplotlib.pyplot as plt
import numpy as np
import time

def check_pytorch_setup():
    """Check PyTorch installation and capabilities"""
    print("=" * 50)
    print("🔧 PYTORCH SETUP VERIFICATION")
    print("=" * 50)
    print(f"PyTorch version: {torch.__version__}")
    print(f"Torchvision version: {torchvision.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU for computation")
    print()

def load_mnist_dataset():
    """Load and explore the MNIST dataset"""
    print("=" * 50)
    print("📊 LOADING MNIST DATASET")
    print("=" * 50)
    
    # Define transformations
    # ToTensor(): Converts PIL Image to tensor [0,1]
    # Normalize(): Converts [0,1] to [-1,1] for better training
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Load training and test datasets
    train_dataset = datasets.MNIST(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform
    )
    
    test_dataset = datasets.MNIST(
        root='./data', 
        train=False, 
        download=True, 
        transform=transform
    )
    
    # Dataset information
    print(f"✅ Training samples: {len(train_dataset):,}")
    print(f"✅ Test samples: {len(test_dataset):,}")
    print(f"✅ Classes: {train_dataset.classes}")
    print(f"✅ Number of classes: {len(train_dataset.classes)}")
    
    # Examine a single sample
    sample_image, sample_label = train_dataset[0]
    print(f"\n📋 Sample Data Point:")
    print(f"   Image shape: {sample_image.shape}")
    print(f"   Image type: {type(sample_image)}")
    print(f"   Label: {sample_label} (type: {type(sample_label)})")
    print(f"   Pixel value range: [{sample_image.min():.3f}, {sample_image.max():.3f}]")
    print(f"   Total pixels: {sample_image.numel()}")
    
    return train_dataset, test_dataset

def visualize_samples(dataset, num_samples=8, save_path=None):
    """Visualize sample images from the dataset"""
    print("=" * 50)
    print("👁️  VISUALIZING SAMPLE IMAGES")
    print("=" * 50)
    
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    fig.suptitle('MNIST Sample Digits', fontsize=16, fontweight='bold')
    
    for i in range(num_samples):
        image_tensor, label = dataset[i]
        
        # Convert tensor to numpy array for display
        # Remove channel dimension: [1, 28, 28] -> [28, 28]
        image_np = image_tensor.squeeze().numpy()
        
        # Plot in grid
        row, col = i // 4, i % 4
        axes[row, col].imshow(image_np, cmap='gray')
        axes[row, col].set_title(f'Label: {label}', fontweight='bold')
        axes[row, col].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📸 Sample images saved to: {save_path}")
    
    plt.show()
    plt.close()

def create_data_loaders(train_dataset, test_dataset, batch_size=64):
    """Create efficient data loaders for training"""
    print("=" * 50)
    print("🔄 CREATING DATA LOADERS")
    print("=" * 50)
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,           # Shuffle training data
        num_workers=0,          # Use 0 for macOS compatibility
        pin_memory=False
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,          # Don't shuffle test data
        num_workers=0,
        pin_memory=False
    )
    
    print(f"✅ Batch size: {batch_size}")
    print(f"✅ Training batches: {len(train_loader):,}")
    print(f"✅ Test batches: {len(test_loader):,}")
    print(f"✅ Images per training batch: {batch_size}")
    print(f"✅ Total training images: {len(train_loader) * batch_size:,}")
    
    return train_loader, test_loader

def explore_batched_data(data_loader):
    """Explore the structure of batched data"""
    print("=" * 50)
    print("🔍 EXPLORING BATCHED DATA")
    print("=" * 50)
    
    # Get one batch
    data_iter = iter(data_loader)
    batch_images, batch_labels = next(data_iter)
    
    print(f"📦 Batch Information:")
    print(f"   Images shape: {batch_images.shape}")  # [batch_size, channels, height, width]
    print(f"   Labels shape: {batch_labels.shape}")  # [batch_size]
    
    print(f"\n📊 Batch Contents:")
    print(f"   Batch size: {batch_images.size(0)}")
    print(f"   Channels: {batch_images.size(1)}")
    print(f"   Height: {batch_images.size(2)}")
    print(f"   Width: {batch_images.size(3)}")
    
    print(f"\n🏷️  Labels in this batch:")
    print(f"   {batch_labels.tolist()}")
    
    return batch_images, batch_labels

def tensor_operations_demo(tensor):
    """Demonstrate basic tensor operations"""
    print("=" * 50)
    print("🧮 TENSOR OPERATIONS DEMO")
    print("=" * 50)
    
    print(f"📊 Tensor Statistics:")
    print(f"   Shape: {tensor.shape}")
    print(f"   Data type: {tensor.dtype}")
    print(f"   Device: {tensor.device}")
    print(f"   Requires gradient: {tensor.requires_grad}")
    print(f"   Memory usage: {tensor.element_size() * tensor.numel()} bytes")
    
    print(f"\n📈 Statistical Values:")
    print(f"   Min: {tensor.min():.4f}")
    print(f"   Max: {tensor.max():.4f}")
    print(f"   Mean: {tensor.mean():.4f}")
    print(f"   Std: {tensor.std():.4f}")
    print(f"   Sum: {tensor.sum():.4f}")
    
    print(f"\n🔄 Shape Operations:")
    print(f"   Original: {tensor.shape}")
    print(f"   Flattened: {tensor.flatten().shape}")
    print(f"   Reshaped to [1, -1]: {tensor.reshape(1, -1).shape}")

def main():
    """Main tutorial function"""
    print("🚀 PYTORCH MNIST TUTORIAL - COMPLETE GUIDE")
    print("=" * 60)
    print("Welcome to PyTorch! Let's explore the MNIST dataset step by step.")
    print("=" * 60)
    
    # Step 1: Check setup
    check_pytorch_setup()
    
    # Step 2: Load dataset
    train_dataset, test_dataset = load_mnist_dataset()
    
    # Step 3: Visualize samples
    print("\n⏳ Generating sample visualization...")
    visualize_samples(train_dataset, save_path='mnist_samples.png')
    
    # Step 4: Create data loaders
    train_loader, test_loader = create_data_loaders(train_dataset, test_dataset)
    
    # Step 5: Explore batched data
    batch_images, batch_labels = explore_batched_data(train_loader)
    
    # Step 6: Tensor operations demo
    sample_tensor = batch_images[0]  # First image in batch
    tensor_operations_demo(sample_tensor)
    
    # Summary
    print("\n" + "=" * 60)
    print("🎉 TUTORIAL COMPLETE!")
    print("=" * 60)
    print("✅ PyTorch setup verified")
    print("✅ MNIST dataset loaded and explored")
    print("✅ Data visualization created")
    print("✅ Data loaders configured")
    print("✅ Tensor operations demonstrated")
    print("\n🎯 Next Steps:")
    print("   1. Build your first neural network")
    print("   2. Train on MNIST data")
    print("   3. Evaluate model performance")
    print("   4. Make predictions on new images")
    print("=" * 60)

if __name__ == "__main__":
    main()
