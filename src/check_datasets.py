"""
Check available PyTorch datasets and basic setup

This script demonstrates:
1. How to check PyTorch installation
2. What built-in datasets are available
3. How to download and load the MNIST dataset
4. Basic dataset properties and structure
"""

# Import necessary PyTorch libraries
import torch                          # Core PyTorch library for tensors and neural networks
import torchvision                    # Computer vision specific tools and datasets
import torchvision.transforms as transforms  # Image preprocessing transformations
from torchvision import datasets      # Pre-built datasets like MNIST, CIFAR10, etc.

# Check PyTorch installation and capabilities
print("=== PyTorch Installation Check ===")
print("PyTorch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)
print("CUDA available:", torch.cuda.is_available())  # CUDA = GPU acceleration (False = using CPU only)

# Built-in datasets you can use for learning:
print("\n=== Popular PyTorch Datasets (no external database needed) ===")
print("1. MNIST - Handwritten digits (28x28 grayscale)")        # Classic ML dataset, 60k training images
print("2. CIFAR10 - 10 classes of 32x32 color images")          # airplanes, cars, birds, cats, etc.
print("3. CIFAR100 - 100 classes of 32x32 color images")        # More complex version of CIFAR10
print("4. Fashion-MNIST - Fashion items (28x28 grayscale)")     # T-shirts, shoes, bags instead of digits
print("5. SVHN - Street View House Numbers")                    # Real-world house numbers from Google Street View
print("6. STL10 - 10 classes unlabeled dataset")               # Larger images (96x96) for unsupervised learning

# Example: Loading MNIST dataset
print("\n=== Example: Loading MNIST dataset ===")

# Define image transformations (preprocessing steps)
transform = transforms.Compose([
    transforms.ToTensor(),           # Convert PIL Image to PyTorch tensor (0-1 range)
    transforms.Normalize((0.5,), (0.5,))  # Normalize: mean=0.5, std=0.5 → converts range to (-1, 1)
])
# Why normalize? Neural networks train better when input data is centered around 0

# Download MNIST dataset (downloads automatically first time)
try:
    # Load training dataset
    train_dataset = datasets.MNIST(
        root='./data',           # Directory to save/load data
        train=True,              # Load training set (60,000 images)
        download=True,           # Download if not already present
        transform=transform      # Apply the transformations we defined above
    )
    
    # Load test dataset  
    test_dataset = datasets.MNIST(
        root='./data',           # Same directory as training data
        train=False,             # Load test set (10,000 images)
        download=True,           # Download if not already present
        transform=transform      # Apply same transformations for consistency
    )
    
    # Display dataset information
    print(f"Training samples: {len(train_dataset)}")     # Should be 60,000
    print(f"Test samples: {len(test_dataset)}")          # Should be 10,000
    print(f"Image shape: {train_dataset[0][0].shape}")   # [channels, height, width] = [1, 28, 28]
    print(f"Number of classes: {len(train_dataset.classes)}")  # 10 classes (digits 0-9)
    
    # Additional information about the dataset structure
    print(f"Classes: {train_dataset.classes}")           # ['0', '1', '2', ..., '9']
    
    # Show what a single data point looks like
    sample_image, sample_label = train_dataset[0]        # Get first training example
    print(f"Single image tensor shape: {sample_image.shape}")  # torch.Size([1, 28, 28])
    print(f"Single label: {sample_label}")               # Integer 0-9
    print(f"Image tensor min/max values: {sample_image.min():.3f} to {sample_image.max():.3f}")
    
except Exception as e:
    print(f"Error loading dataset: {e}")
    print("This might happen if there's no internet connection or disk space issues")
