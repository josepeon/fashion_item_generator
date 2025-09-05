"""
PyTorch Learning Step 2: Data Visualization and Loading

This script demonstrates:
1. How to visualize MNIST images
2. How to create data loaders for efficient batching
3. Understanding tensor operations
4. Preparing data for model training
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
import matplotlib.pyplot as plt
import numpy as np

print("=== Step 2: Data Visualization and Loading ===\n")

# Define the same transformation as before
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load MNIST dataset (already downloaded from previous script)
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)

print(f"✅ Loaded MNIST: {len(train_dataset)} training, {len(test_dataset)} test samples")

# ========================================
# 1. VISUALIZING THE DATA
# ========================================
print("\n1️⃣ === VISUALIZING MNIST DIGITS ===")

def show_sample_images(dataset, num_samples=8):
    """
    Display a grid of sample images from the dataset
    """
    fig, axes = plt.subplots(2, 4, figsize=(10, 5))
    fig.suptitle('Sample MNIST Digits', fontsize=16)
    
    for i in range(num_samples):
        # Get a sample from the dataset
        image_tensor, label = dataset[i]
        
        # Convert tensor back to numpy for matplotlib
        # Remember: our images are normalized to [-1, 1], so we need to denormalize
        image_np = image_tensor.squeeze().numpy()  # Remove channel dimension [1, 28, 28] → [28, 28]
        
        # Plot the image
        row, col = i // 4, i % 4
        axes[row, col].imshow(image_np, cmap='gray')
        axes[row, col].set_title(f'Label: {label}')
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig('mnist_samples.png', dpi=150, bbox_inches='tight')
    print("📸 Saved sample images to 'mnist_samples.png'")
    plt.close()

# Show some sample images
show_sample_images(train_dataset)

# ========================================
# 2. UNDERSTANDING TENSOR OPERATIONS
# ========================================
print("\n2️⃣ === UNDERSTANDING TENSOR OPERATIONS ===")

# Get a single sample and explore it
sample_image, sample_label = train_dataset[0]
print(f"Sample image shape: {sample_image.shape}")
print(f"Sample image type: {type(sample_image)}")
print(f"Sample label: {sample_label} (type: {type(sample_label)})")

# Tensor operations
print(f"\nTensor statistics:")
print(f"  Min value: {sample_image.min():.3f}")
print(f"  Max value: {sample_image.max():.3f}")
print(f"  Mean value: {sample_image.mean():.3f}")
print(f"  Standard deviation: {sample_image.std():.3f}")

# Tensor dimensions
print(f"\nTensor dimensions:")
print(f"  Number of dimensions: {sample_image.ndim}")
print(f"  Shape: {sample_image.shape}")
print(f"  Total elements: {sample_image.numel()}")

# ========================================
# 3. DATA LOADERS FOR BATCHING
# ========================================
print("\n3️⃣ === CREATING DATA LOADERS ===")

# DataLoaders help us:
# - Batch data efficiently
# - Shuffle training data
# - Load data in parallel
# - Handle different batch sizes

batch_size = 64  # Common batch size (try 32, 64, 128)

# Create data loaders
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,    # Number of samples per batch
    shuffle=True,             # Shuffle training data each epoch
    num_workers=0,            # Set to 0 for macOS compatibility (avoids multiprocessing issues)
    pin_memory=False          # Set to False when num_workers=0
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=batch_size,    # Same batch size for consistency
    shuffle=False,            # Don't shuffle test data
    num_workers=0,            # Set to 0 for macOS compatibility
    pin_memory=False          # Set to False when num_workers=0
)

print(f"✅ Created data loaders:")
print(f"  Batch size: {batch_size}")
print(f"  Training batches: {len(train_loader)}")  # 60,000 / 64 ≈ 938 batches
print(f"  Test batches: {len(test_loader)}")       # 10,000 / 64 ≈ 157 batches

# ========================================
# 4. EXPLORING BATCHED DATA
# ========================================
print("\n4️⃣ === EXPLORING BATCHED DATA ===")

# Get one batch from the training loader
data_iter = iter(train_loader)
batch_images, batch_labels = next(data_iter)

print(f"Batch shapes:")
print(f"  Images: {batch_images.shape}")  # [batch_size, channels, height, width]
print(f"  Labels: {batch_labels.shape}")  # [batch_size]

print(f"\nBatch contents:")
print(f"  Batch size: {batch_images.size(0)}")
print(f"  Image channels: {batch_images.size(1)}")
print(f"  Image height: {batch_images.size(2)}")
print(f"  Image width: {batch_images.size(3)}")

# Show the labels in this batch
print(f"\nLabels in this batch: {batch_labels.tolist()}")

# ========================================
# 5. DATA LOADING PERFORMANCE
# ========================================
print("\n5️⃣ === DATA LOADING PERFORMANCE ===")

# Time how long it takes to iterate through one epoch
import time

start_time = time.time()
batch_count = 0

# Iterate through first 10 batches (not full epoch for demo)
for batch_idx, (images, labels) in enumerate(train_loader):
    batch_count += 1
    if batch_idx >= 9:  # Stop after 10 batches for demo
        break

end_time = time.time()
print(f"⏱️  Processed {batch_count} batches in {end_time - start_time:.2f} seconds")
print(f"   That's {batch_count * batch_size} images!")

# ========================================
# 6. NEXT STEPS SUMMARY
# ========================================
print("\n6️⃣ === WHAT WE'VE LEARNED ===")
print("✅ How to visualize image data")
print("✅ Understanding tensor shapes and operations")
print("✅ Creating efficient data loaders")
print("✅ Working with batched data")
print("✅ Performance considerations")

print("\n🎯 === NEXT STEPS ===")
print("1. Build a simple neural network (coming next!)")
print("2. Define loss function and optimizer")
print("3. Train the model on MNIST")
print("4. Evaluate model performance")
print("5. Make predictions on new data")

print(f"\n📁 Files created:")
print(f"  - mnist_samples.png (sample digit images)")
print(f"  - Data loaded and ready for model training!")
