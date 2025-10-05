#!/usr/bin/env python3
"""
Enhanced VAE Usage Example
Shows how to use the Enhanced VAE for conditional fashion generation.
"""

import torch
import matplotlib.pyplot as plt
from enhanced_vae import EnhancedVAE

def generate_specific_fashion_items():
    """Generate specific fashion items using Enhanced VAE."""
    # Setup device
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
    # Load model
    model = EnhancedVAE(latent_dim=32, conditional=True).to(device)
    model.load_state_dict(torch.load('models/enhanced_vae_superior.pth', map_location=device))
    model.eval()
    
    # Fashion class names
    class_names = [
        'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
    ]
    
    # Generate specific items
    with torch.no_grad():
        # Generate 3 dresses
        dresses = model.generate_fashion_class(3, num_samples=3, device=device)
        
        # Generate 3 sneakers
        sneakers = model.generate_fashion_class(7, num_samples=3, device=device)
        
        # Generate one item from each class
        collection = []
        for class_idx in range(10):
            item = model.generate_fashion_class(class_idx, num_samples=1, device=device)
            collection.append(item)
    
    # Visualize specific items
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    
    # Show dresses
    for i in range(3):
        axes[0, i].imshow(dresses[i].squeeze().cpu().numpy(), cmap='gray')
        axes[0, i].set_title(f'Dress {i+1}')
        axes[0, i].axis('off')
    
    # Show sneakers
    for i in range(3):
        axes[1, i].imshow(sneakers[i].squeeze().cpu().numpy(), cmap='gray')
        axes[1, i].set_title(f'Sneaker {i+1}')
        axes[1, i].axis('off')
    
    plt.suptitle('Conditional Fashion Generation - Enhanced VAE')
    plt.tight_layout()
    plt.show()
    
    # Show complete collection
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for i in range(10):
        row, col = i // 5, i % 5
        axes[row, col].imshow(collection[i][0].squeeze().cpu().numpy(), cmap='gray')
        axes[row, col].set_title(class_names[i])
        axes[row, col].axis('off')
    
    plt.suptitle('Complete Fashion Collection - One Item Per Class')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    generate_specific_fashion_items()
