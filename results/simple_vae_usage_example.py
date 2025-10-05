#!/usr/bin/env python3
"""
Simple VAE Usage Example
Shows how to use the Simple VAE for fashion item generation.
"""

import torch
import matplotlib.pyplot as plt
from simple_generator import SimpleVAE

def generate_fashion_items():
    """Generate fashion items using Simple VAE."""
    # Setup device
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
    # Load model
    model = SimpleVAE(latent_dim=20).to(device)
    model.load_state_dict(torch.load('models/simple_vae.pth', map_location=device))
    model.eval()
    
    # Generate samples
    with torch.no_grad():
        samples = model.generate(num_samples=16, device=device)
    
    # Visualize
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    for i in range(16):
        row, col = i // 4, i % 4
        axes[row, col].imshow(samples[i].cpu().numpy(), cmap='gray')
        axes[row, col].axis('off')
    
    plt.suptitle('Generated Fashion Items - Simple VAE')
    plt.show()

if __name__ == "__main__":
    generate_fashion_items()
