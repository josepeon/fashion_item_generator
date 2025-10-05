#!/usr/bin/env python3
"""
Simple Fashion-MNIST Generator
A working VAE that actually generates fashion items properly.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from fashion_handler import FashionMNIST


class SimpleVAE(nn.Module):
    """Simple VAE that actually works for Fashion-MNIST generation."""
    
    def __init__(self, latent_dim=20):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Encoder: 784 -> 400 -> 20*2
        self.encoder = nn.Sequential(
            nn.Linear(784, 400),
            nn.ReLU(),
            nn.Linear(400, latent_dim * 2)  # mu and logvar
        )
        
        # Decoder: 20 -> 400 -> 784
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 400),
            nn.ReLU(),
            nn.Linear(400, 784),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        """Encode input to latent distribution parameters."""
        x = x.view(-1, 784)
        h = self.encoder(x)
        mu, logvar = h.chunk(2, dim=1)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        """Decode latent vector to image."""
        return self.decoder(z)
    
    def forward(self, x):
        """Full VAE forward pass."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
    
    def generate(self, num_samples=1, device='cpu'):
        """Generate new fashion items."""
        self.eval()
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim).to(device)
            samples = self.decode(z)
            return samples.view(-1, 28, 28)


def vae_loss(recon_x, x, mu, logvar):
    """VAE loss function."""
    # Reconstruction loss
    recon_loss = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    
    # KL divergence loss
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return recon_loss + kl_loss


def train_simple_vae(epochs=20):
    """Train a simple VAE on Fashion-MNIST."""
    print("🎨 Training Simple Fashion-MNIST VAE")
    print("=" * 40)
    
    # Setup
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Data
    fashion = FashionMNIST(batch_size=128)
    train_loader = fashion.get_train_loader()
    
    # Model
    model = SimpleVAE(latent_dim=20).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss = vae_loss(recon_batch, data, mu, logvar)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f'  Epoch {epoch+1:2d} [{batch_idx:3d}/{len(train_loader)}] Loss: {loss.item():.0f}')
        
        avg_loss = total_loss / len(train_loader.dataset)
        print(f'Epoch {epoch+1:2d} Average Loss: {avg_loss:.4f}')
    
    # Save model
    torch.save(model.state_dict(), '../models/simple_vae.pth')
    print("✅ Model saved to: models/simple_vae.pth")
    
    return model


def test_generation():
    """Test generation with trained VAE."""
    print("\n🖼️  Testing Fashion Item Generation")
    print("=" * 40)
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
    # Load model
    model = SimpleVAE(latent_dim=20).to(device)
    try:
        model.load_state_dict(torch.load('../models/simple_vae.pth', map_location=device))
        print("✅ Loaded trained VAE model")
    except:
        print("⚠️  No trained model found, using untrained model")
    
    # Generate samples
    model.eval()
    with torch.no_grad():
        samples = model.generate(num_samples=16, device=device)
        
        # Create visualization
        fig, axes = plt.subplots(4, 4, figsize=(10, 10))
        fig.suptitle('Generated Fashion Items (Simple VAE)', fontsize=16)
        
        for i in range(16):
            row = i // 4
            col = i % 4
            axes[row, col].imshow(samples[i].cpu().numpy(), cmap='gray')
            axes[row, col].axis('off')
            axes[row, col].set_title(f'Sample {i+1}')
        
        plt.tight_layout()
        plt.savefig('../results/simple_vae_generation.png', dpi=150, bbox_inches='tight')
        print("💾 Generated samples saved to: results/simple_vae_generation.png")
        
        # Show reconstruction
        fashion = FashionMNIST(batch_size=8)
        test_loader = fashion.get_test_loader()
        data_iter = iter(test_loader)
        real_images, _ = next(data_iter)
        real_images = real_images.to(device)
        
        # Reconstruct
        recon_images, _, _ = model(real_images)
        recon_images = recon_images.view(-1, 28, 28)
        
        # Visualization
        fig, axes = plt.subplots(2, 8, figsize=(16, 4))
        fig.suptitle('Real vs Reconstructed Fashion Items', fontsize=16)
        
        for i in range(8):
            # Real images
            axes[0, i].imshow(real_images[i, 0].cpu().numpy(), cmap='gray')
            axes[0, i].set_title('Real')
            axes[0, i].axis('off')
            
            # Reconstructed images
            axes[1, i].imshow(recon_images[i].cpu().numpy(), cmap='gray')
            axes[1, i].set_title('Reconstructed')
            axes[1, i].axis('off')
        
        plt.tight_layout()
        plt.savefig('../results/simple_vae_reconstruction.png', dpi=150, bbox_inches='tight')
        print("💾 Reconstructions saved to: results/simple_vae_reconstruction.png")


def main():
    """Main function to train and test VAE."""
    print("🎯 SIMPLE FASHION-MNIST GENERATOR")
    print("=" * 50)
    
    choice = input("\nWhat would you like to do?\n1. Train new VAE\n2. Test generation\n3. Both\nChoice (1-3): ")
    
    if choice in ['1', '3']:
        model = train_simple_vae(epochs=10)  # Quick training
    
    if choice in ['2', '3']:
        test_generation()
    
    print("\n✅ Done! Check the results/ folder for generated images.")


if __name__ == "__main__":
    main()