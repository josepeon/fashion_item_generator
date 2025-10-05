"""
Quick Fashion Generator - Simple Autoencoder

A simpler approach to generate Fashion-MNIST images using a Variational Autoencoder.
This trains much faster than GAN and gives good results for learning.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple
import time
import os
from datetime import datetime

from fashion_handler import FashionMNIST


class VAEGenerator(nn.Module):
    """Variational Autoencoder for generating Fashion-MNIST images."""
    
    def __init__(self, latent_dim: int = 20):
        super(VAEGenerator, self).__init__()
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(784, 400),
            nn.ReLU(),
            nn.Linear(400, 200),
            nn.ReLU()
        )
        
        # Latent layers
        self.mu_layer = nn.Linear(200, latent_dim)
        self.logvar_layer = nn.Linear(200, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 200),
            nn.ReLU(),
            nn.Linear(200, 400),
            nn.ReLU(),
            nn.Linear(400, 784),
            nn.Sigmoid()  # Output between 0 and 1
        )
    
    def encode(self, x):
        """Encode input to latent parameters."""
        h = self.encoder(x)
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Sample from latent distribution."""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu
    
    def decode(self, z):
        """Decode latent vector to image."""
        return self.decoder(z)
    
    def forward(self, x):
        """Full forward pass."""
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar
    
    def generate(self, num_samples: int = 8, device: str = 'cpu') -> torch.Tensor:
        """Generate new samples."""
        self.eval()
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim).to(device)
            samples = self.decode(z)
            return samples.view(-1, 1, 28, 28)


def vae_loss(recon_x, x, mu, logvar, beta: float = 1.0):
    """VAE loss function with KL divergence."""
    # Reconstruction loss
    recon_loss = nn.functional.binary_cross_entropy(
        recon_x, x.view(-1, 784), reduction='sum'
    )
    
    # KL divergence loss
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return recon_loss + beta * kl_loss


class QuickTrainer:
    """Simple trainer for the VAE."""
    
    def __init__(self, model: VAEGenerator, lr: float = 1e-3):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        # Optimal device selection for performance
        if torch.cuda.is_available():
            self.device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = 'mps'  # Apple Silicon GPU
        else:
            self.device = 'cpu'
        self.model.to(self.device)
        
        self.train_losses = []
    
    def train_epoch(self, train_loader):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(self.device)
            
            # Forward pass
            recon_batch, mu, logvar = self.model(data)
            loss = vae_loss(recon_batch, data, mu, logvar)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f'    Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.2f}')
        
        avg_loss = total_loss / len(train_loader.dataset)
        self.train_losses.append(avg_loss)
        return avg_loss
    
    def train(self, train_loader, epochs: int = 10):
        """Train the model."""
        print(f"Training VAE for {epochs} epochs on {self.device}")
        start_time = time.time()
        
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            loss = self.train_epoch(train_loader)
            print(f"  Average loss: {loss:.4f}")
        
        training_time = time.time() - start_time
        print(f"\nTraining completed in {training_time:.2f} seconds")
    
    def visualize_results(self, original_data, save_path: str = None):
        """Show original vs reconstructed images."""
        self.model.eval()
        
        with torch.no_grad():
            # Get a batch of test data
            test_input = original_data[:8].to(self.device)
            recon, _, _ = self.model(test_input)
            
            # Generate new samples
            generated = self.model.generate(8, self.device)
        
        # Create visualization
        fig, axes = plt.subplots(3, 8, figsize=(16, 6))
        
        # Original images
        for i in range(8):
            axes[0, i].imshow(test_input[i].cpu().squeeze(), cmap='gray')
            axes[0, i].set_title('Original')
            axes[0, i].axis('off')
        
        # Reconstructed images
        for i in range(8):
            axes[1, i].imshow(recon[i].cpu().view(28, 28), cmap='gray')
            axes[1, i].set_title('Reconstructed')
            axes[1, i].axis('off')
        
        # Generated images
        for i in range(8):
            axes[2, i].imshow(generated[i].cpu().squeeze(), cmap='gray')
            axes[2, i].set_title('Generated')
            axes[2, i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Results saved to {save_path}")
        
        plt.show()
        plt.close()
    
    def plot_loss(self, save_path: str = None):
        """Plot training loss."""
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_losses)
        plt.title('VAE Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Loss plot saved to {save_path}")
        
        plt.show()
        plt.close()


def generate_fashion_grid(model: VAEGenerator, device: str, num_samples: int = 64):
    """Generate a grid of fashion items."""
    model.eval()
    
    with torch.no_grad():
        # Generate samples
        samples = model.generate(num_samples, device)
    
    # Create grid
    grid_size = int(np.sqrt(num_samples))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    fig.suptitle('Generated Fashion Items', fontsize=16, fontweight='bold')
    
    for i in range(grid_size):
        for j in range(grid_size):
            idx = i * grid_size + j
            axes[i, j].imshow(samples[idx].cpu().squeeze(), cmap='gray')
            axes[i, j].axis('off')
    
    plt.tight_layout()
    
    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f'results/generated_fashion_grid_{timestamp}.png'
    os.makedirs('results', exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Fashion grid saved to {save_path}")
    
    plt.show()
    plt.close()
    
    return save_path


def main():
    """Main training and generation pipeline."""
    print("Quick Fashion-MNIST VAE Generator")
    print("=" * 40)
    
    # Load Fashion-MNIST data
    fashion = FashionMNIST(batch_size=128)
    train_loader = fashion.train_loader
    
    print("Dataset loaded:")
    info = fashion.info()
    for key, value in info.items():
        if key == 'class_names':
            print(f"  {key}: {', '.join(value[:3])}... (and 7 more)")
        else:
            print(f"  {key}: {value}")
    
    # Create and train model
    model = VAEGenerator(latent_dim=20)
    trainer = QuickTrainer(model, lr=1e-3)
    
    print(f"\nModel created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Train the model
    trainer.train(train_loader, epochs=20)
    
    # Visualize results
    test_data, _ = next(iter(fashion.test_loader))
    trainer.visualize_results(test_data, 'results/fashion_vae_comparison.png')
    
    # Plot training loss
    trainer.plot_loss('results/fashion_vae_loss.png')
    
    # Generate a grid of samples
    generate_fashion_grid(model, trainer.device, num_samples=64)
    
    # Save model
    torch.save(model.state_dict(), 'models/quick_fashion_vae.pth')
    print("Model saved to models/quick_fashion_vae.pth")
    
    print("\nQuick Fashion VAE training completed! 🎉")


if __name__ == "__main__":
    main()