"""
Quick MNIST Generator - Simple Autoencoder

A simpler approach to generate MNIST-like images using a Variational Autoencoder.
This trains much faster than GAN and gives good results for learning.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple
import time

from mnist_handler import MNIST


class VAEGenerator(nn.Module):
    """Variational Autoencoder for generating MNIST images."""
    
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
        
        # Latent space
        self.fc_mu = nn.Linear(200, latent_dim)
        self.fc_var = nn.Linear(200, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 200),
            nn.ReLU(),
            nn.Linear(200, 400),
            nn.ReLU(),
            nn.Linear(400, 784),
            nn.Tanh()  # Output between -1 and 1
        )
    
    def encode(self, x):
        """Encode input to latent space."""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        log_var = self.fc_var(h)
        return mu, log_var
    
    def reparameterize(self, mu, log_var):
        """Reparameterization trick."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        """Decode from latent space to image."""
        return self.decoder(z)
    
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var


class QuickGenerator:
    """Quick and simple generator trainer."""
    
    def __init__(self, latent_dim: int = 20, lr: float = 1e-3, device: str = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.latent_dim = latent_dim
        
        # Create model
        self.model = VAEGenerator(latent_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        # Training history
        self.losses = []
        
        print(f"Quick Generator initialized on {self.device}")
        print(f"Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def vae_loss(self, recon_x, x, mu, log_var):
        """VAE loss function."""
        # Reconstruction loss
        recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
        
        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        
        return recon_loss + kl_loss
    
    def train(self, dataloader, epochs: int = 20):
        """Train the VAE."""
        print(f"\nTraining Quick Generator for {epochs} epochs...")
        start_time = time.time()
        
        for epoch in range(epochs):
            total_loss = 0
            num_batches = 0
            
            for batch_idx, (images, _) in enumerate(dataloader):
                images = images.view(images.size(0), -1).to(self.device)
                
                self.optimizer.zero_grad()
                
                # Forward pass
                recon_images, mu, log_var = self.model(images)
                loss = self.vae_loss(recon_images, images, mu, log_var)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches
            self.losses.append(avg_loss)
            
            print(f"Epoch [{epoch+1}/{epochs}] | Loss: {avg_loss:.2f}")
        
        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time:.2f} seconds")
        
        # Save model
        torch.save(self.model.state_dict(), 'quick_generator.pth')
        print("Model saved: quick_generator.pth")
    
    def generate_images(self, num_images: int = 16):
        """Generate new images."""
        self.model.eval()
        
        with torch.no_grad():
            # Sample from standard normal distribution
            z = torch.randn(num_images, self.latent_dim).to(self.device)
            generated = self.model.decode(z)
            generated = generated.view(num_images, 1, 28, 28)
        
        return generated.cpu()
    
    def show_generated_images(self, num_images: int = 16):
        """Display generated images."""
        fake_images = self.generate_images(num_images)
        
        fig, axes = plt.subplots(4, 4, figsize=(8, 8))
        fig.suptitle('🚀 Quick Generated MNIST Digits', fontweight='bold')
        
        for i in range(num_images):
            row, col = i // 4, i % 4
            image = fake_images[i].squeeze()
            axes[row, col].imshow(image, cmap='gray')
            axes[row, col].set_title(f'#{i+1}')
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.savefig('quick_generated_digits.png', dpi=150, bbox_inches='tight')
        plt.show()
        plt.close()
    
    def plot_training_history(self):
        """Plot training loss."""
        plt.figure(figsize=(8, 5))
        plt.plot(self.losses, 'b-', linewidth=2)
        plt.title('Quick Generator Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.savefig('quick_generator_loss.png', dpi=150, bbox_inches='tight')
        plt.show()
        plt.close()


def main():
    """Main quick generator pipeline."""
    print("Quick MNIST Generator - Fast Results!")
    print("=" * 40)
    
    # Load data
    mnist = MNIST(batch_size=128)
    train_loader = mnist.train_loader
    
    print(f"Dataset loaded: {mnist.info()}")
    
    # Create and train generator
    generator = QuickGenerator(latent_dim=20)
    generator.train(train_loader, epochs=15)
    
    # Generate and show results
    print("\n🎨 Generating new digits...")
    generator.show_generated_images()
    generator.plot_training_history()
    
    print("\nQuick generation complete!")
    print("Check 'quick_generated_digits.png' for results")


if __name__ == "__main__":
    main()
