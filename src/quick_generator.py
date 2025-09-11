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
import os
from datetime import datetime

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


class EnhancedVAETrainer:
    """Enhanced VAE trainer with advanced features."""
    
    def __init__(self, latent_dim: int = 20, lr: float = 1e-3, device: str = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.latent_dim = latent_dim
        
        # Create model
        self.model = VAEGenerator(latent_dim).to(self.device)
        
        # Enhanced optimizer and scheduler
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
        
        # Training history
        self.losses = []
        self.recon_losses = []
        self.kl_losses = []
        self.learning_rates = []
        
        # Early stopping
        self.best_loss = float('inf')
        self.patience = 15
        self.patience_counter = 0
        self.early_stop = False
        
        print(f"Enhanced VAE Trainer initialized on {self.device}")
        print(f"Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Ready for up to 200 epochs with early stopping")
    
    def vae_loss(self, recon_x, x, mu, log_var):
        """Enhanced VAE loss function with detailed tracking."""
        # Reconstruction loss (MSE)
        recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
        
        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        
        return recon_loss + kl_loss, recon_loss.item(), kl_loss.item()
    
    def save_checkpoint(self, epoch: int, loss: float, is_best: bool = False):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
            'best_loss': self.best_loss,
            'losses': self.losses,
            'recon_losses': self.recon_losses,
            'kl_losses': self.kl_losses,
            'learning_rates': self.learning_rates
        }
        
        if is_best:
            torch.save(checkpoint, 'models/best_vae_generator.pth')
            print(f"New best model saved! Loss: {loss:.4f}")
        
        # Save regular checkpoint every 10 epochs
        if epoch % 10 == 0:
            torch.save(checkpoint, f'checkpoints/vae_generator_epoch_{epoch:03d}.pth')
            print(f"Checkpoint saved: epoch {epoch}")
    
    def enhanced_train(self, dataloader, test_loader=None, epochs: int = 200):
        """Enhanced training with all advanced features."""
        print(f"\nStarting Enhanced VAE Training")
        print(f"Target: {epochs} epochs (early stopping enabled)")
        print("=" * 60)
        
        start_time = time.time()
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            total_loss = 0
            total_recon = 0
            total_kl = 0
            num_batches = 0
            
            for batch_idx, (images, _) in enumerate(dataloader):
                images = images.view(images.size(0), -1).to(self.device)
                
                self.optimizer.zero_grad()
                
                # Forward pass
                recon_images, mu, log_var = self.model(images)
                loss, recon_loss, kl_loss = self.vae_loss(recon_images, images, mu, log_var)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                total_recon += recon_loss
                total_kl += kl_loss
                num_batches += 1
            
            # Calculate averages
            avg_loss = total_loss / num_batches
            avg_recon = total_recon / num_batches
            avg_kl = total_kl / num_batches
            
            # Store metrics
            self.losses.append(avg_loss)
            self.recon_losses.append(avg_recon)
            self.kl_losses.append(avg_kl)
            self.learning_rates.append(self.optimizer.param_groups[0]['lr'])
            
            # Learning rate scheduling
            self.scheduler.step(avg_loss)
            
            # Check for improvement
            is_best = avg_loss < self.best_loss
            if is_best:
                self.best_loss = avg_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            # Save checkpoints
            self.save_checkpoint(epoch + 1, avg_loss, is_best)
            
            # Progress reporting
            if (epoch + 1) % 5 == 0 or epoch < 5:
                current_time = time.time() - start_time
                print(f"Epoch [{epoch+1:3d}/{epochs}] | "
                      f"Loss: {avg_loss:7.2f} | "
                      f"Recon: {avg_recon:6.2f} | "
                      f"KL: {avg_kl:6.2f} | "
                      f"LR: {self.optimizer.param_groups[0]['lr']:.2e} | "
                      f"Time: {current_time:6.1f}s")
            
            # Early stopping check
            if self.patience_counter >= self.patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                print(f"No improvement for {self.patience} consecutive epochs")
                self.early_stop = True
                break
        
        # Training completion
        total_time = time.time() - start_time
        final_epoch = epoch + 1
        
        print(f"\nEnhanced VAE Training Complete")
        print("=" * 60)
        print(f"Final Results:")
        print(f"   • Epochs trained: {final_epoch}")
        print(f"   • Best loss: {self.best_loss:.4f}")
        print(f"   • Final loss: {avg_loss:.4f}")
        print(f"   • Training time: {total_time:.2f}s ({total_time/60:.1f}m)")
        print(f"   • Early stopped: {'Yes' if self.early_stop else 'No'}")
        
        # Save final model
        torch.save(self.model.state_dict(), f'models/vae_generator_final_{self.best_loss:.1f}.pth')
        print(f"Final model saved: models/vae_generator_final_{self.best_loss:.1f}.pth")
        
        return final_epoch, self.best_loss
    
    def generate_images(self, num_images: int = 16):
        """Generate new images with enhanced model."""
        self.model.eval()
        
        with torch.no_grad():
            # Sample from standard normal distribution
            z = torch.randn(num_images, self.latent_dim).to(self.device)
            generated = self.model.decode(z)
            generated = generated.view(num_images, 1, 28, 28)
        
        return generated.cpu()
    
    def show_enhanced_results(self, num_images: int = 16):
        """Display enhanced training results."""
        fake_images = self.generate_images(num_images)
        
        fig, axes = plt.subplots(4, 4, figsize=(10, 10))
        fig.suptitle('Enhanced VAE Generated MNIST Digits', fontsize=16, fontweight='bold')
        
        for i in range(num_images):
            row, col = i // 4, i % 4
            image = fake_images[i].squeeze()
            axes[row, col].imshow(image, cmap='gray')
            axes[row, col].set_title(f'Generated #{i+1}', fontsize=10)
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.savefig('results/enhanced_vae_generated_digits.png', dpi=150, bbox_inches='tight')
        plt.show()
        plt.close()
    
    def plot_enhanced_training_history(self):
        """Plot comprehensive training history."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Enhanced VAE Training History', fontsize=16, fontweight='bold')
        
        epochs = range(1, len(self.losses) + 1)
        
        # Total Loss
        ax1.plot(epochs, self.losses, 'b-', linewidth=2, label='Total Loss')
        ax1.set_title('Total Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.grid(True)
        ax1.legend()
        
        # Component Losses
        ax2.plot(epochs, self.recon_losses, 'r-', linewidth=2, label='Reconstruction')
        ax2.plot(epochs, self.kl_losses, 'g-', linewidth=2, label='KL Divergence')
        ax2.set_title('Loss Components')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.grid(True)
        ax2.legend()
        
        # Learning Rate
        ax3.plot(epochs, self.learning_rates, 'orange', linewidth=2)
        ax3.set_title('Learning Rate Schedule')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        ax3.set_yscale('log')
        ax3.grid(True)
        
        # Training Progress
        ax4.plot(epochs, self.losses, 'b-', linewidth=2, alpha=0.7)
        if len(self.losses) > 0:
            best_epoch = np.argmin(self.losses) + 1
            ax4.axvline(x=best_epoch, color='red', linestyle='--', alpha=0.7, label=f'Best: Epoch {best_epoch}')
            ax4.scatter([best_epoch], [min(self.losses)], color='red', s=100, zorder=5)
        ax4.set_title('Training Progress')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Total Loss')
        ax4.grid(True)
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig('results/enhanced_vae_training_history.png', dpi=150, bbox_inches='tight')
        plt.show()
        plt.close()
        
        print(f"Training history saved: results/enhanced_vae_training_history.png")


class QuickGenerator:
    """Quick and simple generator trainer (legacy for compatibility)."""
    
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
        torch.save(self.model.state_dict(), 'models/quick_generator.pth')
        print("Model saved: models/quick_generator.pth")
    
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
        fig.suptitle('Quick Generated MNIST Digits', fontweight='bold')
        
        for i in range(num_images):
            row, col = i // 4, i % 4
            image = fake_images[i].squeeze()
            axes[row, col].imshow(image, cmap='gray')
            axes[row, col].set_title(f'#{i+1}')
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.savefig('results/quick_generated_digits.png', dpi=150, bbox_inches='tight')
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
        plt.savefig('results/quick_generator_loss.png', dpi=150, bbox_inches='tight')
        plt.show()
        plt.close()


def run_enhanced_training():
    """Run enhanced VAE training with 200 epochs."""
    print("Enhanced VAE Generator Training")
    print("Up to 200 epochs with advanced features")
    print("=" * 60)
    
    # Load data
    mnist = MNIST(batch_size=128)
    train_loader = mnist.train_loader
    test_loader = mnist.test_loader
    
    print(f"Dataset loaded: {mnist.info()}")
    
    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True) 
    os.makedirs('checkpoints', exist_ok=True)
    
    # Create and train enhanced generator
    trainer = EnhancedVAETrainer(latent_dim=20, lr=1e-3)
    
    # Check if we already have a trained model
    if os.path.exists('models/best_vae_generator.pth'):
        response = input("\nEnhanced VAE model already exists. Retrain? (y/N): ").lower()
        if response != 'y':
            print("Using existing enhanced model")
            checkpoint = torch.load('models/best_vae_generator.pth', map_location=trainer.device)
            trainer.model.load_state_dict(checkpoint['model_state_dict'])
            trainer.losses = checkpoint.get('losses', [])
            trainer.recon_losses = checkpoint.get('recon_losses', [])
            trainer.kl_losses = checkpoint.get('kl_losses', [])
            trainer.learning_rates = checkpoint.get('learning_rates', [])
            trainer.best_loss = checkpoint.get('best_loss', float('inf'))
            
            print(f"Loaded model with best loss: {trainer.best_loss:.4f}")
            
            # Show results from existing model
            trainer.show_enhanced_results()
            if trainer.losses:  # Only plot if we have training history
                trainer.plot_enhanced_training_history()
            return trainer
    
    # Run enhanced training
    print(f"\n‍♂️ Starting enhanced training...")
    final_epoch, best_loss = trainer.enhanced_train(train_loader, test_loader, epochs=200)
    
    # Generate and show results
    print(f"\n Generating enhanced results...")
    trainer.show_enhanced_results()
    trainer.plot_enhanced_training_history()
    
    print(f"\nEnhanced VAE training complete")
    print(f"Best loss achieved: {best_loss:.4f} in {final_epoch} epochs")
    print(" Check results/ directory for enhanced visualizations")
    
    return trainer


def main():
    """Main generator pipeline with enhanced training option."""
    print("MNIST VAE Generator")
    print("Choose your training mode:")
    print("=" * 40)
    print("1. Quick training (15 epochs, ~2 minutes)")
    print("2. Enhanced training (up to 200 epochs with early stopping)")
    print()
    
    choice = input("Enter your choice (1 for quick, 2 for enhanced, or Enter for enhanced): ").strip()
    
    if choice == "1":
        # Quick training mode
        print("\n Quick MNIST Generator - Fast Results!")
        print("=" * 40)
        
        # Load data
        mnist = MNIST(batch_size=128)
        train_loader = mnist.train_loader
        
        print(f"Dataset loaded: {mnist.info()}")
        
        # Create and train generator
        generator = QuickGenerator(latent_dim=20)
        generator.train(train_loader, epochs=15)
        
        # Generate and show results
        print("\n Generating new digits...")
        generator.show_generated_images()
        generator.plot_training_history()
        
        print("\nQuick generation complete!")
        print("Check 'results/quick_generated_digits.png' for results")
    
    else:
        # Enhanced training mode (default)
        run_enhanced_training()


if __name__ == "__main__":
    main()
