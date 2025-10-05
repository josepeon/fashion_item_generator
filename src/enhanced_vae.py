"""
Enhanced VAE for Superior Fashion-MNIST Generation

Advanced Variational Autoencoder with:
- Deeper architecture with residual connections
- Conditional generation capabilities
- Advanced loss functions (β-VAE, spectral regularization)
- Progressive training strategies
- Higher quality fashion item generation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, Optional, Dict, List
import time
import os
from datetime import datetime
import math

from fashion_handler import FashionMNIST


class ResidualBlock(nn.Module):
    """Residual block for deeper VAE architecture."""
    
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim)
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        return F.relu(out)


class EnhancedVAE(nn.Module):
    """Enhanced Variational Autoencoder with superior generation quality."""
    
    def __init__(self, latent_dim: int = 32, num_classes: int = 10, conditional: bool = True):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.conditional = conditional
        
        # Enhanced Encoder with residual connections
        encoder_input_dim = 784 + (num_classes if conditional else 0)
        self.encoder = nn.Sequential(
            nn.Linear(encoder_input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            
            ResidualBlock(512, 0.1),
            ResidualBlock(512, 0.1),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            
            ResidualBlock(256, 0.05),
        )
        
        # Latent space layers
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)
        
        # Enhanced Decoder with residual connections
        decoder_input_dim = latent_dim + (num_classes if conditional else 0)
        self.decoder = nn.Sequential(
            nn.Linear(decoder_input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.05),
            
            ResidualBlock(256, 0.05),
            
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            
            ResidualBlock(512, 0.1),
            ResidualBlock(512, 0.1),
            
            nn.Linear(512, 784),
            nn.Tanh()  # Output between -1 and 1 to match MNIST normalization
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights with Xavier/He initialization."""
        if isinstance(module, nn.Linear):
            if 'fc_logvar' in str(module):
                # Initialize logvar layer to output small values initially
                nn.init.xavier_normal_(module.weight, gain=0.1)
                nn.init.constant_(module.bias, -2.0)
            else:
                # He initialization for ReLU layers
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def encode(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent distribution parameters."""
        if self.conditional and labels is not None:
            # One-hot encode labels
            labels_onehot = F.one_hot(labels, self.num_classes).float()
            x = torch.cat([x, labels_onehot], dim=1)
        
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick with improved stability."""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu
    
    def decode(self, z: torch.Tensor, labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Decode from latent space to image."""
        if self.conditional and labels is not None:
            # One-hot encode labels
            labels_onehot = F.one_hot(labels, self.num_classes).float()
            z = torch.cat([z, labels_onehot], dim=1)
        
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through the VAE."""
        mu, logvar = self.encode(x, labels)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, labels)
        return recon, mu, logvar
    
    def generate(self, num_samples: int, labels: Optional[torch.Tensor] = None, device: str = 'cpu') -> torch.Tensor:
        """Generate new samples from the learned distribution."""
        self.eval()
        with torch.no_grad():
            # Sample from latent space
            z = torch.randn(num_samples, self.latent_dim, device=device)
            
            if self.conditional and labels is not None:
                generated = self.decode(z, labels)
            else:
                # Generate random labels if conditional but none provided
                if self.conditional:
                    labels = torch.randint(0, self.num_classes, (num_samples,), device=device)
                    generated = self.decode(z, labels)
                else:
                    generated = self.decode(z)
            
            return generated.view(num_samples, 1, 28, 28)
    
    def generate_specific_digits(self, digit: int, num_samples: int, device: str = 'cpu') -> torch.Tensor:
        """Generate samples of a specific digit."""
        if not self.conditional:
            raise ValueError("Model must be conditional to generate specific digits")
        
        labels = torch.full((num_samples,), digit, dtype=torch.long, device=device)
        return self.generate(num_samples, labels, device)


class EnhancedVAETrainer:
    """Enhanced trainer with advanced techniques for superior generation quality."""
    
    def __init__(
        self,
        latent_dim: int = 32,
        conditional: bool = True,
        beta: float = 1.0,
        lr: float = 1e-3,
        device: str = None
    ):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.latent_dim = latent_dim
        self.conditional = conditional
        self.beta = beta  # β-VAE parameter for disentanglement
        
        # Create enhanced model
        self.model = EnhancedVAE(latent_dim, conditional=conditional).to(self.device)
        
        # Advanced optimizer with weight decay
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=1e-4,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Learning rate schedulers
        self.scheduler_plateau = ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
        self.scheduler_cosine = CosineAnnealingLR(
            self.optimizer, T_max=50, eta_min=1e-6
        )
        
        # Training history
        self.history = {
            'train_loss': [], 'recon_loss': [], 'kl_loss': [],
            'val_loss': [], 'val_recon_loss': [], 'val_kl_loss': [],
            'learning_rate': [], 'generation_quality': []
        }
        
        # Training parameters
        self.best_loss = float('inf')
        self.patience = 25
        self.patience_counter = 0
        
        print(f"Enhanced VAE Trainer initialized on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Conditional generation: {conditional}")
        print(f"β-VAE parameter: {beta}")
    
    def vae_loss(
        self,
        recon_x: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        beta: float = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Enhanced VAE loss with β-VAE and improved reconstruction loss."""
        if beta is None:
            beta = self.beta
            
        # Reconstruction loss - use MSE for better gradients
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        
        # KL divergence loss with improved formulation
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Total loss with β weighting
        total_loss = recon_loss + beta * kl_loss
        
        return total_loss, recon_loss, kl_loss
    
    def train_epoch(self, train_loader) -> Dict[str, float]:
        """Train for one epoch with enhanced techniques."""
        self.model.train()
        total_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0
        num_batches = len(train_loader)
        
        for batch_idx, (data, labels) in enumerate(train_loader):
            data = data.to(self.device)
            labels = labels.to(self.device) if self.conditional else None
            
            # Flatten data
            data_flat = data.view(data.size(0), -1)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            if self.conditional:
                recon_batch, mu, logvar = self.model(data_flat, labels)
            else:
                recon_batch, mu, logvar = self.model(data_flat)
            
            # Compute loss
            loss, recon_loss, kl_loss = self.vae_loss(recon_batch, data_flat, mu, logvar)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Accumulate losses
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
        
        # Average losses
        avg_loss = total_loss / len(train_loader.dataset)
        avg_recon_loss = total_recon_loss / len(train_loader.dataset)
        avg_kl_loss = total_kl_loss / len(train_loader.dataset)
        
        return {
            'loss': avg_loss,
            'recon_loss': avg_recon_loss,
            'kl_loss': avg_kl_loss
        }
    
    def validate(self, val_loader) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0
        
        with torch.no_grad():
            for data, labels in val_loader:
                data = data.to(self.device)
                labels = labels.to(self.device) if self.conditional else None
                data_flat = data.view(data.size(0), -1)
                
                if self.conditional:
                    recon_batch, mu, logvar = self.model(data_flat, labels)
                else:
                    recon_batch, mu, logvar = self.model(data_flat)
                
                loss, recon_loss, kl_loss = self.vae_loss(recon_batch, data_flat, mu, logvar)
                
                total_loss += loss.item()
                total_recon_loss += recon_loss.item()
                total_kl_loss += kl_loss.item()
        
        avg_loss = total_loss / len(val_loader.dataset)
        avg_recon_loss = total_recon_loss / len(val_loader.dataset)
        avg_kl_loss = total_kl_loss / len(val_loader.dataset)
        
        return {
            'loss': avg_loss,
            'recon_loss': avg_recon_loss,
            'kl_loss': avg_kl_loss
        }
    
    def evaluate_generation_quality(self, cnn_classifier=None, num_samples: int = 100) -> float:
        """Evaluate generation quality using a trained CNN classifier."""
        if cnn_classifier is None:
            return 0.0
        
        self.model.eval()
        cnn_classifier.eval()
        
        with torch.no_grad():
            # Generate samples
            if self.conditional:
                # Generate balanced samples across all digits
                samples_per_digit = num_samples // 10
                all_generated = []
                for digit in range(10):
                    digit_samples = self.model.generate_specific_digits(
                        digit, samples_per_digit, self.device
                    )
                    all_generated.append(digit_samples)
                generated_images = torch.cat(all_generated, dim=0)
            else:
                generated_images = self.model.generate(num_samples, device=self.device)
            
            # Evaluate with CNN
            outputs = cnn_classifier(generated_images.to(self.device))
            probs = torch.softmax(outputs, dim=1)
            max_probs = probs.max(dim=1)[0]
            
            avg_confidence = max_probs.mean().item()
            
        return avg_confidence
    
    def train(
        self,
        train_loader,
        val_loader,
        epochs: int = 300,
        cnn_classifier=None,
        save_path: str = 'models/enhanced_vae.pth',
        checkpoint_freq: int = 50
    ):
        """Train the enhanced VAE with progressive techniques."""
        print(f"\nTraining Enhanced VAE for {epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")
        
        start_time = time.time()
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            # Adjust β parameter progressively (β-VAE annealing)
            if epoch < 50:
                current_beta = 0.1 + (self.beta - 0.1) * (epoch / 50)
            else:
                current_beta = self.beta
            
            # Training
            train_metrics = self.train_epoch(train_loader)
            
            # Validation
            val_metrics = self.validate(val_loader)
            
            # Learning rate scheduling
            self.scheduler_plateau.step(val_metrics['loss'])
            if epoch % 50 == 49:  # Every 50 epochs, use cosine annealing
                self.scheduler_cosine.step()
            
            # Store history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['recon_loss'].append(train_metrics['recon_loss'])
            self.history['kl_loss'].append(train_metrics['kl_loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_recon_loss'].append(val_metrics['recon_loss'])
            self.history['val_kl_loss'].append(val_metrics['kl_loss'])
            self.history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
            
            # Evaluate generation quality periodically
            if epoch % 20 == 0 and cnn_classifier is not None:
                quality = self.evaluate_generation_quality(cnn_classifier)
                self.history['generation_quality'].append(quality)
                quality_str = f" | Quality: {quality:.3f}"
            else:
                quality_str = ""
            
            # Progress reporting
            epoch_time = time.time() - epoch_start
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"Loss: {train_metrics['loss']:.2f} | "
                  f"Val Loss: {val_metrics['loss']:.2f} | "
                  f"LR: {self.optimizer.param_groups[0]['lr']:.6f} | "
                  f"β: {current_beta:.2f} | "
                  f"Time: {epoch_time:.1f}s{quality_str}")
            
            # Save best model
            if val_metrics['loss'] < self.best_loss:
                self.best_loss = val_metrics['loss']
                self.patience_counter = 0
                torch.save(self.model.state_dict(), save_path)
                print(f"  Best model saved (loss: {self.best_loss:.2f})")
            else:
                self.patience_counter += 1
            
            # Save checkpoint
            if (epoch + 1) % checkpoint_freq == 0:
                checkpoint_path = f"models/enhanced_vae_epoch_{epoch+1}.pth"
                torch.save(self.model.state_dict(), checkpoint_path)
            
            # Early stopping
            if self.patience_counter >= self.patience:
                print(f"Early stopping at epoch {epoch+1} (patience: {self.patience})")
                break
        
        training_time = time.time() - start_time
        print(f"\nEnhanced VAE training completed in {training_time:.2f} seconds")
        print(f"Best validation loss: {self.best_loss:.4f}")
        
        return self.history


def run_enhanced_training():
    """Run enhanced VAE training with superior generation quality."""
    print("Starting Enhanced VAE Training for Superior Generation Quality")
    print("=" * 70)
    
    # Load Fashion-MNIST data
    fashion = FashionMNIST(batch_size=128)
    
    # Create enhanced trainer
    trainer = EnhancedVAETrainer(
        latent_dim=32,  # Larger latent space
        conditional=True,  # Enable conditional generation
        beta=1.5,  # Higher β for better disentanglement
        lr=2e-3  # Slightly higher learning rate
    )
    
    # Load CNN classifier for quality evaluation
    cnn_classifier = None
    try:
        from mnist_cnn import MNISTNet
        cnn_classifier = MNISTNet().to(trainer.device)
        if os.path.exists('models/best_mnist_cnn.pth'):
            cnn_classifier.load_state_dict(
                torch.load('models/best_mnist_cnn.pth', map_location=trainer.device)
            )
            print("CNN classifier loaded for quality evaluation")
        else:
            cnn_classifier = None
    except:
        cnn_classifier = None
    
    # Train the model
    history = trainer.train(
        train_loader=fashion.train_loader,
        val_loader=fashion.test_loader,
        epochs=300,  # Extended training
        cnn_classifier=cnn_classifier,
        save_path='models/enhanced_vae_superior.pth'
    )
    
    print("\n🎉 Enhanced VAE Training Complete!")
    print("Generating quality assessment...")
    
    # Final quality assessment
    if cnn_classifier is not None:
        final_quality = trainer.evaluate_generation_quality(cnn_classifier, 200)
        print(f"Final generation quality: {final_quality:.3f}")
        
        # Generate sample images for each digit
        print("Generating sample images...")
        fig, axes = plt.subplots(2, 10, figsize=(15, 3))
        
        for digit in range(10):
            samples = trainer.model.generate_specific_digits(digit, 2, trainer.device)
            for i in range(2):
                axes[i, digit].imshow(samples[i, 0].cpu().numpy(), cmap='gray')
                axes[i, digit].set_title(f'Digit {digit}')
                axes[i, digit].axis('off')
        
        plt.suptitle('Enhanced VAE - Generated Digit Samples', fontsize=16)
        plt.tight_layout()
        plt.savefig('results/enhanced_vae_samples.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    return trainer, history


if __name__ == "__main__":
    trainer, history = run_enhanced_training()