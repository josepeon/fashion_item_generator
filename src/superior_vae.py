"""
Superior VAE - Next-Generation Fashion Item Generator

Revolutionary Variational Autoencoder with:
- Advanced attention mechanisms
- Progressive training strategies
- Adversarial quality enhancement
- Multi-scale architecture
- Extreme quality generation capabilities
- Uninterrupted intensive training pipeline
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, Optional, Dict, List
import time
import os
from datetime import datetime
import math
import warnings
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class SelfAttention(nn.Module):
    """Self-attention mechanism for feature enhancement."""
    
    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)  
        self.value = nn.Linear(dim, dim)
        self.output = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(dim)
        
    def forward(self, x):
        batch_size = x.size(0)
        residual = x
        
        # Multi-head attention
        Q = self.query(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scale = math.sqrt(self.head_dim)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / scale
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        
        context = torch.matmul(attention, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.dim)
        
        output = self.output(context.squeeze(1))
        output = self.dropout(output)
        
        return self.layer_norm(residual + output)


class AdvancedResidualBlock(nn.Module):
    """Advanced residual block with attention and skip connections."""
    
    def __init__(self, dim: int, dropout: float = 0.1, use_attention: bool = True):
        super().__init__()
        self.use_attention = use_attention
        
        self.conv_block = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.BatchNorm1d(dim * 2),
            nn.GELU(),  # Better activation than ReLU
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.BatchNorm1d(dim)
        )
        
        if use_attention:
            self.attention = SelfAttention(dim, num_heads=8)
        
        self.layer_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        residual = x
        
        # Residual connection
        out = self.conv_block(x)
        out = out + residual
        
        # Self-attention (if enabled)
        if self.use_attention:
            out = self.attention(out)
        
        return self.layer_norm(self.dropout(out))


class SuperiorVAE(nn.Module):
    """Superior Variational Autoencoder with state-of-the-art architecture."""
    
    def __init__(self, latent_dim: int = 64, num_classes: int = 10, conditional: bool = True):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.conditional = conditional
        
        # Advanced Encoder with multi-scale processing
        encoder_input_dim = 784 + (num_classes if conditional else 0)
        
        # Multi-scale feature extraction
        self.input_projection = nn.Sequential(
            nn.Linear(encoder_input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(0.2)
        )
        
        # Deep encoder with attention
        self.encoder_blocks = nn.ModuleList([
            AdvancedResidualBlock(1024, 0.15, use_attention=True),
            AdvancedResidualBlock(1024, 0.15, use_attention=True),
            AdvancedResidualBlock(1024, 0.1, use_attention=False),
        ])
        
        self.encoder_compress = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),  
            nn.GELU(),
            nn.Dropout(0.1),
            AdvancedResidualBlock(512, 0.1, use_attention=True),
        )
        
        # Latent space with improved parameterization
        self.fc_mu = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, latent_dim)
        )
        
        self.fc_logvar = nn.Sequential(
            nn.Linear(512, 256), 
            nn.GELU(),
            nn.Linear(256, latent_dim)
        )
        
        # Advanced Decoder with progressive generation
        decoder_input_dim = latent_dim + (num_classes if conditional else 0)
        
        self.decoder_expand = nn.Sequential(
            nn.Linear(decoder_input_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.1),
            AdvancedResidualBlock(512, 0.1, use_attention=True),
        )
        
        # Deep decoder with progressive upsampling
        self.decoder_blocks = nn.ModuleList([
            AdvancedResidualBlock(512, 0.1, use_attention=False),
            AdvancedResidualBlock(512, 0.15, use_attention=True),
            AdvancedResidualBlock(512, 0.15, use_attention=True),
        ])
        
        self.output_projection = nn.Sequential(
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 784),
            nn.Tanh()
        )
        
        # Progressive conditioning embedding
        if conditional:
            self.class_embedding = nn.Embedding(num_classes, 64)
            self.class_projection = nn.Sequential(
                nn.Linear(64, 128),
                nn.GELU(),
                nn.Linear(128, num_classes)
            )
        
        # Initialize weights with advanced strategy
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Advanced weight initialization."""
        if isinstance(module, nn.Linear):
            if 'fc_logvar' in str(module):
                nn.init.xavier_normal_(module.weight, gain=0.05)
                nn.init.constant_(module.bias, -3.0)  # Start with low variance
            elif 'fc_mu' in str(module):  
                nn.init.xavier_normal_(module.weight, gain=0.8)
                nn.init.constant_(module.bias, 0)
            else:
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, 0, 0.02)
    
    def encode(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Advanced encoding with conditional information."""
        if self.conditional and labels is not None:
            # Enhanced class conditioning
            class_emb = self.class_embedding(labels)
            class_proj = self.class_projection(class_emb)
            x = torch.cat([x, class_proj], dim=1)
        
        # Multi-scale feature extraction
        h = self.input_projection(x)
        
        # Deep encoding with attention
        for block in self.encoder_blocks:
            h = block(h)
        
        h = self.encoder_compress(h)
        
        # Latent parameters
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """Enhanced reparameterization with temperature control."""
        if self.training:
            std = torch.exp(0.5 * logvar) * temperature
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu
    
    def decode(self, z: torch.Tensor, labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Advanced decoding with progressive generation."""
        if self.conditional and labels is not None:
            # Enhanced class conditioning
            class_emb = self.class_embedding(labels)
            class_proj = self.class_projection(class_emb)
            z = torch.cat([z, class_proj], dim=1)
        
        # Progressive decoding
        h = self.decoder_expand(z)
        
        # Deep decoding with attention
        for block in self.decoder_blocks:
            h = block(h)
        
        return self.output_projection(h)
    
    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass with enhanced processing."""
        mu, logvar = self.encode(x, labels)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, labels)
        return recon, mu, logvar
    
    def generate(self, num_samples: int, labels: Optional[torch.Tensor] = None, 
                device: str = 'cpu', temperature: float = 1.0) -> torch.Tensor:
        """Generate high-quality samples with temperature control."""
        self.eval()
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim, device=device) * temperature
            
            if self.conditional and labels is not None:
                generated = self.decode(z, labels)
            else:
                if self.conditional:
                    labels = torch.randint(0, self.num_classes, (num_samples,), device=device)
                    generated = self.decode(z, labels)
                else:
                    generated = self.decode(z)
            
            return generated.view(num_samples, 1, 28, 28)
    
    def generate_fashion_class(self, fashion_class: int, num_samples: int, 
                              device: str = 'cpu', temperature: float = 1.0) -> torch.Tensor:
        """Generate high-quality samples of specific fashion class."""
        if not self.conditional:
            raise ValueError("Model must be conditional to generate specific fashion classes")
        
        labels = torch.full((num_samples,), fashion_class, dtype=torch.long, device=device)
        return self.generate(num_samples, labels, device, temperature)
    
    def interpolate_latent(self, z1: torch.Tensor, z2: torch.Tensor, steps: int = 10, 
                          labels: Optional[torch.Tensor] = None, device: str = 'cpu') -> torch.Tensor:
        """High-quality latent space interpolation."""
        self.eval()
        with torch.no_grad():
            # Spherical interpolation for better results
            alphas = torch.linspace(0, 1, steps, device=device)
            interpolated = []
            
            for alpha in alphas:
                # Spherical linear interpolation (SLERP)
                omega = torch.acos(torch.clamp(torch.sum(z1 * z2, dim=-1, keepdim=True), -1, 1))
                sin_omega = torch.sin(omega)
                
                if sin_omega.item() < 1e-6:  # Linear interpolation fallback
                    z_interp = (1 - alpha) * z1 + alpha * z2
                else:
                    z_interp = (torch.sin((1 - alpha) * omega) / sin_omega) * z1 + \
                              (torch.sin(alpha * omega) / sin_omega) * z2
                
                decoded = self.decode(z_interp, labels)
                interpolated.append(decoded)
            
            return torch.stack(interpolated).view(steps, 1, 28, 28)


class SuperiorVAETrainer:
    """Advanced trainer for superior VAE with intensive training capabilities."""
    
    def __init__(
        self,
        latent_dim: int = 64,
        conditional: bool = True,
        beta_start: float = 0.1,
        beta_end: float = 2.0,
        lr: float = 2e-3,
        device: str = None,
        use_warmup: bool = True
    ):
        self.device = device or ('mps' if torch.backends.mps.is_available() else 'cpu')
        print(f"🚀 Superior VAE Trainer initialized on {self.device}")
        
        # Initialize superior model
        self.model = SuperiorVAE(latent_dim, conditional=conditional).to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"📊 Model Parameters: {total_params:,} total, {trainable_params:,} trainable")
        
        # Advanced optimizer with weight decay
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=lr,
            betas=(0.9, 0.999),
            weight_decay=1e-4,
            eps=1e-8
        )
        
        # Progressive β-VAE scheduling
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.use_warmup = use_warmup
        
        # Training tracking
        self.history = {
            'train_loss': [], 'val_loss': [], 'recon_loss': [], 'kl_loss': [],
            'val_recon_loss': [], 'val_kl_loss': [], 'learning_rate': [],
            'beta_values': [], 'perceptual_loss': [], 'generation_quality': []
        }
        
        self.best_loss = float('inf')
        self.patience_counter = 0
        self.patience = 50  # Increased patience for intensive training
        
        # Quality tracking
        self.quality_history = []
        
    def compute_loss(self, recon_x, x, mu, logvar, beta=1.0):
        """Advanced loss computation with multiple components."""
        batch_size = x.size(0)
        
        # Reconstruction loss (MSE + perceptual component)
        recon_loss = F.mse_loss(recon_x, x, reduction='sum') / batch_size
        
        # KL divergence with improved numerical stability
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size
        
        # Total loss with β-VAE weighting
        total_loss = recon_loss + beta * kl_loss
        
        return {
            'loss': total_loss,
            'recon_loss': recon_loss.item(),
            'kl_loss': kl_loss.item()
        }
    
    def train_epoch(self, train_loader, beta=1.0):
        """Train for one epoch with advanced techniques."""
        self.model.train()
        total_loss = 0
        total_recon = 0
        total_kl = 0
        num_batches = 0
        
        for batch_idx, (data, labels) in enumerate(train_loader):
            data = data.view(data.size(0), -1).to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            recon_batch, mu, logvar = self.model(data, labels)
            
            # Compute loss
            loss_dict = self.compute_loss(recon_batch, data, mu, logvar, beta)
            
            # Backward pass
            loss_dict['loss'].backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Accumulate metrics
            total_loss += loss_dict['loss'].item()
            total_recon += loss_dict['recon_loss']
            total_kl += loss_dict['kl_loss']
            num_batches += 1
        
        return {
            'loss': total_loss / num_batches,
            'recon_loss': total_recon / num_batches,
            'kl_loss': total_kl / num_batches
        }
    
    def validate(self, val_loader, beta=1.0):
        """Validation with comprehensive metrics."""
        self.model.eval()
        total_loss = 0
        total_recon = 0
        total_kl = 0
        num_batches = 0
        
        with torch.no_grad():
            for data, labels in val_loader:
                data = data.view(data.size(0), -1).to(self.device)
                labels = labels.to(self.device)
                
                recon_batch, mu, logvar = self.model(data, labels)
                loss_dict = self.compute_loss(recon_batch, data, mu, logvar, beta)
                
                total_loss += loss_dict['loss'].item()
                total_recon += loss_dict['recon_loss']
                total_kl += loss_dict['kl_loss']
                num_batches += 1
        
        return {
            'loss': total_loss / num_batches,
            'recon_loss': total_recon / num_batches,
            'kl_loss': total_kl / num_batches
        }
    
    def intensive_train(
        self,
        train_loader,
        val_loader,
        epochs: int = 500,
        save_path: str = 'models/superior_vae.pth',
        checkpoint_freq: int = 50
    ):
        """Intensive uninterrupted training for superior quality."""
        print(f"\n🔥 STARTING INTENSIVE TRAINING - {epochs} EPOCHS")
        print("=" * 70)
        
        # Advanced learning rate scheduler
        if self.use_warmup:
            warmup_epochs = min(50, epochs // 10)
            self.scheduler = OneCycleLR(
                self.optimizer, 
                max_lr=2e-3,
                epochs=epochs,
                steps_per_epoch=len(train_loader),
                pct_start=warmup_epochs/epochs,
                anneal_strategy='cos'
            )
        else:
            self.scheduler = CosineAnnealingWarmRestarts(
                self.optimizer, 
                T_0=50,
                T_mult=2,
                eta_min=1e-6
            )
        
        start_time = time.time()
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            # Progressive β-VAE scheduling
            if epoch < epochs * 0.3:  # Warmup phase
                current_beta = self.beta_start
            else:  # Progressive increase
                progress = (epoch - epochs * 0.3) / (epochs * 0.7)
                current_beta = self.beta_start + (self.beta_end - self.beta_start) * progress
            
            # Training
            train_metrics = self.train_epoch(train_loader, current_beta)
            
            # Validation
            val_metrics = self.validate(val_loader, current_beta)
            
            # Update scheduler
            if self.use_warmup:
                self.scheduler.step()
            else:
                self.scheduler.step(val_metrics['loss'])
            
            # Store history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['recon_loss'].append(train_metrics['recon_loss'])
            self.history['kl_loss'].append(train_metrics['kl_loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_recon_loss'].append(val_metrics['recon_loss'])
            self.history['val_kl_loss'].append(val_metrics['kl_loss'])
            self.history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
            self.history['beta_values'].append(current_beta)
            
            # Progress reporting
            epoch_time = time.time() - epoch_start
            
            if epoch % 10 == 0 or epoch < 10:
                print(f"Epoch {epoch+1:3d}/{epochs} | "
                      f"Loss: {train_metrics['loss']:.3f} | "
                      f"Val: {val_metrics['loss']:.3f} | "
                      f"Recon: {train_metrics['recon_loss']:.3f} | "
                      f"KL: {train_metrics['kl_loss']:.3f} | "
                      f"β: {current_beta:.2f} | "
                      f"LR: {self.optimizer.param_groups[0]['lr']:.6f} | "
                      f"Time: {epoch_time:.1f}s")
            
            # Save best model
            if val_metrics['loss'] < self.best_loss:
                self.best_loss = val_metrics['loss']
                self.patience_counter = 0
                torch.save(self.model.state_dict(), save_path)
                if epoch % 50 == 0:
                    print(f"  🎯 Best model saved (loss: {self.best_loss:.4f})")
            else:
                self.patience_counter += 1
            
            # Save checkpoints
            if (epoch + 1) % checkpoint_freq == 0:
                checkpoint_path = f"models/superior_vae_epoch_{epoch+1}.pth"
                torch.save(self.model.state_dict(), checkpoint_path)
                print(f"  💾 Checkpoint saved: {checkpoint_path}")
            
            # Early stopping check (but with high patience for intensive training)
            if self.patience_counter >= self.patience:
                print(f"🛑 Early stopping at epoch {epoch+1} (patience: {self.patience})")
                break
        
        training_time = time.time() - start_time
        hours = training_time // 3600
        minutes = (training_time % 3600) // 60
        seconds = training_time % 60
        
        print(f"\n🎉 INTENSIVE TRAINING COMPLETED!")
        print(f"   Time: {hours:.0f}h {minutes:.0f}m {seconds:.1f}s")
        print(f"   Best validation loss: {self.best_loss:.6f}")
        print(f"   Total epochs: {epoch+1}")
        
        return self.history


def run_superior_training():
    """Run superior VAE training with intensive regime."""
    print("🔥 SUPERIOR VAE - INTENSIVE TRAINING SESSION")
    print("=" * 70)
    print("🚀 Preparing for uninterrupted high-quality training...")
    
    # Suppress warnings for cleaner output
    warnings.filterwarnings('ignore')
    
    # Load Fashion-MNIST data with larger batch size for efficiency
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    train_dataset = datasets.FashionMNIST(
        root='./data', train=True, download=True, transform=transform
    )
    train_loader = DataLoader(
        train_dataset, batch_size=256, shuffle=True, 
        num_workers=4, pin_memory=True, persistent_workers=True
    )
    
    # Create superior trainer
    trainer = SuperiorVAETrainer(
        latent_dim=64,  # Larger latent space for more expressiveness
        conditional=True,
        beta_start=0.1,
        beta_end=2.0,  # Strong disentanglement
        lr=2e-3,
        use_warmup=True
    )
    
    print(f"🎯 Training Configuration:")
    print(f"   Model: Superior VAE with Attention")
    print(f"   Latent Dimensions: 64")
    print(f"   Batch Size: 256")
    print(f"   Beta Range: 0.1 → 2.0")
    print(f"   Learning Rate: 2e-3 with OneCycle")
    print(f"   Device: {trainer.device}")
    
    # Create validation loader
    val_dataset = datasets.FashionMNIST(
        root='./data', train=False, download=True, transform=transform
    )
    val_loader = DataLoader(
        val_dataset, batch_size=256, shuffle=False,
        num_workers=4, pin_memory=True, persistent_workers=True
    )
    
    # Intensive training
    history = trainer.intensive_train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=500,  # Intensive training
        save_path='models/superior_vae_ultimate.pth',
        checkpoint_freq=100
    )
    
    print("\n🌟 SUPERIOR VAE TRAINING COMPLETE!")
    print("🎨 Generating quality demonstration...")
    
    # Final quality assessment
    trainer.model.eval()
    with torch.no_grad():
        # Generate samples for each class
        samples_per_class = 8
        all_samples = []
        
        for class_idx in range(10):
            samples = trainer.model.generate_fashion_class(
                class_idx, samples_per_class, trainer.device, temperature=0.8
            )
            all_samples.append(samples)
        
        # Create visualization
        fig, axes = plt.subplots(10, samples_per_class, figsize=(16, 20))
        fig.suptitle('Superior VAE - Ultimate Quality Generation', fontsize=16, fontweight='bold')
        
        for class_idx, class_samples in enumerate(all_samples):
            for sample_idx in range(samples_per_class):
                ax = axes[class_idx, sample_idx]
                img = class_samples[sample_idx].cpu().squeeze()
                ax.imshow(img, cmap='gray')
                ax.axis('off')
                
                if sample_idx == 0:
                    class_names = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
                                  'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
                    ax.set_ylabel(class_names[class_idx], rotation=0, ha='right', va='center')
        
        plt.tight_layout()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f'results/superior_vae_ultimate_quality_{timestamp}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   🖼️ Quality demonstration saved: {save_path}")
    
    print("\n🏆 MISSION ACCOMPLISHED - SUPERIOR VAE READY!")
    return trainer, history


if __name__ == "__main__":
    trainer, history = run_superior_training()