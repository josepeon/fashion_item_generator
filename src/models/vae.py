"""Optimized VAE for Fashion-MNIST generation."""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """Simple residual block."""

    def __init__(self, dim: int, hidden_dim: int = None, dropout: float = 0.1):
        super().__init__()
        hidden_dim = hidden_dim or dim * 2
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(x + self.net(x))


class FashionVAE(nn.Module):
    """Efficient conditional VAE for Fashion-MNIST (~3M params)."""

    def __init__(self, latent_dim: int = 32, num_classes: int = 10, conditional: bool = True):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.conditional = conditional

        input_dim = 784 + (num_classes if conditional else 0)

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            ResBlock(512, dropout=0.1),
            nn.Linear(512, 256),
            nn.GELU(),
            ResBlock(256, dropout=0.1),
        )
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

        # Decoder
        decoder_input = latent_dim + (num_classes if conditional else 0)
        self.decoder = nn.Sequential(
            nn.Linear(decoder_input, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            ResBlock(256, dropout=0.1),
            nn.Linear(256, 512),
            nn.GELU(),
            ResBlock(512, dropout=0.1),
            nn.Linear(512, 784),
            nn.Tanh(),
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def encode(self, x, labels=None):
        if self.conditional and labels is not None:
            onehot = F.one_hot(labels, self.num_classes).float()
            x = torch.cat([x, onehot], dim=1)
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            return mu + std * torch.randn_like(std)
        return mu

    def decode(self, z, labels=None):
        if self.conditional and labels is not None:
            onehot = F.one_hot(labels, self.num_classes).float()
            z = torch.cat([z, onehot], dim=1)
        return self.decoder(z)

    def forward(self, x, labels=None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x, labels)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, labels), mu, logvar

    @torch.no_grad()
    def generate(self, num_samples: int, labels=None, device="cpu", temperature=1.0):
        self.eval()
        z = torch.randn(num_samples, self.latent_dim, device=device) * temperature
        if self.conditional:
            if labels is None:
                labels = torch.randint(0, self.num_classes, (num_samples,), device=device)
            return self.decode(z, labels).view(num_samples, 1, 28, 28)
        return self.decode(z).view(num_samples, 1, 28, 28)

    @torch.no_grad()
    def generate_class(self, class_idx: int, num_samples: int, device="cpu", temperature=1.0):
        labels = torch.full((num_samples,), class_idx, dtype=torch.long, device=device)
        return self.generate(num_samples, labels, device, temperature)


# Alias
SuperiorVAE = FashionVAE
