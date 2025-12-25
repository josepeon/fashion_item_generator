"""Variational Autoencoder for Fashion-MNIST generation."""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    """Multi-head self-attention mechanism."""

    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads

        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.output = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        residual = x

        Q = self.query(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        scale = math.sqrt(self.head_dim)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / scale
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)

        context = torch.matmul(attention, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.dim)

        output = self.output(context.squeeze(1))
        output = self.dropout(output)

        return self.layer_norm(residual + output)


class ResidualBlock(nn.Module):
    """Residual block with optional attention."""

    def __init__(self, dim: int, dropout: float = 0.1, use_attention: bool = True):
        super().__init__()
        self.use_attention = use_attention

        self.conv_block = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.BatchNorm1d(dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.BatchNorm1d(dim),
        )

        if use_attention:
            self.attention = SelfAttention(dim, num_heads=8)

        self.layer_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv_block(x)
        out = out + residual

        if self.use_attention:
            out = self.attention(out)

        return self.layer_norm(self.dropout(out))


class FashionVAE(nn.Module):
    """Conditional VAE with attention for Fashion-MNIST generation."""

    def __init__(self, latent_dim: int = 64, num_classes: int = 10, conditional: bool = True):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.conditional = conditional

        encoder_input_dim = 784 + (num_classes if conditional else 0)

        # Encoder
        self.input_projection = nn.Sequential(
            nn.Linear(encoder_input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(0.2),
        )

        self.encoder_blocks = nn.ModuleList([
            ResidualBlock(1024, 0.15, use_attention=True),
            ResidualBlock(1024, 0.15, use_attention=True),
            ResidualBlock(1024, 0.1, use_attention=False),
        ])

        self.encoder_compress = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.1),
            ResidualBlock(512, 0.1, use_attention=True),
        )

        # Latent space
        self.fc_mu = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, latent_dim),
        )

        self.fc_logvar = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, latent_dim),
        )

        # Decoder
        decoder_input_dim = latent_dim + (num_classes if conditional else 0)

        self.decoder_expand = nn.Sequential(
            nn.Linear(decoder_input_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.1),
            ResidualBlock(512, 0.1, use_attention=True),
        )

        self.decoder_blocks = nn.ModuleList([
            ResidualBlock(512, 0.1, use_attention=False),
            ResidualBlock(512, 0.15, use_attention=True),
            ResidualBlock(512, 0.15, use_attention=True),
        ])

        self.output_projection = nn.Sequential(
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 784),
            nn.Tanh(),
        )

        # Class conditioning
        if conditional:
            self.class_embedding = nn.Embedding(num_classes, 64)
            self.class_projection = nn.Sequential(
                nn.Linear(64, 128),
                nn.GELU(),
                nn.Linear(128, num_classes),
            )

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, 0, 0.02)

    def encode(
        self, x: torch.Tensor, labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.conditional and labels is not None:
            class_emb = self.class_embedding(labels)
            class_proj = self.class_projection(class_emb)
            x = torch.cat([x, class_proj], dim=1)

        h = self.input_projection(x)
        for block in self.encoder_blocks:
            h = block(h)
        h = self.encoder_compress(h)

        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(
        self, mu: torch.Tensor, logvar: torch.Tensor, temperature: float = 1.0
    ) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar) * temperature
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def decode(self, z: torch.Tensor, labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.conditional and labels is not None:
            class_emb = self.class_embedding(labels)
            class_proj = self.class_projection(class_emb)
            z = torch.cat([z, class_proj], dim=1)

        h = self.decoder_expand(z)
        for block in self.decoder_blocks:
            h = block(h)

        return self.output_projection(h)

    def forward(
        self, x: torch.Tensor, labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x, labels)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, labels)
        return recon, mu, logvar

    def generate(
        self,
        num_samples: int,
        labels: Optional[torch.Tensor] = None,
        device: str = "cpu",
        temperature: float = 1.0,
    ) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim, device=device) * temperature

            if self.conditional:
                if labels is None:
                    labels = torch.randint(0, self.num_classes, (num_samples,), device=device)
                generated = self.decode(z, labels)
            else:
                generated = self.decode(z)

            return generated.view(num_samples, 1, 28, 28)

    def generate_class(
        self, class_idx: int, num_samples: int, device: str = "cpu", temperature: float = 1.0
    ) -> torch.Tensor:
        if not self.conditional:
            raise ValueError("Model must be conditional to generate specific classes")
        labels = torch.full((num_samples,), class_idx, dtype=torch.long, device=device)
        return self.generate(num_samples, labels, device, temperature)


# Alias for backward compatibility
SuperiorVAE = FashionVAE
