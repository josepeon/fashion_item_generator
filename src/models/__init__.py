"""Model architectures for Fashion-MNIST."""

from .cnn import FashionCNN
from .vae import FashionVAE

__all__ = ["FashionCNN", "FashionVAE"]
