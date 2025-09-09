"""
MNIST Dataset Handler - Optimized and Concise

Clean, minimal implementation for MNIST dataset operations.
"""

import torch
import torchvision.transforms as transforms
from torchvision import datasets
import matplotlib.pyplot as plt
from typing import Tuple


class MNIST:
    """Simple, efficient MNIST dataset handler."""
    
    def __init__(self, data_dir: str = './data', batch_size: int = 64):
        self.data_dir = data_dir
        self.batch_size = batch_size
        
        # Standard MNIST preprocessing
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        self._train_dataset = None
        self._test_dataset = None
        self._train_loader = None
        self._test_loader = None
    
    @property
    def train_dataset(self) -> datasets.MNIST:
        """Lazy load training dataset."""
        if self._train_dataset is None:
            self._train_dataset = datasets.MNIST(
                root=self.data_dir, train=True, download=True, transform=self.transform
            )
        return self._train_dataset
    
    @property
    def test_dataset(self) -> datasets.MNIST:
        """Lazy load test dataset."""
        if self._test_dataset is None:
            self._test_dataset = datasets.MNIST(
                root=self.data_dir, train=False, download=True, transform=self.transform
            )
        return self._test_dataset
    
    @property
    def train_loader(self) -> torch.utils.data.DataLoader:
        """Get training data loader."""
        if self._train_loader is None:
            self._train_loader = torch.utils.data.DataLoader(
                self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0
            )
        return self._train_loader
    
    @property
    def test_loader(self) -> torch.utils.data.DataLoader:
        """Get test data loader."""
        if self._test_loader is None:
            self._test_loader = torch.utils.data.DataLoader(
                self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0
            )
        return self._test_loader
    
    def sample_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a sample batch from training data."""
        return next(iter(self.train_loader))
    
    def visualize_samples(self, num_samples: int = 8, save_path: str = None) -> None:
        """Create visualization of sample images."""
        fig, axes = plt.subplots(2, 4, figsize=(10, 5))
        fig.suptitle('MNIST Sample Digits', fontweight='bold')
        
        for i in range(num_samples):
            image, label = self.train_dataset[i]
            row, col = i // 4, i % 4
            axes[row, col].imshow(image.squeeze(), cmap='gray')
            axes[row, col].set_title(f'{label}')
            axes[row, col].axis('off')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved to {save_path}")
        plt.show()
        plt.close()
    
    def info(self) -> dict:
        """Get dataset information."""
        return {
            'train_samples': len(self.train_dataset),
            'test_samples': len(self.test_dataset),
            'classes': len(self.train_dataset.classes),
            'image_shape': self.train_dataset[0][0].shape,
            'batch_size': self.batch_size
        }


def main():
    """Demo the MNIST handler."""
    print("PyTorch MNIST Handler")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    # Initialize MNIST handler
    mnist = MNIST(batch_size=64)
    
    # Show info
    info = mnist.info()
    print(f"\nDataset Info:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Create visualization
    mnist.visualize_samples(save_path='mnist_samples.png')
    
    # Test data loading
    batch_images, batch_labels = mnist.sample_batch()
    print(f"\nBatch loaded: {batch_images.shape}")
    print(f"Sample labels: {batch_labels[:8].tolist()}")
    
    print("\nReady for model training!")


if __name__ == "__main__":
    main()
