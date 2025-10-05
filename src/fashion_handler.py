"""
Fashion-MNIST Dataset Handler - Optimized and Concise

Clean, minimal implementation for Fashion-MNIST dataset operations.
"""

import torch
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from typing import Tuple


class FashionMNIST:
    """Simple, efficient Fashion-MNIST dataset handler."""
    
    # Fashion-MNIST class names
    CLASS_NAMES = [
        'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
    ]
    
    def __init__(self, data_dir: str = './data', batch_size: int = 64):
        self.data_dir = data_dir
        self.batch_size = batch_size
        
        # Standard Fashion-MNIST preprocessing
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        self._train_dataset = None
        self._test_dataset = None
        self._train_loader = None
        self._test_loader = None
    
    @property
    def train_dataset(self) -> datasets.FashionMNIST:
        """Lazy load training dataset."""
        if self._train_dataset is None:
            self._train_dataset = datasets.FashionMNIST(
                root=self.data_dir, train=True, download=True, transform=self.transform
            )
        return self._train_dataset
    
    @property
    def test_dataset(self) -> datasets.FashionMNIST:
        """Lazy load test dataset."""
        if self._test_dataset is None:
            self._test_dataset = datasets.FashionMNIST(
                root=self.data_dir, train=False, download=True, transform=self.transform
            )
        return self._test_dataset
    
    def get_train_loader(self):
        """Get training data loader with optimized settings."""
        if self._train_loader is None:
            # Optimize workers and memory based on system
            has_gpu = torch.cuda.is_available() or (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available())
            num_workers = 4 if has_gpu else 2  # Reduce workers on CPU-only systems
            
            self._train_loader = DataLoader(
                self.train_dataset, 
                batch_size=self.batch_size, 
                shuffle=True, 
                num_workers=num_workers,
                pin_memory=has_gpu,  # Only pin memory if we have GPU
                persistent_workers=num_workers > 0
            )
        return self._train_loader
    
    def get_test_loader(self):
        """Get test data loader with optimized settings."""
        if self._test_loader is None:
            # Optimize workers and memory based on system
            has_gpu = torch.cuda.is_available() or (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available())
            num_workers = 4 if has_gpu else 2  # Reduce workers on CPU-only systems
            
            self._test_loader = DataLoader(
                self.test_dataset, 
                batch_size=self.batch_size, 
                shuffle=False, 
                num_workers=num_workers,
                pin_memory=has_gpu,  # Only pin memory if we have GPU
                persistent_workers=num_workers > 0
            )
        return self._test_loader
    
    def sample_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a sample batch from training data."""
        return next(iter(self.train_loader))
    
    def visualize_samples(self, num_samples: int = 8, save_path: str = None) -> None:
        """Create visualization of sample images."""
        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        fig.suptitle('Fashion-MNIST Sample Items', fontweight='bold')
        
        for i in range(num_samples):
            image, label = self.train_dataset[i]
            row, col = i // 4, i % 4
            axes[row, col].imshow(image.squeeze(), cmap='gray')
            axes[row, col].set_title(f'{self.CLASS_NAMES[label]}')
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
            'class_names': self.CLASS_NAMES,
            'image_shape': self.train_dataset[0][0].shape,
            'batch_size': self.batch_size
        }


def main():
    """Demo the Fashion-MNIST handler."""
    print("PyTorch Fashion-MNIST Handler")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    # Initialize Fashion-MNIST handler
    fashion = FashionMNIST(batch_size=64)
    
    # Show info
    info = fashion.info()
    print(f"\nDataset Info:")
    for key, value in info.items():
        if key == 'class_names':
            print(f"  {key}: {', '.join(value)}")
        else:
            print(f"  {key}: {value}")
    
    # Create visualization
    fashion.visualize_samples(save_path='fashion_samples.png')
    
    # Test data loading
    batch_images, batch_labels = fashion.sample_batch()
    print(f"\nBatch loaded: {batch_images.shape}")
    print(f"Sample labels: {batch_labels[:8].tolist()}")
    print(f"Sample items: {[fashion.CLASS_NAMES[i] for i in batch_labels[:8]]}")
    
    print("\nReady for model training!")


if __name__ == "__main__":
    main()