"""
MNIST Dataset Handler and Analysis

A clean, production-ready implementation for MNIST dataset loading,
preprocessing, and analysis using PyTorch.
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
import matplotlib.pyplot as plt
import numpy as np
import time
from typing import Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MNISTDataLoader:
    """Handles MNIST dataset loading and preprocessing."""
    
    def __init__(self, data_dir: str = './data', batch_size: int = 64):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        self.train_dataset = None
        self.test_dataset = None
        self.train_loader = None
        self.test_loader = None
    
    def load_datasets(self) -> Tuple[datasets.MNIST, datasets.MNIST]:
        """Load MNIST training and test datasets."""
        logger.info("Loading MNIST datasets...")
        
        self.train_dataset = datasets.MNIST(
            root=self.data_dir,
            train=True,
            download=True,
            transform=self.transform
        )
        
        self.test_dataset = datasets.MNIST(
            root=self.data_dir,
            train=False,
            download=True,
            transform=self.transform
        )
        
        logger.info(f"Loaded {len(self.train_dataset)} training and {len(self.test_dataset)} test samples")
        return self.train_dataset, self.test_dataset
    
    def create_data_loaders(self) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        """Create efficient data loaders for training and testing."""
        if self.train_dataset is None or self.test_dataset is None:
            raise ValueError("Datasets not loaded. Call load_datasets() first.")
        
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False
        )
        
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )
        
        logger.info(f"Created data loaders with batch size {self.batch_size}")
        return self.train_loader, self.test_loader
    
    def get_sample_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a sample batch from the training loader."""
        if self.train_loader is None:
            raise ValueError("Data loaders not created. Call create_data_loaders() first.")
        
        return next(iter(self.train_loader))


class DatasetAnalyzer:
    """Analyzes and visualizes dataset properties."""
    
    @staticmethod
    def analyze_dataset(dataset: datasets.MNIST) -> dict:
        """Analyze basic dataset properties."""
        sample_image, sample_label = dataset[0]
        
        analysis = {
            'total_samples': len(dataset),
            'num_classes': len(dataset.classes),
            'classes': dataset.classes,
            'image_shape': tuple(sample_image.shape),
            'image_dtype': sample_image.dtype,
            'pixel_range': (float(sample_image.min()), float(sample_image.max())),
            'total_pixels': sample_image.numel()
        }
        
        return analysis
    
    @staticmethod
    def visualize_samples(dataset: datasets.MNIST, num_samples: int = 8, 
                         save_path: Optional[str] = None) -> None:
        """Create a visualization grid of sample images."""
        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        fig.suptitle('MNIST Sample Digits', fontsize=16, fontweight='bold')
        
        for i in range(num_samples):
            image_tensor, label = dataset[i]
            image_np = image_tensor.squeeze().numpy()
            
            row, col = i // 4, i % 4
            axes[row, col].imshow(image_np, cmap='gray')
            axes[row, col].set_title(f'Label: {label}', fontweight='bold')
            axes[row, col].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Visualization saved to {save_path}")
        
        plt.show()
        plt.close()
    
    @staticmethod
    def analyze_tensor_properties(tensor: torch.Tensor) -> dict:
        """Analyze tensor properties and statistics."""
        return {
            'shape': tuple(tensor.shape),
            'dtype': tensor.dtype,
            'device': str(tensor.device),
            'requires_grad': tensor.requires_grad,
            'memory_usage_bytes': tensor.element_size() * tensor.numel(),
            'statistics': {
                'min': float(tensor.min()),
                'max': float(tensor.max()),
                'mean': float(tensor.mean()),
                'std': float(tensor.std()),
                'sum': float(tensor.sum())
            }
        }


class SystemInfo:
    """System and PyTorch environment information."""
    
    @staticmethod
    def get_pytorch_info() -> dict:
        """Get PyTorch installation information."""
        return {
            'pytorch_version': torch.__version__,
            'torchvision_version': torchvision.__version__,
            'cuda_available': torch.cuda.is_available(),
            'cuda_device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
        }
    
    @staticmethod
    def print_system_info() -> None:
        """Print formatted system information."""
        info = SystemInfo.get_pytorch_info()
        
        print("=" * 50)
        print("SYSTEM INFORMATION")
        print("=" * 50)
        
        for key, value in info.items():
            if value is not None:
                print(f"{key.replace('_', ' ').title()}: {value}")
        
        print()


def benchmark_data_loading(data_loader: torch.utils.data.DataLoader, 
                          num_batches: int = 10) -> float:
    """Benchmark data loading performance."""
    start_time = time.time()
    
    for i, (images, labels) in enumerate(data_loader):
        if i >= num_batches - 1:
            break
    
    end_time = time.time()
    total_time = end_time - start_time
    images_processed = num_batches * data_loader.batch_size
    
    logger.info(f"Processed {images_processed} images in {total_time:.3f}s "
                f"({images_processed/total_time:.0f} images/sec)")
    
    return total_time


def main():
    """Main execution function."""
    # System information
    SystemInfo.print_system_info()
    
    # Initialize data loader
    mnist_loader = MNISTDataLoader(batch_size=64)
    
    # Load datasets
    train_dataset, test_dataset = mnist_loader.load_datasets()
    
    # Analyze datasets
    train_analysis = DatasetAnalyzer.analyze_dataset(train_dataset)
    test_analysis = DatasetAnalyzer.analyze_dataset(test_dataset)
    
    logger.info("Dataset Analysis:")
    logger.info(f"Training: {train_analysis['total_samples']} samples")
    logger.info(f"Testing: {test_analysis['total_samples']} samples")
    logger.info(f"Image shape: {train_analysis['image_shape']}")
    logger.info(f"Classes: {train_analysis['num_classes']}")
    
    # Create visualization
    DatasetAnalyzer.visualize_samples(train_dataset, save_path='mnist_samples.png')
    
    # Create data loaders
    train_loader, test_loader = mnist_loader.create_data_loaders()
    
    # Get sample batch and analyze
    batch_images, batch_labels = mnist_loader.get_sample_batch()
    
    logger.info(f"Batch shape: {batch_images.shape}")
    logger.info(f"Sample labels: {batch_labels[:10].tolist()}")
    
    # Tensor analysis
    sample_tensor = batch_images[0]
    tensor_analysis = DatasetAnalyzer.analyze_tensor_properties(sample_tensor)
    
    logger.info("Sample Tensor Analysis:")
    for key, value in tensor_analysis.items():
        if key != 'statistics':
            logger.info(f"{key}: {value}")
    
    # Performance benchmark
    benchmark_data_loading(train_loader)
    
    logger.info("MNIST data loading and analysis complete")


if __name__ == "__main__":
    main()
