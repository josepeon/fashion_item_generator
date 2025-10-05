#!/usr/bin/env python3
"""
VAE Improvement and Usage Guide
Comprehensive guide for using and improving the Fashion-MNIST VAE models.
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import os


class VAEUsageGuide:
    """Guide for using VAE models effectively."""
    
    def __init__(self, device=None):
        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = 'mps'
        else:
            self.device = 'cpu'
        
        print(f"VAE Usage Guide initialized on {self.device}")
        
        # Fashion-MNIST class names
        self.class_names = [
            'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
            'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
        ]
    
    def demonstrate_simple_vae_usage(self):
        """Demonstrate how to use Simple VAE effectively."""
        print("\n🎨 SIMPLE VAE USAGE DEMONSTRATION")
        print("=" * 50)
        
        try:
            from simple_generator import SimpleVAE
            
            # Load model
            model = SimpleVAE(latent_dim=20).to(self.device)
            model.load_state_dict(torch.load('models/simple_vae.pth', map_location=self.device, weights_only=True))
            model.eval()
            
            print("✅ Simple VAE loaded successfully")
            print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
            
            # Example 1: Basic generation
            print("\n📝 Example 1: Basic Fashion Item Generation")
            with torch.no_grad():
                samples = model.generate(num_samples=8, device=self.device)
                print(f"   Generated {samples.shape} fashion items")
            
            # Example 2: Batch generation
            print("\n📝 Example 2: Batch Generation for Variety")
            with torch.no_grad():
                batch_samples = []
                for _ in range(5):  # Generate 5 batches
                    batch = model.generate(num_samples=4, device=self.device)
                    batch_samples.append(batch)
                all_samples = torch.cat(batch_samples, dim=0)
                print(f"   Generated {all_samples.shape} fashion items across batches")
            
            # Example 3: Latent space interpolation
            print("\n📝 Example 3: Latent Space Interpolation")
            with torch.no_grad():
                # Sample two random points in latent space
                z1 = torch.randn(1, 20, device=self.device)
                z2 = torch.randn(1, 20, device=self.device)
                
                # Interpolate between them
                interpolations = []
                for alpha in np.linspace(0, 1, 10):
                    z_interp = alpha * z1 + (1 - alpha) * z2
                    sample = model.decode(z_interp).view(28, 28)
                    interpolations.append(sample)
                
                print(f"   Created {len(interpolations)} interpolated samples")
            
            # Create usage demonstration visualization
            fig, axes = plt.subplots(3, 8, figsize=(16, 6))
            fig.suptitle('Simple VAE Usage Examples', fontsize=16)
            
            # Row 1: Basic generation
            for i in range(8):
                axes[0, i].imshow(samples[i].cpu().numpy(), cmap='gray')
                axes[0, i].set_title(f'Gen {i+1}', fontsize=8)
                axes[0, i].axis('off')
            
            # Row 2: Batch generation (first 8)
            for i in range(8):
                axes[1, i].imshow(all_samples[i].cpu().numpy(), cmap='gray')
                axes[1, i].set_title(f'Batch {i+1}', fontsize=8)
                axes[1, i].axis('off')
            
            # Row 3: Interpolation (first 8 steps)
            for i in range(8):
                axes[2, i].imshow(interpolations[i].cpu().numpy(), cmap='gray')
                axes[2, i].set_title(f'Interp {i+1}', fontsize=8)
                axes[2, i].axis('off')
            
            plt.tight_layout()
            plt.savefig('results/simple_vae_usage_examples.png', dpi=150, bbox_inches='tight')
            plt.close()
            
            print("💾 Usage examples saved to: results/simple_vae_usage_examples.png")
            
        except Exception as e:
            print(f"❌ Simple VAE demonstration failed: {e}")
    
    def demonstrate_enhanced_vae_usage(self):
        """Demonstrate how to use Enhanced VAE effectively."""
        print("\n🎯 ENHANCED VAE USAGE DEMONSTRATION")
        print("=" * 50)
        
        try:
            from enhanced_vae import EnhancedVAE
            
            # Load model
            model = EnhancedVAE(latent_dim=32, conditional=True).to(self.device)
            model.load_state_dict(torch.load('models/enhanced_vae_superior.pth', map_location=self.device, weights_only=True))
            model.eval()
            
            print("✅ Enhanced VAE loaded successfully")
            print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
            
            # Example 1: Conditional generation
            print("\n📝 Example 1: Conditional Fashion Class Generation")
            with torch.no_grad():
                # Generate specific fashion items
                dresses = model.generate_fashion_class(3, num_samples=3, device=self.device)  # Dresses
                sneakers = model.generate_fashion_class(7, num_samples=3, device=self.device)  # Sneakers
                bags = model.generate_fashion_class(8, num_samples=3, device=self.device)  # Bags
                
                print(f"   Generated {dresses.shape} dresses")
                print(f"   Generated {sneakers.shape} sneakers") 
                print(f"   Generated {bags.shape} bags")
            
            # Example 2: Fashion collection generation
            print("\n📝 Example 2: Complete Fashion Collection")
            collection = {}
            with torch.no_grad():
                for class_idx in range(10):
                    items = model.generate_fashion_class(class_idx, num_samples=2, device=self.device)
                    collection[class_idx] = items
                    print(f"   {self.class_names[class_idx]}: {items.shape}")
            
            # Example 3: Conditional interpolation
            print("\n📝 Example 3: Class-Conditional Interpolation")
            with torch.no_grad():
                # Interpolate between two dresses
                z1 = torch.randn(1, 32, device=self.device)
                z2 = torch.randn(1, 32, device=self.device)
                dress_label = torch.tensor([3], device=self.device)  # Dress class
                
                dress_interpolations = []
                for alpha in np.linspace(0, 1, 8):
                    z_interp = alpha * z1 + (1 - alpha) * z2
                    sample = model.decode(z_interp, dress_label).view(28, 28)
                    dress_interpolations.append(sample)
                
                print(f"   Created {len(dress_interpolations)} dress interpolations")
            
            # Create enhanced usage demonstration visualization
            fig, axes = plt.subplots(4, 8, figsize=(16, 8))
            fig.suptitle('Enhanced VAE Usage Examples', fontsize=16)
            
            # Row 1: Specific class examples
            examples = [dresses, sneakers, bags]
            class_names_ex = ['Dress', 'Sneaker', 'Bag']
            for i in range(3):
                for j in range(3):
                    col_idx = i * 3 + j
                    if col_idx < 8:
                        img = examples[i][j].squeeze().cpu().numpy()
                        axes[0, col_idx].imshow(img, cmap='gray')
                        axes[0, col_idx].set_title(f'{class_names_ex[i]} {j+1}', fontsize=8)
                        axes[0, col_idx].axis('off')
            
            # Fill remaining columns in row 1
            for col_idx in range(7, 8):
                axes[0, col_idx].axis('off')
            
            # Row 2: Fashion collection (one item per class, first 8 classes)
            for class_idx in range(8):
                img = collection[class_idx][0].squeeze().cpu().numpy()
                axes[1, class_idx].imshow(img, cmap='gray')
                axes[1, class_idx].set_title(f'{self.class_names[class_idx][:8]}', fontsize=8)
                axes[1, class_idx].axis('off')
            
            # Row 3: Second item from collection
            for class_idx in range(8):
                img = collection[class_idx][1].squeeze().cpu().numpy()
                axes[2, class_idx].imshow(img, cmap='gray')
                axes[2, class_idx].set_title(f'{self.class_names[class_idx][:8]} 2', fontsize=8)
                axes[2, class_idx].axis('off')
            
            # Row 4: Dress interpolations
            for i in range(8):
                axes[3, i].imshow(dress_interpolations[i].cpu().numpy(), cmap='gray')
                axes[3, i].set_title(f'Dress Int {i+1}', fontsize=8)
                axes[3, i].axis('off')
            
            plt.tight_layout()
            plt.savefig('results/enhanced_vae_usage_examples.png', dpi=150, bbox_inches='tight')
            plt.close()
            
            print("💾 Usage examples saved to: results/enhanced_vae_usage_examples.png")
            
        except Exception as e:
            print(f"❌ Enhanced VAE demonstration failed: {e}")
    
    def provide_improvement_suggestions(self):
        """Provide suggestions for VAE improvements."""
        print("\n💡 VAE IMPROVEMENT SUGGESTIONS")
        print("=" * 50)
        
        suggestions = [
            {
                "title": "1. Architecture Improvements",
                "details": [
                    "• Add more residual blocks for deeper architecture",
                    "• Implement attention mechanisms in decoder",
                    "• Use spectral normalization for training stability",
                    "• Add progressive growing for higher resolution"
                ]
            },
            {
                "title": "2. Training Enhancements",
                "details": [
                    "• Implement β-VAE annealing schedule",
                    "• Add perceptual loss using pre-trained features",
                    "• Use adversarial training (VAE-GAN hybrid)",
                    "• Implement curriculum learning"
                ]
            },
            {
                "title": "3. Conditional Generation Improvements",
                "details": [
                    "• Add multi-attribute conditioning (color, style, etc.)",
                    "• Implement disentangled representation learning",
                    "• Add auxiliary classifier for better conditioning",
                    "• Use label smoothing for more robust training"
                ]
            },
            {
                "title": "4. Evaluation and Quality Metrics",
                "details": [
                    "• Implement FID (Frechet Inception Distance) scoring",
                    "• Add IS (Inception Score) evaluation",
                    "• Use LPIPS for perceptual similarity",
                    "• Implement semantic consistency checks"
                ]
            }
        ]
        
        for suggestion in suggestions:
            print(f"\n{suggestion['title']}:")
            for detail in suggestion['details']:
                print(f"  {detail}")
        
        # Save improvement suggestions to file
        with open('results/vae_improvement_suggestions.md', 'w') as f:
            f.write("# VAE Improvement Suggestions\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            for suggestion in suggestions:
                f.write(f"## {suggestion['title']}\n\n")
                for detail in suggestion['details']:
                    f.write(f"{detail}\n")
                f.write("\n")
        
        print("\n💾 Improvement suggestions saved to: results/vae_improvement_suggestions.md")
    
    def create_vae_usage_code_examples(self):
        """Create code examples for VAE usage."""
        print("\n📝 CREATING VAE USAGE CODE EXAMPLES")
        print("=" * 50)
        
        simple_vae_example = '''#!/usr/bin/env python3
"""
Simple VAE Usage Example
Shows how to use the Simple VAE for fashion item generation.
"""

import torch
import matplotlib.pyplot as plt
from simple_generator import SimpleVAE

def generate_fashion_items():
    """Generate fashion items using Simple VAE."""
    # Setup device
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
    # Load model
    model = SimpleVAE(latent_dim=20).to(device)
    model.load_state_dict(torch.load('models/simple_vae.pth', map_location=device))
    model.eval()
    
    # Generate samples
    with torch.no_grad():
        samples = model.generate(num_samples=16, device=device)
    
    # Visualize
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    for i in range(16):
        row, col = i // 4, i % 4
        axes[row, col].imshow(samples[i].cpu().numpy(), cmap='gray')
        axes[row, col].axis('off')
    
    plt.suptitle('Generated Fashion Items - Simple VAE')
    plt.show()

if __name__ == "__main__":
    generate_fashion_items()
'''
        
        enhanced_vae_example = '''#!/usr/bin/env python3
"""
Enhanced VAE Usage Example
Shows how to use the Enhanced VAE for conditional fashion generation.
"""

import torch
import matplotlib.pyplot as plt
from enhanced_vae import EnhancedVAE

def generate_specific_fashion_items():
    """Generate specific fashion items using Enhanced VAE."""
    # Setup device
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
    # Load model
    model = EnhancedVAE(latent_dim=32, conditional=True).to(device)
    model.load_state_dict(torch.load('models/enhanced_vae_superior.pth', map_location=device))
    model.eval()
    
    # Fashion class names
    class_names = [
        'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
    ]
    
    # Generate specific items
    with torch.no_grad():
        # Generate 3 dresses
        dresses = model.generate_fashion_class(3, num_samples=3, device=device)
        
        # Generate 3 sneakers
        sneakers = model.generate_fashion_class(7, num_samples=3, device=device)
        
        # Generate one item from each class
        collection = []
        for class_idx in range(10):
            item = model.generate_fashion_class(class_idx, num_samples=1, device=device)
            collection.append(item)
    
    # Visualize specific items
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    
    # Show dresses
    for i in range(3):
        axes[0, i].imshow(dresses[i].squeeze().cpu().numpy(), cmap='gray')
        axes[0, i].set_title(f'Dress {i+1}')
        axes[0, i].axis('off')
    
    # Show sneakers
    for i in range(3):
        axes[1, i].imshow(sneakers[i].squeeze().cpu().numpy(), cmap='gray')
        axes[1, i].set_title(f'Sneaker {i+1}')
        axes[1, i].axis('off')
    
    plt.suptitle('Conditional Fashion Generation - Enhanced VAE')
    plt.tight_layout()
    plt.show()
    
    # Show complete collection
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for i in range(10):
        row, col = i // 5, i % 5
        axes[row, col].imshow(collection[i][0].squeeze().cpu().numpy(), cmap='gray')
        axes[row, col].set_title(class_names[i])
        axes[row, col].axis('off')
    
    plt.suptitle('Complete Fashion Collection - One Item Per Class')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    generate_specific_fashion_items()
'''
        
        # Save code examples
        with open('results/simple_vae_usage_example.py', 'w') as f:
            f.write(simple_vae_example)
        
        with open('results/enhanced_vae_usage_example.py', 'w') as f:
            f.write(enhanced_vae_example)
        
        print("💾 Code examples saved to:")
        print("   - results/simple_vae_usage_example.py")
        print("   - results/enhanced_vae_usage_example.py")
    
    def run_complete_guide(self):
        """Run the complete VAE usage guide."""
        print("🎯 VAE COMPREHENSIVE USAGE GUIDE")
        print("=" * 60)
        print(f"Device: {self.device}")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Create results directory
        os.makedirs('results', exist_ok=True)
        
        # Run all demonstrations
        self.demonstrate_simple_vae_usage()
        self.demonstrate_enhanced_vae_usage()
        self.provide_improvement_suggestions()
        self.create_vae_usage_code_examples()
        
        print("\n🎉 VAE USAGE GUIDE COMPLETED!")
        print("   All examples, suggestions, and code samples saved to results/")


def main():
    """Main function to run VAE usage guide."""
    guide = VAEUsageGuide()
    guide.run_complete_guide()


if __name__ == "__main__":
    main()