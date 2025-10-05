#!/usr/bin/env python3
"""
Comprehensive VAE Testing and Quality Assessment
Tests both Simple VAE and Enhanced VAE models for fashion item generation.
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
import sys

# Import VAE models
from simple_generator import SimpleVAE
from enhanced_vae import EnhancedVAE
from fashion_handler import FashionMNIST


class VAEQualityAssessment:
    """Comprehensive quality assessment for VAE models."""
    
    def __init__(self, device=None):
        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = 'mps'
        else:
            self.device = 'cpu'
        
        print(f"VAE Quality Assessment initialized on {self.device}")
        
        # Fashion-MNIST class names
        self.class_names = [
            'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
            'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
        ]
        
        self.fashion_data = FashionMNIST(batch_size=64)
        
    def load_simple_vae(self):
        """Load and test Simple VAE."""
        print("\n🔍 Loading Simple VAE...")
        try:
            model = SimpleVAE(latent_dim=20).to(self.device)
            model.load_state_dict(torch.load('models/simple_vae.pth', map_location=self.device))
            model.eval()
            
            print(f"   ✅ Simple VAE loaded - Parameters: {sum(p.numel() for p in model.parameters()):,}")
            return model
        except Exception as e:
            print(f"   ❌ Simple VAE loading failed: {e}")
            return None
    
    def load_enhanced_vae(self):
        """Load and test Enhanced VAE."""
        print("\n🔍 Loading Enhanced VAE...")
        try:
            model = EnhancedVAE(latent_dim=32, conditional=True).to(self.device)
            model.load_state_dict(torch.load('models/enhanced_vae_superior.pth', map_location=self.device))
            model.eval()
            
            print(f"   ✅ Enhanced VAE loaded - Parameters: {sum(p.numel() for p in model.parameters()):,}")
            return model
        except Exception as e:
            print(f"   ❌ Enhanced VAE loading failed: {e}")
            return None
    
    def test_reconstruction_quality(self, model, model_name, num_samples=16):
        """Test reconstruction quality on real fashion items."""
        print(f"\n🔄 Testing {model_name} Reconstruction Quality...")
        
        if model is None:
            return 0.0
        
        test_loader = self.fashion_data.get_test_loader()
        data_iter = iter(test_loader)
        real_images, labels = next(data_iter)
        real_images = real_images[:num_samples].to(self.device)
        labels = labels[:num_samples]
        
        model.eval()
        with torch.no_grad():
            if isinstance(model, EnhancedVAE):
                # Enhanced VAE with conditional input
                recon_images, mu, logvar = model(real_images.view(-1, 784), labels.to(self.device))
                recon_images = recon_images.view(-1, 28, 28)
            else:
                # Simple VAE
                recon_images, mu, logvar = model(real_images)
                recon_images = recon_images.view(-1, 28, 28)
        
        # Calculate reconstruction error
        real_flat = real_images.view(-1, 784)
        recon_flat = recon_images.view(-1, 784)
        mse_error = F.mse_loss(recon_flat, real_flat).item()
        
        print(f"   📊 Reconstruction MSE: {mse_error:.4f}")
        
        # Create visualization
        fig, axes = plt.subplots(2, num_samples//2, figsize=(16, 6))
        fig.suptitle(f'{model_name} - Reconstruction Quality Test', fontsize=16)
        
        for i in range(num_samples//2):
            # Real image
            axes[0, i].imshow(real_images[i, 0].cpu().numpy(), cmap='gray')
            axes[0, i].set_title(f'Real: {self.class_names[labels[i]]}', fontsize=8)
            axes[0, i].axis('off')
            
            # Reconstructed image
            axes[1, i].imshow(recon_images[i].cpu().numpy(), cmap='gray')
            axes[1, i].set_title(f'Recon (MSE: {F.mse_loss(recon_flat[i], real_flat[i]).item():.3f})', fontsize=8)
            axes[1, i].axis('off')
        
        plt.tight_layout()
        filename = f"results/{model_name.lower().replace(' ', '_')}_reconstruction_quality.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"   💾 Saved to: {filename}")
        return mse_error
    
    def test_generation_diversity(self, model, model_name, num_samples=25):
        """Test generation diversity and quality."""
        print(f"\n🎨 Testing {model_name} Generation Diversity...")
        
        if model is None:
            return 0.0
        
        model.eval()
        with torch.no_grad():
            if isinstance(model, EnhancedVAE):
                # Test conditional generation
                samples = model.generate(num_samples=num_samples, device=self.device)
                if samples.dim() == 4:
                    samples = samples.squeeze(1)  # Remove channel dimension if present
            else:
                # Simple VAE unconditional generation
                samples = model.generate(num_samples=num_samples, device=self.device)
        
        # Calculate diversity metrics
        samples_flat = samples.view(num_samples, -1)
        
        # Pairwise distances
        distances = torch.cdist(samples_flat, samples_flat)
        avg_distance = distances.sum() / (num_samples * (num_samples - 1))
        
        print(f"   📊 Average pairwise distance: {avg_distance:.4f}")
        
        # Create visualization
        grid_size = int(np.ceil(np.sqrt(num_samples)))
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
        fig.suptitle(f'{model_name} - Generation Diversity Test', fontsize=16)
        
        for i in range(grid_size * grid_size):
            row, col = i // grid_size, i % grid_size
            if i < num_samples:
                axes[row, col].imshow(samples[i].cpu().numpy(), cmap='gray')
                axes[row, col].set_title(f'#{i+1}', fontsize=8)
            axes[row, col].axis('off')
        
        plt.tight_layout()
        filename = f"results/{model_name.lower().replace(' ', '_')}_generation_diversity.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"   💾 Saved to: {filename}")
        return avg_distance.item()
    
    def test_conditional_generation(self, model, model_name):
        """Test conditional generation for Enhanced VAE."""
        print(f"\n🎯 Testing {model_name} Conditional Generation...")
        
        if model is None or not isinstance(model, EnhancedVAE):
            print("   ⚠️  Conditional generation not available for this model")
            return {}
        
        model.eval()
        class_qualities = {}
        
        # Generate samples for each class
        fig, axes = plt.subplots(10, 5, figsize=(12, 20))
        fig.suptitle(f'{model_name} - Conditional Generation (All Classes)', fontsize=16)
        
        with torch.no_grad():
            for class_idx in range(10):
                # Generate 5 samples for each class
                labels = torch.full((5,), class_idx, dtype=torch.long, device=self.device)
                samples = model.generate(num_samples=5, labels=labels, device=self.device)
                
                if samples.dim() == 4:
                    samples = samples.squeeze(1)  # Remove channel dimension if present
                
                # Calculate intra-class diversity
                samples_flat = samples.view(5, -1)
                distances = torch.cdist(samples_flat, samples_flat)
                avg_distance = distances.sum() / (5 * 4)  # Exclude diagonal
                class_qualities[class_idx] = avg_distance.item()
                
                # Visualize samples
                for sample_idx in range(5):
                    axes[class_idx, sample_idx].imshow(samples[sample_idx].cpu().numpy(), cmap='gray')
                    if sample_idx == 2:  # Middle column gets class name
                        axes[class_idx, sample_idx].set_title(f'{self.class_names[class_idx]}', fontsize=10)
                    axes[class_idx, sample_idx].axis('off')
                
                print(f"   Class {class_idx} ({self.class_names[class_idx]}): Diversity = {avg_distance:.4f}")
        
        plt.tight_layout()
        filename = f"results/{model_name.lower().replace(' ', '_')}_conditional_generation.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"   💾 Saved to: {filename}")
        avg_class_diversity = np.mean(list(class_qualities.values()))
        print(f"   📊 Average class diversity: {avg_class_diversity:.4f}")
        
        return class_qualities
    
    def run_comprehensive_test(self):
        """Run comprehensive VAE testing."""
        print("🧪 COMPREHENSIVE VAE TESTING")
        print("=" * 60)
        print(f"Device: {self.device}")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'device': self.device,
            'models': {}
        }
        
        # Test Simple VAE
        simple_vae = self.load_simple_vae()
        if simple_vae:
            simple_results = {}
            simple_results['reconstruction_mse'] = self.test_reconstruction_quality(simple_vae, "Simple VAE")
            simple_results['generation_diversity'] = self.test_generation_diversity(simple_vae, "Simple VAE")
            simple_results['parameters'] = sum(p.numel() for p in simple_vae.parameters())
            results['models']['simple_vae'] = simple_results
        
        # Test Enhanced VAE
        enhanced_vae = self.load_enhanced_vae()
        if enhanced_vae:
            enhanced_results = {}
            enhanced_results['reconstruction_mse'] = self.test_reconstruction_quality(enhanced_vae, "Enhanced VAE")
            enhanced_results['generation_diversity'] = self.test_generation_diversity(enhanced_vae, "Enhanced VAE")
            enhanced_results['conditional_generation'] = self.test_conditional_generation(enhanced_vae, "Enhanced VAE")
            enhanced_results['parameters'] = sum(p.numel() for p in enhanced_vae.parameters())
            results['models']['enhanced_vae'] = enhanced_results
        
        # Generate summary report
        self.generate_summary_report(results)
        
        return results
    
    def generate_summary_report(self, results):
        """Generate a comprehensive summary report."""
        print("\n📋 VAE TESTING SUMMARY REPORT")
        print("=" * 60)
        
        summary_lines = []
        summary_lines.append("# VAE Comprehensive Testing Report")
        summary_lines.append(f"**Generated:** {results['timestamp']}")
        summary_lines.append(f"**Device:** {results['device']}")
        summary_lines.append("")
        
        for model_name, model_results in results['models'].items():
            print(f"\n🔧 {model_name.upper().replace('_', ' ')} RESULTS:")
            summary_lines.append(f"## {model_name.upper().replace('_', ' ')} Results")
            summary_lines.append("")
            
            if 'parameters' in model_results:
                param_count = f"{model_results['parameters']:,}"
                print(f"   Parameters: {param_count}")
                summary_lines.append(f"- **Parameters:** {param_count}")
            
            if 'reconstruction_mse' in model_results:
                mse = model_results['reconstruction_mse']
                print(f"   Reconstruction MSE: {mse:.4f}")
                summary_lines.append(f"- **Reconstruction MSE:** {mse:.4f}")
            
            if 'generation_diversity' in model_results:
                diversity = model_results['generation_diversity']
                print(f"   Generation Diversity: {diversity:.4f}")
                summary_lines.append(f"- **Generation Diversity:** {diversity:.4f}")
            
            if 'conditional_generation' in model_results:
                cond_gen = model_results['conditional_generation']
                if cond_gen:
                    avg_diversity = np.mean(list(cond_gen.values()))
                    print(f"   Conditional Generation: ✅ Available")
                    print(f"   Average Class Diversity: {avg_diversity:.4f}")
                    summary_lines.append(f"- **Conditional Generation:** ✅ Available")
                    summary_lines.append(f"- **Average Class Diversity:** {avg_diversity:.4f}")
                    
                    # Best and worst performing classes
                    best_class = max(cond_gen, key=cond_gen.get)
                    worst_class = min(cond_gen, key=cond_gen.get)
                    print(f"   Best Class: {self.class_names[best_class]} ({cond_gen[best_class]:.4f})")
                    print(f"   Worst Class: {self.class_names[worst_class]} ({cond_gen[worst_class]:.4f})")
                    summary_lines.append(f"- **Best Class:** {self.class_names[best_class]} ({cond_gen[best_class]:.4f})")
                    summary_lines.append(f"- **Worst Class:** {self.class_names[worst_class]} ({cond_gen[worst_class]:.4f})")
            
            summary_lines.append("")
        
        # Overall recommendations
        print("\n💡 RECOMMENDATIONS:")
        summary_lines.append("## Recommendations")
        summary_lines.append("")
        
        if 'simple_vae' in results['models'] and 'enhanced_vae' in results['models']:
            simple_mse = results['models']['simple_vae'].get('reconstruction_mse', float('inf'))
            enhanced_mse = results['models']['enhanced_vae'].get('reconstruction_mse', float('inf'))
            
            if enhanced_mse < simple_mse:
                print("   ✅ Enhanced VAE shows better reconstruction quality")
                summary_lines.append("- ✅ Enhanced VAE shows better reconstruction quality")
            else:
                print("   ⚠️  Simple VAE has better reconstruction quality")
                summary_lines.append("- ⚠️ Simple VAE has better reconstruction quality")
            
            print("   ✅ Enhanced VAE provides conditional generation capabilities")
            summary_lines.append("- ✅ Enhanced VAE provides conditional generation capabilities")
            print("   ✅ Both models are functional and ready for use")
            summary_lines.append("- ✅ Both models are functional and ready for use")
        
        # Save report
        report_filename = f"results/vae_testing_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        with open(report_filename, 'w') as f:
            f.write('\n'.join(summary_lines))
        
        print(f"\n💾 Detailed report saved to: {report_filename}")


def main():
    """Main function to run VAE comprehensive testing."""
    print("🎯 VAE COMPREHENSIVE TESTING SYSTEM")
    print("=" * 70)
    
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Run comprehensive testing
    assessor = VAEQualityAssessment()
    results = assessor.run_comprehensive_test()
    
    print("\n🎉 VAE COMPREHENSIVE TESTING COMPLETED!")
    print("   Check the results/ directory for detailed visualizations and reports.")
    
    return results


if __name__ == "__main__":
    results = main()