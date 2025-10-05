#!/usr/bin/env python3
"""
Comprehensive VAE Testing and Evaluation

Test and evaluate VAE models for:
- Generation quality
- Reconstruction accuracy  
- Latent space interpolation
- Fashion class conditional generation
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import os
import json
from typing import Dict, List, Tuple

import sys
sys.path.append('.')

from src.fashion_handler import FashionMNIST
from src.enhanced_vae import EnhancedVAE


class VAEEvaluator:
    """Comprehensive VAE model evaluator."""
    
    def __init__(self, model_path: str, latent_dim: int = 32, conditional: bool = True):
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        self.model_path = model_path
        self.latent_dim = latent_dim
        self.conditional = conditional
        
        # Fashion-MNIST class names
        self.class_names = [
            'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
            'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
        ]
        
        # Load model
        self.model = EnhancedVAE(latent_dim=latent_dim, conditional=conditional).to(self.device)
        
        try:
            if os.path.exists(model_path):
                state_dict = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                print(f"✅ Loaded model: {model_path}")
            else:
                print(f"⚠️  Model not found: {model_path}")
                print("Using untrained model for architecture testing")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("Using untrained model")
        
        self.model.eval()
        
        # Load test data
        self.fashion_data = FashionMNIST(batch_size=64)
        self.test_loader = self.fashion_data.get_test_loader()
        
        print(f"🔬 VAE Evaluator initialized")
        print(f"   Device: {self.device}")
        print(f"   Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"   Conditional: {conditional}")
        print(f"   Latent dimensions: {latent_dim}")
    
    def test_reconstruction(self, num_samples: int = 16) -> Dict[str, float]:
        """Test reconstruction quality."""
        print("\n🔍 Testing reconstruction quality...")
        
        with torch.no_grad():
            # Get test batch
            data_iter = iter(self.test_loader)
            data, labels = next(data_iter)
            data = data[:num_samples].to(self.device)
            labels = labels[:num_samples].to(self.device)
            
            # Normalize to [-1, 1] (matching training)
            data_normalized = data * 2.0 - 1.0
            data_flat = data_normalized.view(data_normalized.size(0), -1)
            
            # Reconstruct
            if self.conditional:
                recon, mu, logvar = self.model(data_flat, labels)
            else:
                recon, mu, logvar = self.model(data_flat)
            
            # Convert back to [0, 1] range
            recon = (recon + 1.0) / 2.0
            recon = torch.clamp(recon, 0, 1)
            recon = recon.view(-1, 1, 28, 28)
            
            # Calculate metrics
            mse_loss = F.mse_loss(recon, data, reduction='mean').item()
            ssim_scores = []
            
            # Calculate per-image SSIM (simplified)
            for i in range(num_samples):
                orig = data[i, 0].cpu().numpy()
                rec = recon[i, 0].cpu().numpy()
                
                # Simple SSIM approximation
                mu1, mu2 = orig.mean(), rec.mean()
                sigma1, sigma2 = orig.std(), rec.std()
                sigma12 = ((orig - mu1) * (rec - mu2)).mean()
                
                c1, c2 = 0.01**2, 0.03**2
                ssim = ((2*mu1*mu2 + c1) * (2*sigma12 + c2)) / ((mu1**2 + mu2**2 + c1) * (sigma1**2 + sigma2**2 + c2))
                ssim_scores.append(ssim)
            
            avg_ssim = np.mean(ssim_scores)
            
            # Visualize results
            self._visualize_reconstruction(data[:8], recon[:8], labels[:8])
            
            metrics = {
                'mse_loss': mse_loss,
                'avg_ssim': avg_ssim,
                'reconstruction_quality': (1 - mse_loss) * avg_ssim  # Combined metric
            }
            
            print(f"   MSE Loss: {mse_loss:.4f}")
            print(f"   Average SSIM: {avg_ssim:.4f}")
            print(f"   Quality Score: {metrics['reconstruction_quality']:.4f}")
            
            return metrics
    
    def test_generation(self, samples_per_class: int = 8) -> Dict[str, float]:
        """Test generation quality and diversity."""
        print(f"\n🎨 Testing generation quality...")
        
        if not self.conditional:
            print("⚠️  Non-conditional model - generating random samples")
            samples_per_class = 8
            with torch.no_grad():
                generated = self.model.generate(samples_per_class * 10, device=self.device)
                generated = (generated + 1.0) / 2.0
                generated = torch.clamp(generated, 0, 1)
                
                self._visualize_generation(generated[:64], None)
                
                # Simple diversity metric
                flat_gen = generated.view(generated.size(0), -1)
                distances = torch.pdist(flat_gen)
                diversity = distances.mean().item()
                
                return {'diversity': diversity, 'avg_quality': 0.5}
        
        # Conditional generation - test each class
        all_generated = []
        class_qualities = []
        
        with torch.no_grad():
            for class_idx in range(10):
                class_samples = self.model.generate_fashion_class(
                    class_idx, samples_per_class, self.device
                )
                # Convert back to [0, 1] range
                class_samples = (class_samples + 1.0) / 2.0
                class_samples = torch.clamp(class_samples, 0, 1)
                
                all_generated.append(class_samples)
                
                # Simple quality metric based on activation patterns
                flat_samples = class_samples.view(class_samples.size(0), -1)
                activation_std = flat_samples.std(dim=1).mean().item()
                class_qualities.append(activation_std)
        
        # Combine all generated samples
        all_samples = torch.cat(all_generated, dim=0)
        
        # Calculate diversity (average pairwise distance)
        flat_all = all_samples.view(all_samples.size(0), -1)
        distances = torch.pdist(flat_all)
        diversity = distances.mean().item()
        
        # Average quality across classes
        avg_quality = np.mean(class_qualities)
        
        # Visualize generated samples
        self._visualize_generation(all_samples[:80], list(range(10)) * 8)
        
        metrics = {
            'diversity': diversity,
            'avg_quality': avg_quality,
            'class_qualities': class_qualities,
            'generation_score': diversity * avg_quality  # Combined metric
        }
        
        print(f"   Diversity: {diversity:.4f}")
        print(f"   Average Quality: {avg_quality:.4f}")
        print(f"   Generation Score: {metrics['generation_score']:.4f}")
        
        return metrics
    
    def test_latent_interpolation(self, class1: int = 0, class2: int = 6, steps: int = 8):
        """Test latent space interpolation between two classes."""
        print(f"\\n🔄 Testing latent interpolation: {self.class_names[class1]} → {self.class_names[class2]}")
        
        if not self.conditional:
            print("⚠️  Interpolation requires conditional model")
            return {}
        
        with torch.no_grad():
            # Generate latent codes for both classes
            z1 = torch.randn(1, self.latent_dim, device=self.device)
            z2 = torch.randn(1, self.latent_dim, device=self.device)
            
            # Create interpolation
            interpolations = []
            alphas = np.linspace(0, 1, steps)
            
            for alpha in alphas:
                z_interp = (1 - alpha) * z1 + alpha * z2
                
                # Use class1 for first half, class2 for second half
                class_label = class1 if alpha < 0.5 else class2
                labels = torch.tensor([class_label], device=self.device)
                
                sample = self.model.decode(z_interp, labels)
                sample = (sample + 1.0) / 2.0
                sample = torch.clamp(sample, 0, 1)
                sample = sample.view(1, 28, 28)
                
                interpolations.append(sample)
            
            # Visualize interpolation
            self._visualize_interpolation(interpolations, class1, class2)
            
            # Simple smoothness metric
            smoothness_scores = []
            for i in range(len(interpolations) - 1):
                diff = F.mse_loss(interpolations[i], interpolations[i+1]).item()
                smoothness_scores.append(diff)
            
            avg_smoothness = np.mean(smoothness_scores)
            
            print(f"   Average smoothness: {avg_smoothness:.4f}")
            
            return {'interpolation_smoothness': avg_smoothness}
    
    def _visualize_reconstruction(self, original, reconstructed, labels):
        """Visualize reconstruction results."""
        fig, axes = plt.subplots(2, 8, figsize=(16, 4))
        fig.suptitle('VAE Reconstruction Test', fontsize=16)
        
        for i in range(8):
            # Original
            axes[0, i].imshow(original[i, 0].cpu().numpy(), cmap='gray')
            axes[0, i].set_title(f'Original\\n{self.class_names[labels[i]]}', fontsize=8)
            axes[0, i].axis('off')
            
            # Reconstructed
            axes[1, i].imshow(reconstructed[i, 0].cpu().numpy(), cmap='gray')
            axes[1, i].set_title('Reconstructed', fontsize=8)
            axes[1, i].axis('off')
        
        plt.tight_layout()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plt.savefig(f'results/vae_reconstruction_{timestamp}.png', dpi=300, bbox_inches='tight')
        print(f"   📊 Reconstruction saved: results/vae_reconstruction_{timestamp}.png")
        plt.show()
    
    def _visualize_generation(self, generated, class_labels):
        """Visualize generated samples."""
        if generated.size(0) > 80:
            generated = generated[:80]
            
        rows, cols = 8, 10
        fig, axes = plt.subplots(rows, cols, figsize=(20, 16))
        fig.suptitle('VAE Generated Fashion Items', fontsize=16)
        
        for i in range(min(80, generated.size(0))):
            row = i // cols
            col = i % cols
            
            axes[row, col].imshow(generated[i, 0].cpu().numpy(), cmap='gray')
            
            if class_labels and i < len(class_labels):
                class_idx = class_labels[i]
                axes[row, col].set_title(self.class_names[class_idx], fontsize=8)
            else:
                axes[row, col].set_title(f'Sample {i}', fontsize=8)
                
            axes[row, col].axis('off')
        
        plt.tight_layout()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plt.savefig(f'results/vae_generation_{timestamp}.png', dpi=300, bbox_inches='tight')
        print(f"   📊 Generation saved: results/vae_generation_{timestamp}.png")
        plt.show()
    
    def _visualize_interpolation(self, interpolations, class1, class2):
        """Visualize latent space interpolation."""
        fig, axes = plt.subplots(1, len(interpolations), figsize=(16, 2))
        fig.suptitle(f'Latent Interpolation: {self.class_names[class1]} → {self.class_names[class2]}', fontsize=14)
        
        for i, sample in enumerate(interpolations):
            axes[i].imshow(sample[0].cpu().numpy(), cmap='gray')
            axes[i].set_title(f'Step {i}', fontsize=10)
            axes[i].axis('off')
        
        plt.tight_layout()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plt.savefig(f'results/vae_interpolation_{timestamp}.png', dpi=300, bbox_inches='tight')
        print(f"   📊 Interpolation saved: results/vae_interpolation_{timestamp}.png")
        plt.show()
    
    def comprehensive_evaluation(self) -> Dict:
        """Run comprehensive VAE evaluation."""
        print(f"\\n🔬 COMPREHENSIVE VAE EVALUATION")
        print("=" * 60)
        print(f"Model: {self.model_path}")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        results = {}
        
        # Test reconstruction
        results['reconstruction'] = self.test_reconstruction()
        
        # Test generation
        results['generation'] = self.test_generation()
        
        # Test interpolation
        results['interpolation'] = self.test_latent_interpolation()
        
        # Calculate overall score
        recon_score = results['reconstruction'].get('reconstruction_quality', 0)
        gen_score = results['generation'].get('generation_score', 0)
        interp_score = 1 - results['interpolation'].get('interpolation_smoothness', 1)  # Lower is better
        
        overall_score = (recon_score + gen_score + interp_score) / 3
        results['overall_score'] = overall_score
        
        print(f"\\n🎯 EVALUATION SUMMARY")
        print("-" * 40)
        print(f"   Reconstruction Quality: {recon_score:.4f}")
        print(f"   Generation Score: {gen_score:.4f}")
        print(f"   Interpolation Quality: {interp_score:.4f}")
        print(f"   Overall Score: {overall_score:.4f}")
        
        # Grade the model
        if overall_score >= 0.8:
            grade = "A - EXCELLENT"
        elif overall_score >= 0.6:
            grade = "B - GOOD"
        elif overall_score >= 0.4:
            grade = "C - FAIR"
        else:
            grade = "D - NEEDS IMPROVEMENT"
        
        print(f"   Grade: {grade}")
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = f'results/vae_evaluation_{timestamp}.json'
        
        # Convert numpy arrays to lists for JSON serialization
        json_results = self._convert_for_json(results)
        json_results['model_path'] = self.model_path
        json_results['timestamp'] = timestamp
        json_results['grade'] = grade
        
        with open(results_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"   Results saved: {results_file}")
        
        return results
    
    def _convert_for_json(self, obj):
        """Convert numpy arrays and tensors to JSON-serializable format."""
        if isinstance(obj, dict):
            return {k: self._convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_for_json(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        else:
            return obj


def evaluate_all_vae_models():
    """Evaluate all available VAE models."""
    print("🚀 EVALUATING ALL VAE MODELS")
    print("=" * 70)
    
    # List of models to test
    models_to_test = [
        ('models/enhanced_vae_superior.pth', 32, True, 'Enhanced VAE (Current)'),
        ('models/simple_vae.pth', 20, False, 'Simple VAE (Baseline)'),
    ]
    
    results_summary = {}
    
    for model_path, latent_dim, conditional, description in models_to_test:
        if os.path.exists(model_path):
            print(f"\\n{'='*60}")
            print(f"TESTING: {description}")
            print(f"Path: {model_path}")
            print(f"{'='*60}")
            
            try:
                evaluator = VAEEvaluator(model_path, latent_dim, conditional)
                results = evaluator.comprehensive_evaluation()
                results_summary[description] = results['overall_score']
            except Exception as e:
                print(f"❌ Error evaluating {description}: {e}")
                results_summary[description] = 0.0
        else:
            print(f"⚠️  Model not found: {model_path}")
            results_summary[description] = 0.0
    
    # Print summary
    print(f"\\n🏆 FINAL COMPARISON")
    print("=" * 50)
    sorted_results = sorted(results_summary.items(), key=lambda x: x[1], reverse=True)
    
    for i, (model_name, score) in enumerate(sorted_results, 1):
        medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
        print(f"{medal} {i}. {model_name:<30} {score:.4f}")
    
    return results_summary


if __name__ == "__main__":
    # Test specific model or all models
    import sys
    
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
        latent_dim = int(sys.argv[2]) if len(sys.argv) > 2 else 32
        conditional = bool(int(sys.argv[3])) if len(sys.argv) > 3 else True
        
        evaluator = VAEEvaluator(model_path, latent_dim, conditional)
        evaluator.comprehensive_evaluation()
    else:
        evaluate_all_vae_models()