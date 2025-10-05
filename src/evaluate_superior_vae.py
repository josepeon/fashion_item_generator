"""
Superior VAE Quality Assessment Framework

Advanced evaluation system for measuring generation quality without
MPS-incompatible operations. Focuses on practical quality metrics
that work across all device types.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import time
from datetime import datetime
import os
import json

from fashion_handler import FashionMNIST
from superior_vae import SuperiorVAE


class QualityMetrics:
    """Advanced quality metrics for VAE evaluation."""
    
    @staticmethod
    def reconstruction_quality(original: torch.Tensor, reconstructed: torch.Tensor) -> Dict[str, float]:
        """Measure reconstruction quality with multiple metrics."""
        # MSE Loss
        mse = F.mse_loss(reconstructed, original).item()
        
        # MAE Loss  
        mae = F.l1_loss(reconstructed, original).item()
        
        # Pixel-wise correlation (more robust than SSIM for MPS)
        def correlation_score(x, y):
            x_flat = x.view(x.size(0), -1)
            y_flat = y.view(y.size(0), -1)
            
            # Center the data
            x_centered = x_flat - x_flat.mean(dim=1, keepdim=True)
            y_centered = y_flat - y_flat.mean(dim=1, keepdim=True)
            
            # Compute correlation coefficient
            numerator = (x_centered * y_centered).sum(dim=1)
            x_norm = torch.sqrt((x_centered ** 2).sum(dim=1))
            y_norm = torch.sqrt((y_centered ** 2).sum(dim=1))
            correlation = numerator / (x_norm * y_norm + 1e-8)
            
            return correlation.mean().item()
        
        correlation = correlation_score(original, reconstructed)
        
        # Combined quality score (higher is better)
        quality_score = correlation - 0.1 * mse - 0.05 * mae
        
        return {
            'mse': mse,
            'mae': mae,
            'correlation': correlation,
            'quality_score': quality_score
        }
    
    @staticmethod
    def generation_diversity(samples: torch.Tensor) -> Dict[str, float]:
        """Measure diversity of generated samples without using pdist."""
        batch_size = samples.size(0)
        samples_flat = samples.view(batch_size, -1)
        
        # Compute pairwise distances manually (MPS-compatible)
        diversity_scores = []
        
        for i in range(min(50, batch_size)):  # Sample subset for efficiency
            for j in range(i + 1, min(50, batch_size)):
                # L2 distance
                diff = samples_flat[i] - samples_flat[j]
                distance = torch.sqrt(torch.sum(diff ** 2)).item()
                diversity_scores.append(distance)
        
        if not diversity_scores:
            return {'diversity': 0.0, 'std_diversity': 0.0}
        
        diversity_mean = np.mean(diversity_scores)
        diversity_std = np.std(diversity_scores)
        
        return {
            'diversity': diversity_mean,
            'std_diversity': diversity_std
        }
    
    @staticmethod
    def latent_space_smoothness(model: SuperiorVAE, num_tests: int = 20, device: str = 'cpu') -> float:
        """Test latent space smoothness with interpolation."""
        model.eval()
        smoothness_scores = []
        
        with torch.no_grad():
            for _ in range(num_tests):
                # Random latent vectors
                z1 = torch.randn(1, model.latent_dim, device=device)
                z2 = torch.randn(1, model.latent_dim, device=device)
                
                # Random class
                class_label = torch.randint(0, 10, (1,), device=device)
                
                # Generate interpolation
                steps = 5
                interpolation_smooth = 0.0
                
                for i in range(steps - 1):
                    alpha1 = i / (steps - 1)
                    alpha2 = (i + 1) / (steps - 1)
                    
                    z_interp1 = (1 - alpha1) * z1 + alpha1 * z2
                    z_interp2 = (1 - alpha2) * z1 + alpha2 * z2
                    
                    img1 = model.decode(z_interp1, class_label)
                    img2 = model.decode(z_interp2, class_label)
                    
                    # Measure smoothness as inverse of difference
                    diff = F.mse_loss(img1, img2).item()
                    smoothness = 1.0 / (1.0 + diff * 10)  # Normalize
                    interpolation_smooth += smoothness
                
                smoothness_scores.append(interpolation_smooth / (steps - 1))
        
        return np.mean(smoothness_scores)


class SuperiorVAEEvaluator:
    """Comprehensive evaluator for Superior VAE models."""
    
    def __init__(self, model_path: str, device: str = None):
        self.device = device or ('mps' if torch.backends.mps.is_available() else 'cpu')
        self.model_path = model_path
        
        # Load model
        self.model = SuperiorVAE(latent_dim=64, conditional=True).to(self.device)
        
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(state_dict)
            print(f"✅ Loaded Superior VAE: {model_path}")
        else:
            print(f"⚠️  Model not found: {model_path}")
            return
        
        # Load test data
        self.fashion = FashionMNIST(batch_size=64)
        self.test_loader = self.fashion.get_test_loader()
        
        # Model info
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"🔬 Superior VAE Evaluator initialized")
        print(f"   Device: {self.device}")
        print(f"   Model parameters: {total_params:,}")
        print(f"   Conditional: {self.model.conditional}")
        print(f"   Latent dimensions: {self.model.latent_dim}")
    
    def comprehensive_evaluation(self, num_samples: int = 100) -> Dict:
        """Run comprehensive evaluation suite."""
        print(f"\n🔬 COMPREHENSIVE SUPERIOR VAE EVALUATION")
        print("=" * 60)
        print(f"Model: {self.model_path}")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        results = {
            'model_path': self.model_path,
            'timestamp': datetime.now().isoformat(),
            'device': self.device,
            'total_parameters': sum(p.numel() for p in self.model.parameters()),
        }
        
        # 1. Reconstruction Quality
        print(f"\n🔍 Testing reconstruction quality...")
        recon_results = self._test_reconstruction_quality()
        results.update(recon_results)
        
        # 2. Generation Quality
        print(f"\n🎨 Testing generation quality...")
        gen_results = self._test_generation_quality(num_samples)
        results.update(gen_results)
        
        # 3. Latent Space Properties
        print(f"\n🌌 Testing latent space properties...")
        latent_results = self._test_latent_space()
        results.update(latent_results)
        
        # 4. Conditional Generation
        print(f"\n🎯 Testing conditional generation...")
        cond_results = self._test_conditional_generation()
        results.update(cond_results)
        
        # 5. Overall Quality Score
        overall_score = self._compute_overall_score(results)
        results['overall_score'] = overall_score
        results['grade'] = self._assign_grade(overall_score)
        
        # Save results (convert numpy/tensor values to Python types)
        def convert_to_serializable(obj):
            if hasattr(obj, 'item'):  # tensor/numpy scalar
                return obj.item()
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(v) for v in obj]
            return obj
        
        serializable_results = convert_to_serializable(results)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = f'results/superior_vae_evaluation_{timestamp}.json'
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"\n🏆 EVALUATION COMPLETE")
        print(f"   📊 Results saved: {results_path}")
        print(f"   🎯 Overall Score: {overall_score:.4f}")
        print(f"   🏅 Grade: {results['grade']}")
        
        return results
    
    def _test_reconstruction_quality(self) -> Dict:
        """Test reconstruction quality on test set."""
        self.model.eval()
        total_metrics = {'mse': 0, 'mae': 0, 'correlation': 0, 'quality_score': 0}
        num_batches = 0
        
        reconstruction_samples = []
        
        with torch.no_grad():
            for i, (data, labels) in enumerate(self.test_loader):
                if i >= 10:  # Test on subset for efficiency
                    break
                
                data = data.view(data.size(0), -1).to(self.device)
                labels = labels.to(self.device)
                
                # Reconstruct
                recon, mu, logvar = self.model(data, labels)
                
                # Compute metrics
                metrics = QualityMetrics.reconstruction_quality(data, recon)
                
                for key, value in metrics.items():
                    total_metrics[key] += value
                
                # Save some samples for visualization
                if i == 0:
                    reconstruction_samples = {
                        'original': data[:8].cpu(),
                        'reconstructed': recon[:8].cpu(),
                        'labels': labels[:8].cpu()
                    }
                
                num_batches += 1
        
        # Average metrics
        avg_metrics = {f'recon_{k}': v / num_batches for k, v in total_metrics.items()}
        
        # Create reconstruction visualization
        if reconstruction_samples:
            self._save_reconstruction_demo(reconstruction_samples)
        
        print(f"   📊 Reconstruction MSE: {avg_metrics['recon_mse']:.4f}")
        print(f"   📊 Correlation: {avg_metrics['recon_correlation']:.4f}")
        print(f"   📊 Quality Score: {avg_metrics['recon_quality_score']:.4f}")
        
        return avg_metrics
    
    def _test_generation_quality(self, num_samples: int) -> Dict:
        """Test generation quality and diversity."""
        self.model.eval()
        
        # Generate samples for each class
        all_samples = []
        class_samples = {}
        
        with torch.no_grad():
            for class_idx in range(10):
                samples = self.model.generate_fashion_class(
                    class_idx, num_samples // 10, self.device, temperature=0.8
                )
                all_samples.append(samples)
                class_samples[class_idx] = samples
        
        # Combine all samples
        all_samples_tensor = torch.cat(all_samples, dim=0)
        
        # Compute diversity metrics
        diversity_metrics = QualityMetrics.generation_diversity(all_samples_tensor)
        
        # Visual quality assessment (simplified)
        visual_quality = self._assess_visual_quality(all_samples_tensor)
        
        gen_results = {
            'gen_diversity': diversity_metrics['diversity'],
            'gen_diversity_std': diversity_metrics['std_diversity'],
            'gen_visual_quality': visual_quality,
            'gen_quality_score': diversity_metrics['diversity'] * 0.5 + visual_quality * 0.5
        }
        
        # Save generation showcase
        self._save_generation_showcase(class_samples)
        
        print(f"   📊 Generation diversity: {gen_results['gen_diversity']:.4f}")
        print(f"   📊 Visual quality: {gen_results['gen_visual_quality']:.4f}")
        print(f"   📊 Generation score: {gen_results['gen_quality_score']:.4f}")
        
        return gen_results
    
    def _test_latent_space(self) -> Dict:
        """Test latent space properties."""
        # Smoothness test
        smoothness = QualityMetrics.latent_space_smoothness(self.model, 30, self.device)
        
        # Interpolation quality
        interp_quality = self._test_interpolation_quality()
        
        latent_results = {
            'latent_smoothness': smoothness,
            'interpolation_quality': interp_quality,
            'latent_score': (smoothness + interp_quality) * 0.5
        }
        
        print(f"   📊 Latent smoothness: {latent_results['latent_smoothness']:.4f}")
        print(f"   📊 Interpolation quality: {latent_results['interpolation_quality']:.4f}")
        
        return latent_results
    
    def _test_conditional_generation(self) -> Dict:
        """Test conditional generation capabilities.""" 
        self.model.eval()
        
        # Test class separation
        class_separation = self._measure_class_separation()
        
        # Test class consistency
        class_consistency = self._measure_class_consistency()
        
        cond_results = {
            'class_separation': class_separation,
            'class_consistency': class_consistency,
            'conditional_score': (class_separation + class_consistency) * 0.5
        }
        
        print(f"   📊 Class separation: {cond_results['class_separation']:.4f}")
        print(f"   📊 Class consistency: {cond_results['class_consistency']:.4f}")
        
        return cond_results
    
    def _compute_overall_score(self, results: Dict) -> float:
        """Compute overall quality score."""
        # Weighted combination of all metrics
        weights = {
            'recon_quality_score': 0.25,
            'gen_quality_score': 0.30,
            'latent_score': 0.20,
            'conditional_score': 0.25
        }
        
        score = 0.0
        total_weight = 0.0
        
        for metric, weight in weights.items():
            if metric in results:
                score += results[metric] * weight
                total_weight += weight
        
        return score / total_weight if total_weight > 0 else 0.0
    
    def _assign_grade(self, score: float) -> str:
        """Assign letter grade based on score."""
        if score >= 0.9:
            return "A+ - EXCEPTIONAL"
        elif score >= 0.8:
            return "A - EXCELLENT" 
        elif score >= 0.7:
            return "B - GOOD"
        elif score >= 0.6:
            return "C - SATISFACTORY"
        elif score >= 0.5:
            return "D - NEEDS IMPROVEMENT"
        else:
            return "F - POOR"
    
    def _assess_visual_quality(self, samples: torch.Tensor) -> float:
        """Assess visual quality of generated samples."""
        # Simple heuristics for visual quality
        samples_np = samples.cpu().numpy()
        
        # Check for diversity in pixel intensities
        intensity_std = np.std(samples_np)
        
        # Check for reasonable contrast
        contrast_score = np.mean(np.std(samples_np, axis=(2, 3)))
        
        # Normalize and combine
        quality = min(1.0, (intensity_std * 2 + contrast_score) / 3)
        
        return quality
    
    def _test_interpolation_quality(self) -> float:
        """Test quality of latent space interpolations."""
        return QualityMetrics.latent_space_smoothness(self.model, 20, self.device)
    
    def _measure_class_separation(self) -> float:
        """Measure how well different classes are separated."""
        # Generate samples from different classes and measure separation
        self.model.eval()
        
        with torch.no_grad():
            class_embeddings = []
            
            for class_idx in range(10):
                samples = self.model.generate_fashion_class(class_idx, 10, self.device)
                # Use mean of samples as class representation
                class_embed = samples.mean(dim=0).flatten()
                class_embeddings.append(class_embed)
            
            # Compute pairwise distances
            separations = []
            for i in range(10):
                for j in range(i + 1, 10):
                    diff = class_embeddings[i] - class_embeddings[j] 
                    separation = torch.sqrt(torch.sum(diff ** 2)).item()
                    separations.append(separation)
            
            return np.mean(separations) / 10  # Normalize
    
    def _measure_class_consistency(self) -> float:
        """Measure consistency within classes."""
        self.model.eval()
        
        consistencies = []
        
        with torch.no_grad():
            for class_idx in range(10):
                samples = self.model.generate_fashion_class(class_idx, 20, self.device)
                
                # Measure intra-class similarity
                samples_flat = samples.view(samples.size(0), -1)
                mean_sample = samples_flat.mean(dim=0)
                
                distances = []
                for i in range(samples_flat.size(0)):
                    diff = samples_flat[i] - mean_sample
                    distance = torch.sqrt(torch.sum(diff ** 2)).item()
                    distances.append(distance)
                
                # Consistency is inverse of average distance
                consistency = 1.0 / (1.0 + np.mean(distances))
                consistencies.append(consistency)
        
        return np.mean(consistencies)
    
    def _save_reconstruction_demo(self, samples: Dict):
        """Save reconstruction demonstration."""
        fig, axes = plt.subplots(2, 8, figsize=(16, 4))
        fig.suptitle('Superior VAE - Reconstruction Quality', fontsize=14, fontweight='bold')
        
        class_names = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
                      'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        
        for i in range(8):
            # Original
            axes[0, i].imshow(samples['original'][i].view(28, 28), cmap='gray')
            axes[0, i].set_title(f"Original\\n{class_names[samples['labels'][i]]}", fontsize=8)
            axes[0, i].axis('off')
            
            # Reconstructed 
            axes[1, i].imshow(samples['reconstructed'][i].view(28, 28), cmap='gray')
            axes[1, i].set_title("Reconstructed", fontsize=8)
            axes[1, i].axis('off')
        
        plt.tight_layout()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f'results/superior_vae_reconstruction_{timestamp}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   📊 Reconstruction demo saved: {save_path}")
    
    def _save_generation_showcase(self, class_samples: Dict):
        """Save generation quality showcase."""
        fig, axes = plt.subplots(10, 8, figsize=(16, 20))
        fig.suptitle('Superior VAE - Generation Quality Showcase', fontsize=16, fontweight='bold')
        
        class_names = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
                      'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        
        for class_idx in range(10):
            samples = class_samples[class_idx]
            for sample_idx in range(min(8, samples.size(0))):
                ax = axes[class_idx, sample_idx]
                img = samples[sample_idx].cpu().squeeze()
                ax.imshow(img, cmap='gray')
                ax.axis('off')
                
                if sample_idx == 0:
                    ax.set_ylabel(class_names[class_idx], rotation=0, ha='right', va='center')
        
        plt.tight_layout()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f'results/superior_vae_showcase_{timestamp}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   📊 Generation showcase saved: {save_path}")


def run_superior_evaluation():
    """Run superior VAE evaluation."""
    print("🏆 SUPERIOR VAE EVALUATION SUITE")
    print("=" * 50)
    
    # Check for model
    model_path = 'models/superior_vae_ultimate.pth'
    if not os.path.exists(model_path):
        print(f"⚠️  Superior VAE model not found: {model_path}")
        print("   Run the training first: python src/superior_vae.py")
        return None
    
    # Run evaluation
    evaluator = SuperiorVAEEvaluator(model_path)
    results = evaluator.comprehensive_evaluation(num_samples=200)
    
    return results


if __name__ == "__main__":
    results = run_superior_evaluation()