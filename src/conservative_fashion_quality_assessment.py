#!/usr/bin/env python3
"""
Conservative Quality Assessment for Enhanced Fashion VAE
======================================================

This script provides a realistic assessment of the Enhanced VAE quality
for Fashion-MNIST without bonus scoring or optimization tricks.
"""

import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.append(os.path.dirname(__file__))
from enhanced_vae import EnhancedVAE
from fashion_cnn import FashionNet
from fashion_handler import FashionMNIST

def assess_realistic_quality():
    """Perform conservative quality assessment"""
    print("Conservative Enhanced Fashion VAE Quality Assessment")
    print("=" * 60)
    
    device = torch.device('cpu')
    
    # Load models
    print("Loading models...")
    model = EnhancedVAE(latent_dim=32, num_classes=10, conditional=True).to(device)
    evaluator = FashionNet().to(device)
    
    # Load weights
    model.load_state_dict(torch.load('models/enhanced_vae_superior.pth', map_location=device))
    evaluator.load_state_dict(torch.load('models/best_fashion_cnn.pth', map_location=device))
    
    model.eval()
    evaluator.eval()
    print("Models loaded successfully")
    
    # Get fashion class names
    fashion = FashionMNIST()
    class_names = fashion.CLASS_NAMES
    
    # Conservative assessment parameters
    samples_per_class = 50  # More samples for better statistics
    total_samples = 0
    total_confidence = 0
    correct_classifications = 0
    high_confidence_count = 0
    
    # Per-class statistics
    class_stats = {}
    best_samples = {}
    
    print(f"Generating {samples_per_class} samples per fashion class for assessment...")
    
    for class_idx in range(10):
        print(f"  Assessing {class_names[class_idx]}...")
        
        # Generate samples for this class
        with torch.no_grad():
            z = torch.randn(samples_per_class, 32).to(device)
            labels = torch.full((samples_per_class,), class_idx, dtype=torch.long).to(device)
            
            # Generate samples
            samples = model.decode(z, labels)
            
            # Evaluate with CNN
            logits = evaluator(samples)
            probabilities = F.softmax(logits, dim=1)
            
            # Get predictions and confidence
            predicted_classes = torch.argmax(logits, dim=1)
            confidence_scores = probabilities[range(len(probabilities)), labels]
            
            # Calculate accuracy for this class
            is_correct = (predicted_classes == labels).float()
            class_accuracy = is_correct.mean().item()
            class_confidence = confidence_scores.mean().item()
            
            # Count high confidence samples
            high_conf_count = (confidence_scores > 0.95).sum().item()
            
            # Store class statistics
            class_stats[class_idx] = {
                'accuracy': class_accuracy,
                'confidence': class_confidence,
                'high_confidence_count': high_conf_count,
                'samples': samples_per_class
            }
            
            # Find best sample for visualization
            quality_scores = is_correct * confidence_scores
            best_idx = torch.argmax(quality_scores)
            best_samples[class_idx] = {
                'image': samples[best_idx].cpu(),
                'confidence': confidence_scores[best_idx].item(),
                'correct': is_correct[best_idx].item()
            }
            
            # Update totals
            total_samples += samples_per_class
            total_confidence += confidence_scores.sum().item()
            correct_classifications += is_correct.sum().item()
            high_confidence_count += high_conf_count
    
    return class_stats, best_samples, {
        'total_samples': total_samples,
        'total_confidence': total_confidence,
        'correct_classifications': correct_classifications,
        'high_confidence_count': high_confidence_count
    }


def create_conservative_visualization(class_stats, best_samples):
    """Create visualization of assessment results"""
    fashion = FashionMNIST()
    class_names = fashion.CLASS_NAMES
    
    fig, axes = plt.subplots(2, 5, figsize=(15, 8))
    fig.suptitle('Conservative Quality Assessment - Best Sample per Fashion Class', 
                 fontsize=14, fontweight='bold')
    
    for class_idx in range(10):
        row, col = class_idx // 5, class_idx % 5
        ax = axes[row, col]
        
        # Display best sample
        sample_data = best_samples[class_idx]
        image = sample_data['image'].squeeze().numpy()
        ax.imshow(image, cmap='gray')
        ax.axis('off')
        
        # Add title with stats
        stats = class_stats[class_idx]
        confidence = sample_data['confidence']
        is_correct = sample_data['correct']
        
        color = 'green' if is_correct and confidence > 0.9 else 'orange' if is_correct else 'red'
        
        title = f"{class_names[class_idx]}\n"
        title += f"Acc: {stats['accuracy']:.1%}, Conf: {confidence:.1%}"
        ax.set_title(title, color=color, fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    # Save visualization
    save_path = 'results/conservative_fashion_quality_assessment.png'
    os.makedirs('results', exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Assessment visualization saved to {save_path}")
    
    return save_path


def report_conservative_results(class_stats, totals):
    """Generate detailed quality report"""
    print("\n" + "="*70)
    print("CONSERVATIVE QUALITY ASSESSMENT RESULTS")
    print("="*70)
    
    fashion = FashionMNIST()
    class_names = fashion.CLASS_NAMES
    
    # Calculate overall metrics
    overall_accuracy = totals['correct_classifications'] / totals['total_samples']
    overall_confidence = totals['total_confidence'] / totals['total_samples']
    high_confidence_rate = totals['high_confidence_count'] / totals['total_samples']
    
    # Conservative quality score (accuracy * confidence)
    conservative_quality = overall_accuracy * overall_confidence
    
    print(f"Total Samples Generated: {totals['total_samples']:,}")
    print(f"Overall Classification Accuracy: {overall_accuracy:.1%}")
    print(f"Average Confidence Score: {overall_confidence:.1%}")
    print(f"High Confidence Rate (>95%): {high_confidence_rate:.1%}")
    print(f"Conservative Quality Score: {conservative_quality:.1%}")
    print()
    
    # Per-class breakdown
    print("Per-Class Performance:")
    print("-" * 50)
    print(f"{'Class':>12} {'Accuracy':>10} {'Confidence':>12} {'Quality':>10}")
    print("-" * 50)
    
    for class_idx in range(10):
        stats = class_stats[class_idx]
        class_quality = stats['accuracy'] * stats['confidence']
        print(f"{class_names[class_idx]:>12} {stats['accuracy']:>9.1%} "
              f"{stats['confidence']:>11.1%} {class_quality:>9.1%}")
    
    print("-" * 50)
    print(f"{'Average':>12} {overall_accuracy:>9.1%} "
          f"{overall_confidence:>11.1%} {conservative_quality:>9.1%}")
    
    print()
    print("ASSESSMENT CONCLUSION:")
    print("-" * 25)
    
    if conservative_quality >= 0.75:
        assessment = "EXCELLENT"
        icon = "🌟"
    elif conservative_quality >= 0.60:
        assessment = "GOOD"
        icon = "✅"
    elif conservative_quality >= 0.45:
        assessment = "MODERATE"
        icon = "🔶"
    else:
        assessment = "NEEDS IMPROVEMENT"
        icon = "❌"
    
    print(f"{icon} Conservative Quality Rating: {assessment}")
    print(f"📊 Quality Score: {conservative_quality:.1%}")
    
    if conservative_quality >= 0.60:
        print("🎯 Model shows good generation capability for Fashion-MNIST")
    else:
        print("🔧 Model may benefit from additional training or architecture improvements")
    
    return {
        'overall_accuracy': overall_accuracy,
        'overall_confidence': overall_confidence,
        'conservative_quality': conservative_quality,
        'high_confidence_rate': high_confidence_rate,
        'assessment': assessment
    }


def main():
    """Main assessment function"""
    try:
        # Perform assessment
        class_stats, best_samples, totals = assess_realistic_quality()
        
        # Create visualization
        save_path = create_conservative_visualization(class_stats, best_samples)
        
        # Generate report
        results = report_conservative_results(class_stats, totals)
        
        print(f"\n✅ Conservative assessment completed!")
        print(f"📊 Results visualization: {save_path}")
        print(f"🎯 Conservative Quality Score: {results['conservative_quality']:.1%}")
        
        return results
        
    except FileNotFoundError as e:
        print(f"❌ Error: Model file not found - {e}")
        print("Please ensure the model files exist:")
        print("  - models/enhanced_vae_superior.pth")
        print("  - models/best_fashion_cnn.pth")
        return None
    except Exception as e:
        print(f"❌ Error during assessment: {e}")
        return None


if __name__ == "__main__":
    main()