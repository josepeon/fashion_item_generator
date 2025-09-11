#!/usr/bin/env python3
"""
Conservative Quality Assessment for Enhanced VAE
===============================================

This script provides a realistic assessment of the Enhanced VAE quality
without bonus scoring or optimization tricks.
"""

import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.append(os.path.dirname(__file__))
from enhanced_vae import EnhancedVAE
from mnist_cnn import MNISTNet

def assess_realistic_quality():
    """Perform conservative quality assessment"""
    print("Conservative Enhanced VAE Quality Assessment")
    print("=" * 55)
    
    device = torch.device('cpu')
    
    # Load models
    print("Loading models...")
    model = EnhancedVAE(latent_dim=32, num_classes=10, conditional=True).to(device)
    evaluator = MNISTNet().to(device)
    
    # Load weights
    model.load_state_dict(torch.load('models/enhanced_vae_superior.pth', map_location=device))
    evaluator.load_state_dict(torch.load('models/best_mnist_cnn.pth', map_location=device))
    
    model.eval()
    evaluator.eval()
    print("Models loaded successfully")
    
    # Conservative assessment parameters
    samples_per_digit = 50  # More samples for better statistics
    total_samples = 0
    total_confidence = 0
    correct_classifications = 0
    high_confidence_count = 0
    
    # Per-digit statistics
    digit_stats = {}
    best_samples = {}
    
    print(f"Generating {samples_per_digit} samples per digit for assessment...")
    
    for digit in range(10):
        print(f"  Testing digit {digit}...")
        
        digit_confidences = []
        digit_correct = 0
        best_confidence = 0
        best_sample = None
        
        for sample_idx in range(samples_per_digit):
            with torch.no_grad():
                # Generate sample
                z = torch.randn(1, 32, device=device)
                labels = torch.tensor([digit], device=device)
                generated = model.decode(z, labels)
                
                # Evaluate with CNN (no bonus scoring)
                generated_4d = generated.view(1, 1, 28, 28)
                outputs = evaluator(generated_4d)
                probs = F.softmax(outputs, dim=1)
                
                # Get raw confidence (no bonuses)
                confidence = probs.max(dim=1)[0].item()
                predicted_digit = outputs.argmax(dim=1).item()
                
                # Track statistics
                digit_confidences.append(confidence)
                total_confidence += confidence
                total_samples += 1
                
                if predicted_digit == digit:
                    digit_correct += 1
                    correct_classifications += 1
                
                if confidence > 0.7:
                    high_confidence_count += 1
                
                # Track best sample for this digit
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_sample = generated.squeeze().cpu().view(28, 28)
        
        # Calculate per-digit statistics
        avg_confidence = np.mean(digit_confidences)
        accuracy = digit_correct / samples_per_digit
        high_conf_rate = sum(1 for c in digit_confidences if c > 0.7) / samples_per_digit
        
        digit_stats[digit] = {
            'avg_confidence': avg_confidence,
            'accuracy': accuracy,
            'high_confidence_rate': high_conf_rate,
            'max_confidence': max(digit_confidences)
        }
        
        best_samples[digit] = best_sample
        
        print(f"    → Confidence: {avg_confidence:.3f}, Accuracy: {accuracy:.1%}, High-conf: {high_conf_rate:.1%}")
    
    # Calculate overall statistics
    overall_confidence = total_confidence / total_samples
    overall_accuracy = correct_classifications / total_samples
    overall_high_conf_rate = high_confidence_count / total_samples
    
    # Calculate quality score (conservative formula)
    # Quality = (confidence × accuracy) with penalty for low accuracy
    quality_score = overall_confidence * overall_accuracy
    if overall_accuracy < 0.8:  # Penalty for poor conditional generation
        quality_score *= 0.8
    
    print(f"CONSERVATIVE QUALITY ASSESSMENT:")
    print(f"=" * 45)
    print(f"  Total samples generated: {total_samples}")
    print(f"  Average confidence: {overall_confidence:.3f} ({overall_confidence:.1%})")
    print(f"  Classification accuracy: {overall_accuracy:.3f} ({overall_accuracy:.1%})")
    print(f"  High confidence rate (>70%): {overall_high_conf_rate:.1%}")
    print(f"  Conservative Quality Score: {quality_score:.3f} ({quality_score:.1%})")
    
    # Quality assessment
    print(f"QUALITY EVALUATION:")
    if quality_score >= 0.98:
        print(f"  EXCELLENT: Target achieved (98%+)")
    elif quality_score >= 0.95:
        print(f"  VERY GOOD: Close to target ({quality_score:.1%} vs 98%)")
        print(f"     Gap to target: {0.98 - quality_score:.1%}")
    elif quality_score >= 0.90:
        print(f"  GOOD: Strong performance ({quality_score:.1%})")
        print(f"     Gap to target: {0.98 - quality_score:.1%}")
    else:
        print(f"  NEEDS IMPROVEMENT: {quality_score:.1%} vs 98% target")
        print(f"     Gap to target: {0.98 - quality_score:.1%}")
    
    # Per-digit breakdown
    print(f"PER-DIGIT PERFORMANCE:")
    print(f"  {'Digit':<5} {'Confidence':<10} {'Accuracy':<8} {'High-Conf':<9} {'Max':<6}")
    print(f"  {'-'*5:<5} {'-'*10:<10} {'-'*8:<8} {'-'*9:<9} {'-'*6:<6}")
    
    for digit in range(10):
        stats = digit_stats[digit]
        print(f"  {digit:<5} {stats['avg_confidence']:<10.3f} {stats['accuracy']:<8.1%} "
              f"{stats['high_confidence_rate']:<9.1%} {stats['max_confidence']:<6.3f}")
    
    # Create visualization
    create_quality_visualization(best_samples, digit_stats, quality_score)
    
    return quality_score, digit_stats

def create_quality_visualization(best_samples, digit_stats, quality_score):
    """Create visualization of results"""
    print(f"Creating visualization...")
    
    os.makedirs('results', exist_ok=True)
    
    fig = plt.figure(figsize=(16, 10))
    
    # Best samples visualization (top 2 rows)
    for digit in range(10):
        ax = plt.subplot(4, 5, digit + 1)
        if digit in best_samples and best_samples[digit] is not None:
            ax.imshow(best_samples[digit].numpy(), cmap='gray')
            stats = digit_stats[digit]
            ax.set_title(f'Digit {digit}\nConf: {stats["max_confidence"]:.3f}\n'
                        f'Acc: {stats["accuracy"]:.1%}', fontsize=10)
        ax.axis('off')
    
    # Statistics visualization (bottom 2 rows)
    
    # Confidence per digit
    ax1 = plt.subplot(4, 2, 5)
    digits = list(range(10))
    confidences = [digit_stats[d]['avg_confidence'] for d in digits]
    bars1 = ax1.bar(digits, confidences, color='skyblue', edgecolor='navy', alpha=0.7)
    ax1.axhline(y=0.98, color='red', linestyle='--', alpha=0.7, label='Target (98%)')
    ax1.set_xlabel('Digit')
    ax1.set_ylabel('Average Confidence')
    ax1.set_title('Confidence by Digit')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Accuracy per digit
    ax2 = plt.subplot(4, 2, 6)
    accuracies = [digit_stats[d]['accuracy'] for d in digits]
    bars2 = ax2.bar(digits, accuracies, color='lightgreen', edgecolor='darkgreen', alpha=0.7)
    ax2.axhline(y=0.98, color='red', linestyle='--', alpha=0.7, label='Target (98%)')
    ax2.set_xlabel('Digit')
    ax2.set_ylabel('Classification Accuracy')
    ax2.set_title('Accuracy by Digit')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # High confidence rates
    ax3 = plt.subplot(4, 2, 7)
    high_conf_rates = [digit_stats[d]['high_confidence_rate'] for d in digits]
    bars3 = ax3.bar(digits, high_conf_rates, color='orange', edgecolor='darkorange', alpha=0.7)
    ax3.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='Good (80%)')
    ax3.set_xlabel('Digit')
    ax3.set_ylabel('High Confidence Rate')
    ax3.set_title('High Confidence Rate by Digit')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Overall summary
    ax4 = plt.subplot(4, 2, 8)
    ax4.text(0.1, 0.8, f'Conservative Quality Assessment', fontsize=14, fontweight='bold')
    ax4.text(0.1, 0.6, f'Overall Quality Score: {quality_score:.1%}', fontsize=12)
    ax4.text(0.1, 0.5, f'Target: 98%', fontsize=12)
    
    if quality_score >= 0.98:
        ax4.text(0.1, 0.4, f'Status: TARGET ACHIEVED', fontsize=12, color='green')
    else:
        gap = 0.98 - quality_score
        ax4.text(0.1, 0.4, f'Gap to target: {gap:.1%}', fontsize=12, color='red')
        ax4.text(0.1, 0.3, f'Status: In Progress', fontsize=12, color='orange')
    
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis('off')
    
    plt.suptitle(f'Enhanced VAE - Conservative Quality Assessment\n'
                f'Quality Score: {quality_score:.1%} | Target: 98%', 
                fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results/conservative_quality_assessment.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"Visualization saved to results/conservative_quality_assessment.png")

def main():
    """Main assessment"""
    try:
        quality_score, digit_stats = assess_realistic_quality()
        
        print(f"FINAL ASSESSMENT:")
        print(f"  Conservative Quality: {quality_score:.1%}")
        print(f"  Target Achievement: {'YES' if quality_score >= 0.98 else 'IN PROGRESS'}")
        
        if quality_score < 0.98:
            gap = 0.98 - quality_score
            print(f"  Remaining gap: {gap:.1%}")
            print(f"NEXT STEPS:")
            print(f"  - Fine-tune model parameters")
            print(f"  - Implement advanced sampling techniques")
            print(f"  - Consider ensemble methods")
        
    except Exception as e:
        print(f"Error during assessment: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()