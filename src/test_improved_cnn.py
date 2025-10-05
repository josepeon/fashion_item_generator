#!/usr/bin/env python3
"""
Test the improved CNN model performance.
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import json

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__)))

from fashion_handler import FashionMNIST
from analyze_cnn_improvements import ImprovedFashionCNN


class ImprovedCNNTester:
    """Test framework for the improved CNN model."""
    
    def __init__(self, model_path='models/improved_fashion_cnn.pth'):
        self.model_path = model_path
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        self.class_names = [
            'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
            'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
        ]
        
        # Load model
        self.model = ImprovedFashionCNN().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        # Load test data
        fashion_data = FashionMNIST(batch_size=128)
        self.test_loader = fashion_data.get_test_loader()
    
    def test_model(self):
        """Test the model and return detailed results."""
        print("🧪 TESTING IMPROVED CNN MODEL")
        print("=" * 60)
        print(f"Model: {self.model_path}")
        print(f"Device: {self.device}")
        
        correct = 0
        total = 0
        class_correct = np.zeros(10)
        class_total = np.zeros(10)
        class_predictions = [[] for _ in range(10)]
        all_predictions = []
        all_targets = []
        confidence_scores = []
        
        with torch.no_grad():
            for data, targets in self.test_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = self.model(data)
                
                # Get predictions and confidence
                probabilities = F.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
                
                # Store for detailed analysis
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                confidence_scores.extend(confidence.cpu().numpy())
                
                # Per-class accuracy
                for i in range(targets.size(0)):
                    label = targets[i].item()
                    pred = predicted[i].item()
                    class_total[label] += 1
                    if label == pred:
                        class_correct[label] += 1
                    class_predictions[label].append(pred == label)
        
        # Calculate metrics
        overall_accuracy = 100. * correct / total
        avg_confidence = np.mean(confidence_scores) * 100
        
        # Per-class accuracies
        class_accuracies = []
        for i in range(10):
            if class_total[i] > 0:
                acc = 100. * class_correct[i] / class_total[i]
                class_accuracies.append(acc)
            else:
                class_accuracies.append(0.0)
        
        return {
            'overall_accuracy': overall_accuracy,
            'average_confidence': avg_confidence,
            'class_accuracies': class_accuracies,
            'class_names': self.class_names,
            'predictions': all_predictions,
            'targets': all_targets,
            'confidence_scores': confidence_scores,
            'total_samples': total
        }
    
    def print_results(self, results):
        """Print detailed test results."""
        print(f"\n📊 IMPROVED CNN TEST RESULTS")
        print("=" * 60)
        print(f"Overall Accuracy: {results['overall_accuracy']:.2f}%")
        print(f"Average Confidence: {results['average_confidence']:.1f}%")
        print(f"Total Test Samples: {results['total_samples']:,}")
        
        # Performance grade
        acc = results['overall_accuracy']
        if acc >= 98.0:
            grade = "A+ - EXCELLENT"
        elif acc >= 96.0:
            grade = "A - VERY GOOD"  
        elif acc >= 94.0:
            grade = "B+ - GOOD"
        elif acc >= 90.0:
            grade = "B - ACCEPTABLE"
        else:
            grade = "C - NEEDS IMPROVEMENT"
        
        print(f"Performance Grade: {grade}")
        
        print(f"\n📋 PER-CLASS PERFORMANCE:")
        print("-" * 60)
        print(f"{'Class':<15} {'Accuracy':<10} {'Status'}")
        print("-" * 60)
        
        for i, (name, acc) in enumerate(zip(results['class_names'], results['class_accuracies'])):
            status = "🟢 EXCELLENT" if acc >= 95 else "🟡 GOOD" if acc >= 90 else "🔴 IMPROVE"
            print(f"{name:<15} {acc:>7.1f}%    {status}")
        
        # Find best and worst performing classes
        best_idx = np.argmax(results['class_accuracies'])
        worst_idx = np.argmin(results['class_accuracies'])
        
        print(f"\n🏆 Best: {results['class_names'][best_idx]} ({results['class_accuracies'][best_idx]:.1f}%)")
        print(f"⚠️  Worst: {results['class_names'][worst_idx]} ({results['class_accuracies'][worst_idx]:.1f}%)")
        
        return results
    
    def compare_with_baseline(self, results, baseline_accuracy=94.50):
        """Compare with baseline performance."""
        print(f"\n🔄 COMPARISON WITH BASELINE")
        print("=" * 60)
        print(f"Baseline Accuracy: {baseline_accuracy:.2f}%")
        print(f"Improved Accuracy: {results['overall_accuracy']:.2f}%")
        
        improvement = results['overall_accuracy'] - baseline_accuracy
        print(f"Improvement: {improvement:+.2f} percentage points")
        
        if improvement > 2.0:
            print("🎉 SIGNIFICANT IMPROVEMENT!")
        elif improvement > 0.5:
            print("✅ MEANINGFUL IMPROVEMENT")
        elif improvement > 0:
            print("📈 MINOR IMPROVEMENT")
        else:
            print("❌ NO IMPROVEMENT")
        
        # Compare class-specific improvements (if we have baseline data)
        baseline_class_accuracies = [90.4, 98.7, 92.3, 95.1, 94.0, 99.4, 80.9, 97.0, 99.2, 98.0]  # From previous test
        
        print(f"\n📈 CLASS-SPECIFIC IMPROVEMENTS:")
        print("-" * 70)
        print(f"{'Class':<15} {'Baseline':<10} {'Improved':<10} {'Change'}")
        print("-" * 70)
        
        for i, (name, baseline, improved) in enumerate(zip(results['class_names'], baseline_class_accuracies, results['class_accuracies'])):
            change = improved - baseline
            status = "🎯" if change > 5 else "✅" if change > 1 else "📊" if change > -1 else "⚠️"
            print(f"{name:<15} {baseline:>7.1f}%   {improved:>7.1f}%   {change:>+6.1f}% {status}")
    
    def run_comprehensive_test(self):
        """Run the complete test suite."""
        print("🚀 COMPREHENSIVE IMPROVED CNN TESTING")
        print("=" * 70)
        
        results = self.test_model()
        self.print_results(results)
        self.compare_with_baseline(results)
        
        # Generate simple visualization
        self.create_results_visualization(results)
        
        return results
    
    def create_results_visualization(self, results):
        """Create a simple visualization of results."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Overall accuracy comparison
        categories = ['Baseline\n(94.50%)', f'Improved\n({results["overall_accuracy"]:.2f}%)']
        accuracies = [94.50, results['overall_accuracy']]
        colors = ['#ff7f7f', '#7fbf7f']
        
        bars1 = ax1.bar(categories, accuracies, color=colors, alpha=0.8)
        ax1.set_ylabel('Accuracy (%)')
        ax1.set_title('Overall Accuracy Comparison')
        ax1.set_ylim(90, 100)
        
        # Add value labels on bars
        for bar, acc in zip(bars1, accuracies):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold')
        
        # Per-class accuracy
        class_names_short = [name.replace('/', '/\n') for name in results['class_names']]
        bars2 = ax2.bar(range(10), results['class_accuracies'], color='steelblue', alpha=0.8)
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Per-Class Accuracy')
        ax2.set_xticks(range(10))
        ax2.set_xticklabels(class_names_short, rotation=45, ha='right')
        ax2.set_ylim(70, 100)
        
        # Add value labels on bars
        for i, (bar, acc) in enumerate(zip(bars2, results['class_accuracies'])):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{acc:.1f}%', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        
        # Save the plot
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'results/improved_cnn_results_{timestamp}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"\n📊 Results visualization saved: {filename}")
        
        plt.show()


def main():
    """Main function to test the improved CNN."""
    # Test the latest checkpoint
    model_paths = [
        'models/improved_fashion_cnn.pth',  # Final model
        'models/improved_cnn_epoch_75.pth',  # 75-epoch checkpoint
        'models/improved_cnn_epoch_50.pth',  # 50-epoch checkpoint
    ]
    
    for model_path in model_paths:
        try:
            print(f"\n{'='*80}")
            print(f"TESTING: {model_path}")
            print(f"{'='*80}")
            
            tester = ImprovedCNNTester(model_path)
            results = tester.run_comprehensive_test()
            
            # Save detailed results
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            results_file = f'results/improved_cnn_detailed_{timestamp}.json'
            
            # Convert numpy arrays to lists for JSON serialization
            json_results = {
                'model_path': model_path,
                'timestamp': timestamp,
                'overall_accuracy': float(results['overall_accuracy']),
                'average_confidence': float(results['average_confidence']),
                'class_accuracies': [float(x) for x in results['class_accuracies']],
                'class_names': results['class_names'],
                'total_samples': int(results['total_samples'])
            }
            
            with open(results_file, 'w') as f:
                json.dump(json_results, f, indent=2)
            
            print(f"📁 Detailed results saved: {results_file}")
            
            break  # Test only the first available model
            
        except FileNotFoundError:
            print(f"⚠️  Model not found: {model_path}")
            continue
        except Exception as e:
            print(f"❌ Error testing {model_path}: {e}")
            continue


if __name__ == "__main__":
    main()