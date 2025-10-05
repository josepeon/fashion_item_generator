#!/usr/bin/env python3
"""
Simple CNN Performance Testing Script
Test Fashion-MNIST CNN model performance without external dependencies.
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime

# Import our models and data handler
from fashion_handler import FashionMNIST
from enhanced_fashion_cnn import EnhancedFashionNet


class SimpleCNNTester:
    """Simple CNN performance testing without external dependencies."""
    
    def __init__(self, model_path=None, device=None):
        # Setup device
        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = 'mps'
        else:
            self.device = 'cpu'
        
        print(f"CNN Tester initialized on {self.device}")
        
        # Fashion-MNIST class names
        self.class_names = [
            'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
            'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
        ]
        
        # Load data
        self.fashion_data = FashionMNIST(batch_size=64)
        
        # Load model
        self.model = None
        self.model_path = model_path or 'models/enhanced_fashion_cnn_200epochs.pth'
        self.load_model()
    
    def load_model(self):
        """Load the trained CNN model."""
        print(f"\n🔍 Loading CNN model from: {self.model_path}")
        
        try:
            self.model = EnhancedFashionNet().to(self.device)
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device, weights_only=True))
            self.model.eval()
            
            # Count parameters
            total_params = sum(p.numel() for p in self.model.parameters())
            
            print(f"✅ Model loaded successfully")
            print(f"   Total parameters: {total_params:,}")
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            print("   Trying to load basic FashionNet instead...")
            
            try:
                from fashion_cnn import FashionNet
                self.model = FashionNet().to(self.device)
                self.model.load_state_dict(torch.load(self.model_path, map_location=self.device, weights_only=True))
                self.model.eval()
                print(f"✅ Basic FashionNet loaded successfully")
            except Exception as e2:
                print(f"❌ Failed to load basic model too: {e2}")
                self.model = None
    
    def test_on_samples(self, num_batches=5):
        """Test model on sample batches and show results."""
        print(f"\n📊 TESTING ON SAMPLE BATCHES")
        print("=" * 50)
        
        if self.model is None:
            print("❌ No model loaded for testing")
            return None
        
        test_loader = self.fashion_data.get_test_loader()
        
        total_correct = 0
        total_samples = 0
        all_confidences = []
        class_correct = [0] * 10
        class_total = [0] * 10
        
        self.model.eval()
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(test_loader):
                if batch_idx >= num_batches:
                    break
                    
                images, labels = images.to(self.device), labels.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                probabilities = F.softmax(outputs, dim=1)
                
                # Get predictions and confidences
                confidences, predictions = torch.max(probabilities, 1)
                
                # Calculate batch accuracy
                correct = (predictions == labels).sum().item()
                total_correct += correct
                total_samples += labels.size(0)
                
                # Store confidences
                all_confidences.extend(confidences.cpu().numpy())
                
                # Per-class accuracy
                for i in range(labels.size(0)):
                    label = labels[i].item()
                    class_total[label] += 1
                    if predictions[i] == labels[i]:
                        class_correct[label] += 1
                
                print(f"   Batch {batch_idx+1}: {correct}/{labels.size(0)} correct ({100.0*correct/labels.size(0):.1f}%)")
        
        # Calculate overall metrics
        overall_accuracy = 100.0 * total_correct / total_samples
        avg_confidence = np.mean(all_confidences)
        
        print(f"\n🎯 SAMPLE TEST RESULTS:")
        print(f"   Overall Accuracy: {overall_accuracy:.2f}% ({total_correct}/{total_samples})")
        print(f"   Average Confidence: {avg_confidence:.3f}")
        
        # Per-class accuracy
        print(f"\n📋 PER-CLASS ACCURACY (Sample):")
        print("-" * 50)
        for i in range(10):
            if class_total[i] > 0:
                class_acc = 100.0 * class_correct[i] / class_total[i]
                print(f"   {self.class_names[i]:<15}: {class_acc:>5.1f}% ({class_correct[i]}/{class_total[i]})")
            else:
                print(f"   {self.class_names[i]:<15}: No samples in test batches")
        
        return {
            'accuracy': overall_accuracy,
            'avg_confidence': avg_confidence,
            'class_accuracies': [(class_correct[i]/class_total[i] if class_total[i] > 0 else 0) for i in range(10)]
        }
    
    def test_full_dataset(self):
        """Test on the complete test dataset."""
        print(f"\n📊 FULL DATASET EVALUATION")
        print("=" * 50)
        
        if self.model is None:
            print("❌ No model loaded for testing")
            return None
        
        test_loader = self.fashion_data.get_test_loader()
        
        total_correct = 0
        total_samples = 0
        all_confidences = []
        class_correct = [0] * 10
        class_total = [0] * 10
        
        print("   Processing test dataset...")
        
        self.model.eval()
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(test_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                probabilities = F.softmax(outputs, dim=1)
                
                # Get predictions and confidences
                confidences, predictions = torch.max(probabilities, 1)
                
                # Calculate batch accuracy
                correct = (predictions == labels).sum().item()
                total_correct += correct
                total_samples += labels.size(0)
                
                # Store confidences
                all_confidences.extend(confidences.cpu().numpy())
                
                # Per-class accuracy
                for i in range(labels.size(0)):
                    label = labels[i].item()
                    class_total[label] += 1
                    if predictions[i] == labels[i]:
                        class_correct[label] += 1
                
                if batch_idx % 20 == 0:
                    print(f"   Processed {total_samples:,} samples...")
        
        # Calculate overall metrics
        overall_accuracy = 100.0 * total_correct / total_samples
        avg_confidence = np.mean(all_confidences)
        
        print(f"\n🎯 FULL DATASET RESULTS:")
        print(f"   Overall Accuracy: {overall_accuracy:.2f}% ({total_correct:,}/{total_samples:,})")
        print(f"   Average Confidence: {avg_confidence:.3f}")
        
        # Per-class accuracy
        print(f"\n📋 DETAILED PER-CLASS PERFORMANCE:")
        print("-" * 60)
        print(f"{'Class':<15} {'Accuracy':<10} {'Correct':<8} {'Total':<8}")
        print("-" * 60)
        
        for i in range(10):
            class_acc = 100.0 * class_correct[i] / class_total[i]
            print(f"{self.class_names[i]:<15} {class_acc:>7.2f}% {class_correct[i]:>7} {class_total[i]:>7}")
        
        return {
            'accuracy': overall_accuracy,
            'avg_confidence': avg_confidence,
            'class_accuracies': [class_correct[i]/class_total[i] for i in range(10)],
            'class_correct': class_correct,
            'class_total': class_total
        }
    
    def visualize_predictions(self, num_samples=20):
        """Visualize model predictions on sample images."""
        print(f"\n🖼️  CREATING PREDICTION VISUALIZATIONS")
        print("=" * 50)
        
        if self.model is None:
            print("❌ No model loaded for visualization")
            return
        
        # Get test data
        test_loader = self.fashion_data.get_test_loader()
        data_iter = iter(test_loader)
        images, labels = next(data_iter)
        
        # Get predictions for these samples
        images_gpu = images.to(self.device)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(images_gpu)
            probabilities = F.softmax(outputs, dim=1)
            confidences, predictions = torch.max(probabilities, 1)
        
        # Move to CPU for visualization
        predictions = predictions.cpu()
        confidences = confidences.cpu()
        
        # Create visualization
        fig, axes = plt.subplots(4, 5, figsize=(15, 12))
        fig.suptitle('CNN Prediction Results - Sample Images', fontsize=16)
        
        correct_count = 0
        for i in range(min(num_samples, len(images))):
            row, col = i // 5, i % 5
            
            # Display image
            axes[row, col].imshow(images[i, 0].numpy(), cmap='gray')
            
            # Create title with prediction info
            true_label = self.class_names[labels[i]]
            pred_label = self.class_names[predictions[i]]
            confidence = confidences[i].item()
            
            # Color based on correctness
            is_correct = labels[i] == predictions[i]
            if is_correct:
                correct_count += 1
            color = 'green' if is_correct else 'red'
            
            title = f'True: {true_label[:8]}\nPred: {pred_label[:8]}\nConf: {confidence:.3f}'
            axes[row, col].set_title(title, fontsize=8, color=color)
            axes[row, col].axis('off')
        
        plt.tight_layout()
        
        # Save visualization
        os.makedirs('results', exist_ok=True)
        filename = f'results/cnn_predictions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        
        sample_accuracy = 100.0 * correct_count / min(num_samples, len(images))
        print(f"💾 Prediction visualization saved to: {filename}")
        print(f"   Sample accuracy: {sample_accuracy:.1f}% ({correct_count}/{min(num_samples, len(images))})")
    
    def analyze_performance(self, results):
        """Analyze performance and provide recommendations."""
        print(f"\n💡 PERFORMANCE ANALYSIS & RECOMMENDATIONS")
        print("=" * 50)
        
        if results is None:
            return
        
        accuracy = results['accuracy']
        avg_confidence = results['avg_confidence']
        class_accuracies = results['class_accuracies']
        
        print(f"🎯 OVERALL ASSESSMENT:")
        if accuracy >= 95.0:
            print(f"   ✅ EXCELLENT: {accuracy:.2f}% accuracy - Target achieved!")
        elif accuracy >= 90.0:
            print(f"   ✅ GOOD: {accuracy:.2f}% accuracy - Room for improvement")
        elif accuracy >= 85.0:
            print(f"   ⚠️  FAIR: {accuracy:.2f}% accuracy - Needs improvement")
        else:
            print(f"   ❌ POOR: {accuracy:.2f}% accuracy - Significant improvement needed")
        
        # Confidence analysis
        print(f"\n📊 CONFIDENCE ANALYSIS:")
        print(f"   Average confidence: {avg_confidence:.3f}")
        if avg_confidence >= 0.9:
            print(f"   ✅ High confidence in predictions")
        elif avg_confidence >= 0.8:
            print(f"   ✅ Good confidence level")
        else:
            print(f"   ⚠️  Low confidence - model may be uncertain")
        
        # Find problematic classes
        class_acc_with_names = list(zip(self.class_names, class_accuracies))
        class_acc_with_names.sort(key=lambda x: x[1])
        
        print(f"\n🎯 CLASSES NEEDING IMPROVEMENT:")
        for i, (class_name, class_acc) in enumerate(class_acc_with_names[:3]):
            print(f"   {i+1}. {class_name}: {class_acc*100:.1f}% accuracy")
        
        print(f"\n🏆 BEST PERFORMING CLASSES:")
        for i, (class_name, class_acc) in enumerate(class_acc_with_names[-3:]):
            print(f"   {3-i}. {class_name}: {class_acc*100:.1f}% accuracy")
        
        # Improvement suggestions
        print(f"\n🔧 IMPROVEMENT SUGGESTIONS:")
        if accuracy < 95.0:
            print(f"   • Add data augmentation (rotation, translation, noise)")
            print(f"   • Try deeper network architectures")
            print(f"   • Implement ensemble methods")
            print(f"   • Use focal loss for hard examples")
        
        if avg_confidence < 0.85:
            print(f"   • Add temperature scaling for better calibration")
            print(f"   • Use label smoothing during training")
            print(f"   • Implement dropout for uncertainty estimation")
        
        print(f"   • Focus training on classes: {', '.join([name for name, _ in class_acc_with_names[:3]])}")
        print(f"   • Consider class-specific data augmentation")
    
    def run_comprehensive_test(self):
        """Run complete CNN performance evaluation."""
        print("🧪 FASHION-MNIST CNN PERFORMANCE TEST")
        print("=" * 60)
        print(f"Device: {self.device}")
        print(f"Model: {self.model_path}")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        if self.model is None:
            print("❌ Cannot run tests without a loaded model")
            return None
        
        # Create results directory
        os.makedirs('results', exist_ok=True)
        
        # Run tests
        print(f"\n🔬 Running sample test...")
        sample_results = self.test_on_samples(num_batches=10)
        
        print(f"\n🔬 Running full dataset evaluation...")
        full_results = self.test_full_dataset()
        
        # Create visualizations
        self.visualize_predictions()
        
        # Analyze performance
        self.analyze_performance(full_results)
        
        print(f"\n🎉 CNN PERFORMANCE TESTING COMPLETED!")
        
        return full_results


def main():
    """Main function to run CNN performance testing."""
    print("🎯 FASHION-MNIST CNN PERFORMANCE TESTING")
    print("=" * 70)
    
    # Run comprehensive testing
    tester = SimpleCNNTester()
    results = tester.run_comprehensive_test()
    
    if results:
        print(f"\n📊 FINAL SUMMARY:")
        print(f"   Model Accuracy: {results['accuracy']:.2f}%")
        print(f"   Average Confidence: {results['avg_confidence']:.3f}")
        
        # Quick assessment
        if results['accuracy'] >= 95.0:
            grade = "A+"
            status = "EXCELLENT"
        elif results['accuracy'] >= 90.0:
            grade = "A"
            status = "GOOD"
        elif results['accuracy'] >= 85.0:
            grade = "B"
            status = "FAIR"
        else:
            grade = "C"
            status = "NEEDS WORK"
        
        print(f"   Grade: {grade} ({status})")


if __name__ == "__main__":
    main()