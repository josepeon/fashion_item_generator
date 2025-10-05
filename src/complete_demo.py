#!/usr/bin/env python3
"""
Complete Fashion-MNIST Demo
Tests both prediction (CNN) and generation (VAE) in one place.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from fashion_handler import FashionMNIST
from fashion_cnn import FashionNet
from simple_generator import SimpleVAE


def test_prediction():
    """Test the CNN prediction model."""
    print("🔍 TESTING PREDICTION (CNN)")
    print("=" * 40)
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
    # Load CNN model
    model = FashionNet().to(device)
    model_path = 'models/best_fashion_cnn_100epochs.pth'
    if not os.path.exists(model_path):
        model_path = '../models/best_fashion_cnn_100epochs.pth'
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Get test data
    fashion = FashionMNIST(batch_size=8)
    test_loader = fashion.get_test_loader()
    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    images, labels = images.to(device), labels.to(device)
    
    # Make predictions
    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
    
    # Fashion-MNIST class names
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    # Visualize predictions
    fig, axes = plt.subplots(2, 4, figsize=(15, 8))
    fig.suptitle('Fashion-MNIST Predictions (CNN)', fontsize=16)
    
    for i in range(8):
        row = i // 4
        col = i % 4
        
        axes[row, col].imshow(images[i, 0].cpu().numpy(), cmap='gray')
        axes[row, col].axis('off')
        
        true_label = class_names[labels[i]]
        pred_label = class_names[predicted[i]]
        color = 'green' if labels[i] == predicted[i] else 'red'
        
        axes[row, col].set_title(f'True: {true_label}\nPred: {pred_label}', 
                                color=color, fontsize=10)
    
    plt.tight_layout()
    result_path = 'results/complete_demo_predictions.png'
    if not os.path.exists('results'):
        result_path = '../results/complete_demo_predictions.png'
    plt.savefig(result_path, dpi=150, bbox_inches='tight')
    print("💾 Predictions saved to: results/complete_demo_predictions.png")
    
    # Calculate accuracy
    accuracy = (predicted == labels).float().mean()
    print(f"📊 Batch Accuracy: {accuracy:.2%}")
    
    return accuracy


def test_generation():
    """Test the VAE generation model."""
    print("\n🎨 TESTING GENERATION (VAE)")
    print("=" * 40)
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
    # Load VAE model
    model = SimpleVAE(latent_dim=20).to(device)
    model_path = 'models/simple_vae.pth'
    if not os.path.exists(model_path):
        model_path = '../models/simple_vae.pth'
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Generate new fashion items
    with torch.no_grad():
        samples = model.generate(num_samples=8, device=device)
    
    # Visualize generated items
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    fig.suptitle('Generated Fashion Items (VAE)', fontsize=16)
    
    for i in range(8):
        row = i // 4
        col = i % 4
        
        axes[row, col].imshow(samples[i].cpu().numpy(), cmap='gray')
        axes[row, col].axis('off')
        axes[row, col].set_title(f'Generated #{i+1}')
    
    plt.tight_layout()
    result_path = 'results/complete_demo_generation.png'
    if not os.path.exists('results'):
        result_path = '../results/complete_demo_generation.png'
    plt.savefig(result_path, dpi=150, bbox_inches='tight')
    print("💾 Generated items saved to: results/complete_demo_generation.png")
    
    return True


def main():
    """Complete demonstration of both prediction and generation."""
    print("🚀 COMPLETE FASHION-MNIST DEMO")
    print("Testing both PREDICTION and GENERATION")
    print("=" * 50)
    
    # Test prediction
    try:
        accuracy = test_prediction()
        print("✅ Prediction working!")
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        return
    
    # Test generation
    try:
        test_generation()
        print("✅ Generation working!")
    except Exception as e:
        print(f"❌ Generation failed: {e}")
        return
    
    print("\n🎯 PROJECT STATUS:")
    print("=" * 30)
    print(f"✅ CNN Prediction: {accuracy:.1%} accuracy")
    print("✅ VAE Generation: Working")
    print("📁 Models saved in: models/")
    print("🖼️  Results saved in: results/")
    print("\n🎉 Both prediction AND generation are now working!")


if __name__ == "__main__":
    main()