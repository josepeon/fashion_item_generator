"""
Training Monitor for Superior VAE

Real-time monitoring and status tracking for the intensive training session.
"""

import os
import time
import matplotlib.pyplot as plt
import json
from datetime import datetime

def monitor_training_progress():
    """Monitor the training progress and display status."""
    print("🔍 SUPERIOR VAE TRAINING MONITOR")
    print("=" * 50)
    
    # Check for model checkpoints
    models_dir = "models"
    if os.path.exists(models_dir):
        checkpoint_files = [f for f in os.listdir(models_dir) if "superior_vae" in f]
        print(f"📁 Found {len(checkpoint_files)} Superior VAE files:")
        for f in sorted(checkpoint_files):
            filepath = os.path.join(models_dir, f)
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            mod_time = datetime.fromtimestamp(os.path.getmtime(filepath))
            print(f"   📦 {f} ({size_mb:.1f}MB, {mod_time.strftime('%H:%M:%S')})")
    
    # Check for results
    results_dir = "results"
    if os.path.exists(results_dir):
        result_files = [f for f in os.listdir(results_dir) if "superior_vae" in f]
        print(f"\\n📊 Found {len(result_files)} Superior VAE results:")
        for f in sorted(result_files)[-5:]:  # Show latest 5
            filepath = os.path.join(results_dir, f)
            mod_time = datetime.fromtimestamp(os.path.getmtime(filepath))
            print(f"   🖼️  {f} ({mod_time.strftime('%H:%M:%S')})")
    
    # Training status
    ultimate_model = "models/superior_vae_ultimate.pth"
    if os.path.exists(ultimate_model):
        size_mb = os.path.getsize(ultimate_model) / (1024 * 1024)
        mod_time = datetime.fromtimestamp(os.path.getmtime(ultimate_model))
        print(f"\\n🎯 ULTIMATE MODEL STATUS:")
        print(f"   ✅ Found: {ultimate_model}")
        print(f"   📦 Size: {size_mb:.1f}MB")
        print(f"   🕒 Modified: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   🚀 Status: TRAINING COMPLETE!")
        return True
    else:
        print(f"\\n🔄 TRAINING STATUS:")
        print(f"   🚀 Training in progress...")
        print(f"   ⏳ Ultimate model not yet available")
        print(f"   📊 Monitor will check again...")
        return False

def create_training_summary():
    """Create a summary of the training improvements."""
    print("\\n📈 SUPERIOR VAE IMPROVEMENTS SUMMARY")
    print("=" * 60)
    
    improvements = [
        ("🏗️ Architecture", "33.5M parameters (10× larger than Enhanced VAE)"),
        ("🧠 Attention", "Multi-head self-attention mechanisms"),
        ("🔗 Residual Blocks", "Advanced residual connections with GELU"),
        ("📏 Latent Space", "64 dimensions (2× larger)"),
        ("🎯 Conditioning", "Enhanced class embedding with projection"),
        ("⚡ Training", "OneCycle LR with 500 epochs intensive training"),
        ("🎨 Quality", "Temperature-controlled generation"),
        ("🌊 β-VAE", "Progressive β: 0.1 → 2.0 for better disentanglement"),
        ("🔄 Interpolation", "Spherical interpolation (SLERP)"),
        ("📊 Evaluation", "MPS-compatible quality metrics")
    ]
    
    for feature, description in improvements:
        print(f"   {feature:<15} {description}")
    
    print(f"\\n🎊 EXPECTED QUALITY IMPROVEMENTS:")
    print(f"   • 🎨 Significantly higher generation quality")
    print(f"   • 🎯 Better class-conditional control") 
    print(f"   • 🌈 Smoother latent space interpolations")
    print(f"   • 🧩 Improved disentanglement of features")
    print(f"   • ✨ More realistic and diverse samples")

if __name__ == "__main__":
    # Monitor training
    training_complete = monitor_training_progress()
    
    # Show improvements
    create_training_summary()
    
    if not training_complete:
        print(f"\\n⏳ Training is still running...")
        print(f"   Use: python src/training_monitor.py")
        print(f"   To check progress again")
    else:
        print(f"\\n🏆 TRAINING COMPLETE!")
        print(f"   Ready to evaluate: python src/evaluate_superior_vae.py")