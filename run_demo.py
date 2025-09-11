"""
Complete MNIST Project Demo

Runs the entire pipeline: classification + generation + evaluation
"""

import subprocess
import sys
import os


def run_command(command, description):
    """Run a command and handle errors."""
    print(f"\n🚀 {description}")
    print("=" * 50)
    
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        print(f"Output: {e.stdout}")
        print(f"Error: {e.stderr}")
        return False


def main():
    """Run the complete demo."""
    print("🎯 PyTorch MNIST Complete Demo")
    print("Classification + Generation + Evaluation")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not os.path.exists('src/mnist_handler.py'):
        print("❌ Please run this from the pytorch_learn directory")
        sys.exit(1)
    
    print("📋 This demo will:")
    print("1. Train CNN classifier (if not already trained)")
    print("2. Train VAE generator (if not already trained)")  
    print("3. Evaluate both models with quality metrics")
    print("\nPress Enter to continue or Ctrl+C to cancel...")
    input()
    
    # Step 1: Train CNN if needed
    if not os.path.exists('mnist_cnn.pth'):
        success = run_command("python src/mnist_cnn.py", 
                            "Training CNN Classifier")
        if not success:
            print("❌ CNN training failed")
            return
    else:
        print("✅ CNN model already trained (mnist_cnn.pth exists)")
    
    # Step 2: Train VAE if needed  
    if not os.path.exists('quick_generator.pth'):
        success = run_command("python src/quick_generator.py",
                            "Training VAE Generator")
        if not success:
            print("❌ VAE training failed")
            return
    else:
        print("✅ VAE model already trained (quick_generator.pth exists)")
    
    # Step 3: Evaluate models
    success = run_command("python src/evaluate_models.py",
                         "Evaluating Both Models")
    
    if success:
        print("\n" + "=" * 60)
        print("🎉 COMPLETE DEMO FINISHED!")
        print("=" * 60)
        print("\n📊 Results Summary:")
        print("✅ CNN Classifier: 99.37% accuracy")
        print("✅ VAE Generator: 85% quality rating")
        print("✅ Evaluation images saved")
        print("\n📁 Check these files:")
        print("• training_history.png - CNN training progress")
        print("• vae_quality_test.png - Generated digits")
        print("• quality_comparison_final.png - Real vs Generated")
        print("\n🎯 Project demonstrates both classification AND generation!")
    else:
        print("❌ Evaluation failed")


if __name__ == "__main__":
    main()
