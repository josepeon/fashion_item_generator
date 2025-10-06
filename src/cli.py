"""
Fashion Item Generator CLI

Command-line interface for the Fashion Item Generator featuring
the Superior VAE with A+ EXCEPTIONAL performance.
"""

import argparse
import sys
import os
from typing import Optional
import torch

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from superior_vae import SuperiorVAE, run_superior_training
from evaluate_superior_vae import run_superior_evaluation
from test_superior_generation import test_superior_generation


def generate_samples(args):
    """Generate fashion samples using Superior VAE."""
    print("🎨 GENERATING FASHION SAMPLES")
    print("=" * 40)
    
    # Setup device
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"🖥️  Device: {device}")
    
    # Check model
    if not os.path.exists(args.model_path):
        print(f"❌ Model not found: {args.model_path}")
        print("   Run: fashion-generator train")
        return 1
    
    # Load model
    print(f"🚀 Loading Superior VAE...")
    model = SuperiorVAE(latent_dim=64, conditional=True).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=False))
    model.eval()
    
    print(f"✅ Model loaded successfully!")
    
    # Generate samples
    if args.fashion_class is not None:
        # Class-conditional generation
        class_names = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
                      'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        
        if args.fashion_class < 0 or args.fashion_class >= len(class_names):
            print(f"❌ Invalid class {args.fashion_class}. Use 0-9.")
            return 1
        
        print(f"🎯 Generating {args.num_samples} {class_names[args.fashion_class]} samples...")
        samples = model.generate_fashion_class(
            args.fashion_class, args.num_samples, device, temperature=args.temperature
        )
        output_name = f"generated_{class_names[args.fashion_class].lower()}_{args.num_samples}"
    else:
        # Random generation
        print(f"🎲 Generating {args.num_samples} random fashion samples...")
        samples = model.generate(args.num_samples, device=device, temperature=args.temperature)
        output_name = f"generated_random_{args.num_samples}"
    
    # Save samples
    import matplotlib.pyplot as plt
    from datetime import datetime
    
    # Create grid
    grid_size = int(args.num_samples ** 0.5)
    if grid_size * grid_size < args.num_samples:
        grid_size += 1
    
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size*2, grid_size*2))
    if grid_size == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for i in range(args.num_samples):
        img = samples[i].cpu().squeeze().numpy()
        axes[i].imshow(img, cmap='gray', vmin=-1, vmax=1)
        axes[i].axis('off')
    
    # Hide extra subplots
    for i in range(args.num_samples, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"results/{output_name}_{timestamp}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Samples saved: {output_path}")
    return 0


def train_model(args):
    """Train the Superior VAE model."""
    print("🔥 TRAINING SUPERIOR VAE")
    print("=" * 40)
    
    try:
        trainer, history = run_superior_training()
        print("✅ Training completed successfully!")
        return 0
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return 1


def evaluate_model(args):
    """Evaluate the Superior VAE model."""
    print("📊 EVALUATING SUPERIOR VAE")
    print("=" * 40)
    
    try:
        results = run_superior_evaluation()
        if results:
            print("✅ Evaluation completed successfully!")
            print(f"🏆 Grade: {results.get('grade', 'N/A')}")
            print(f"📈 Overall Score: {results.get('overall_score', 'N/A'):.4f}")
            return 0
        else:
            print("❌ Evaluation failed")
            return 1
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        return 1


def test_generation(args):
    """Test generation capabilities."""
    print("🧪 TESTING GENERATION CAPABILITIES")
    print("=" * 40)
    
    try:
        results = test_superior_generation()
        if results:
            print("✅ Generation test completed!")
            print(f"🎨 Diversity Score: {results.get('diversity_score', 'N/A'):.3f}")
            return 0
        else:
            print("❌ Generation test failed")
            return 1
    except Exception as e:
        print(f"❌ Generation test failed: {e}")
        return 1


def validate_environment(args):
    """Validate the environment setup."""
    print("✅ VALIDATING ENVIRONMENT")
    print("=" * 40)
    
    try:
        # Import validation script
        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        from validate_environment import validate_environment
        
        success = validate_environment()
        return 0 if success else 1
    except Exception as e:
        print(f"❌ Environment validation failed: {e}")
        return 1


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Fashion Item Generator - Superior VAE CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train the Superior VAE model
  fashion-generator train
  
  # Generate 16 random fashion items
  fashion-generator generate --num-samples 16
  
  # Generate 8 sneakers (class 7)
  fashion-generator generate --fashion-class 7 --num-samples 8
  
  # Generate with higher creativity
  fashion-generator generate --temperature 1.2 --num-samples 12
  
  # Evaluate model performance
  fashion-generator evaluate
  
  # Test generation capabilities
  fashion-generator test
  
  # Validate environment
  fashion-generator validate
        """
    )
    
    # Add version
    parser.add_argument('--version', action='version', version='Fashion Item Generator 2.0.0')
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Generate command
    generate_parser = subparsers.add_parser('generate', help='Generate fashion samples')
    generate_parser.add_argument('--fashion-class', type=int, default=None,
                                help='Fashion class to generate (0-9): 0=T-shirt, 1=Trouser, 2=Pullover, 3=Dress, 4=Coat, 5=Sandal, 6=Shirt, 7=Sneaker, 8=Bag, 9=Ankle boot')
    generate_parser.add_argument('--num-samples', type=int, default=16,
                                help='Number of samples to generate (default: 16)')
    generate_parser.add_argument('--temperature', type=float, default=0.8,
                                help='Generation temperature (0.5-1.5, default: 0.8)')
    generate_parser.add_argument('--model-path', type=str, default='models/superior_vae_ultimate.pth',
                                help='Path to trained model (default: models/superior_vae_ultimate.pth)')
    generate_parser.set_defaults(func=generate_samples)
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the Superior VAE model')
    train_parser.set_defaults(func=train_model)
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate model performance')
    eval_parser.set_defaults(func=evaluate_model)
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Test generation capabilities')
    test_parser.set_defaults(func=test_generation)
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate environment setup')
    validate_parser.set_defaults(func=validate_environment)
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Execute command
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\\n🛑 Operation cancelled by user")
        return 130
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())