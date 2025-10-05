#!/usr/bin/env python3
"""
Environment Validation Script
Verify that all dependencies are correctly installed and working.
"""

import sys
import importlib
from typing import List, Tuple

def check_package(package_name: str, min_version: str = None, version_compare_func=None) -> Tuple[bool, str]:
    """Check if a package is installed and meets minimum version requirements."""
    try:
        module = importlib.import_module(package_name)
        
        if hasattr(module, '__version__'):
            version = module.__version__
            if min_version:
                if version_compare_func:
                    if not version_compare_func(version, min_version):
                        return False, f"❌ {package_name} {version} < required {min_version}"
                else:
                    # Simple string comparison fallback
                    if version < min_version:
                        return False, f"❌ {package_name} {version} < required {min_version}"
            return True, f"✅ {package_name} {version}"
        else:
            return True, f"✅ {package_name} (version not available)"
            
    except ImportError:
        return False, f"❌ {package_name} not installed"

def validate_environment():
    """Validate the Enhanced VAE environment setup."""
    print("🔍 Enhanced VAE Environment Validation")
    print("=" * 50)
    print(f"Python version: {sys.version}")
    print()
    
    # Required packages with minimum versions
    packages_to_check = [
        ("torch", "2.0.0"),
        ("torchvision", "0.15.0"), 
        ("numpy", "1.24.0"),
        ("matplotlib", "3.7.0"),
        ("PIL", "10.0.0"),  # Pillow
        ("sklearn", "1.3.0"),  # scikit-learn (optional)
    ]
    
    # Special handling for version comparison
    def version_compare(current, required):
        """Compare version strings handling complex formats like '3.10.6.dev0+g0318a2669a'."""
        try:
            # Clean version strings by taking only the numeric part
            def clean_version(v):
                # Split on common separators and take numeric parts only
                import re
                # Extract major.minor.patch pattern
                match = re.match(r'(\d+)\.(\d+)\.(\d+)', v)
                if match:
                    return [int(match.group(1)), int(match.group(2)), int(match.group(3))]
                # Fallback: split on dots and take first numeric parts
                parts = []
                for part in v.split('.'):
                    # Extract numbers from the beginning of each part
                    num_match = re.match(r'(\d+)', part)
                    if num_match:
                        parts.append(int(num_match.group(1)))
                    else:
                        break
                return parts[:3]  # Limit to major.minor.patch
            
            current_parts = clean_version(current)
            required_parts = clean_version(required)
            
            # Pad shorter version with zeros
            max_len = max(len(current_parts), len(required_parts))
            current_parts += [0] * (max_len - len(current_parts))
            required_parts += [0] * (max_len - len(required_parts))
            
            return current_parts >= required_parts
            
        except Exception:
            # Fallback to string comparison if parsing fails
            return current >= required
    
    all_ok = True
    
    for package, min_ver in packages_to_check:
        success, message = check_package(package, min_ver, version_compare)
        print(message)
        if not success:
            all_ok = False
    
    print()
    print("🧪 Testing PyTorch functionality...")
    
    try:
        import torch
        
        # Test basic tensor operations
        x = torch.randn(2, 3)
        y = torch.mm(x, x.t())
        print(f"✅ Basic tensor operations work")
        
        # Test device availability
        if torch.backends.mps.is_available():
            print(f"✅ Apple Silicon MPS acceleration available")
            device = torch.device('mps')
            test_tensor = torch.randn(10, 10, device=device)
            print(f"✅ MPS device operations work")
        elif torch.cuda.is_available():
            print(f"✅ CUDA GPU acceleration available")
        else:
            print(f"ℹ️  Using CPU (no GPU acceleration)")
            
    except Exception as e:
        print(f"❌ PyTorch functionality test failed: {e}")
        all_ok = False
    
    print()
    print("🎨 Testing matplotlib...")
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend for testing
        
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 2])
        plt.close(fig)
        print("✅ Matplotlib plotting works")
    except Exception as e:
        print(f"❌ Matplotlib test failed: {e}")
        all_ok = False
    
    print()
    if all_ok:
        print("🎉 Environment validation SUCCESSFUL!")
        print("✅ Ready for Enhanced VAE fashion item generation")
        print()
        print("Quick start:")
        print("  python src/showcase_enhanced_vae.py")
        print("  python src/test_vae_comprehensive.py")
    else:
        print("❌ Environment validation FAILED!")
        print("Please install missing dependencies:")
        print("  conda env update -f environment.yml")
        print("  # or")
        print("  pip install -r requirements.txt")
    
    return all_ok

if __name__ == "__main__":
    validate_environment()