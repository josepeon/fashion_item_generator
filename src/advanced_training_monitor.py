"""
Real-Time Superior VAE Training Monitor

Monitors the intensive 500-epoch training session and provides
comprehensive progress tracking and quality metrics.
"""

import os
import time
import json
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import numpy as np

class TrainingMonitor:
    """Real-time training monitor for Superior VAE."""
    
    def __init__(self):
        self.start_time = datetime.now()
        self.models_dir = "models"
        self.results_dir = "results"
        
    def get_training_status(self):
        """Get current training status."""
        status = {
            'training_active': False,
            'current_epoch': 0,
            'total_epochs': 500,
            'best_model_size': 0,
            'last_update': None,
            'estimated_completion': None
        }
        
        # Check for ultimate model
        ultimate_model = os.path.join(self.models_dir, "superior_vae_ultimate.pth")
        if os.path.exists(ultimate_model):
            stat = os.stat(ultimate_model)
            status['best_model_size'] = stat.st_size / (1024 * 1024)  # MB
            status['last_update'] = datetime.fromtimestamp(stat.st_mtime)
            
            # Check if recently updated (training active)
            time_diff = datetime.now() - status['last_update']
            status['training_active'] = time_diff.total_seconds() < 300  # 5 minutes
        
        # Check for checkpoints to estimate progress
        if os.path.exists(self.models_dir):
            checkpoint_files = [f for f in os.listdir(self.models_dir) 
                              if f.startswith("superior_vae_epoch_")]
            
            if checkpoint_files:
                # Extract epoch numbers
                epochs = []
                for f in checkpoint_files:
                    try:
                        epoch_num = int(f.split("_epoch_")[1].split(".")[0])
                        epochs.append(epoch_num)
                    except:
                        continue
                
                if epochs:
                    status['current_epoch'] = max(epochs)
                    
                    # Estimate completion time
                    if status['current_epoch'] > 0:
                        elapsed = datetime.now() - self.start_time
                        epochs_per_hour = status['current_epoch'] / (elapsed.total_seconds() / 3600)
                        remaining_epochs = status['total_epochs'] - status['current_epoch']
                        
                        if epochs_per_hour > 0:
                            hours_remaining = remaining_epochs / epochs_per_hour
                            status['estimated_completion'] = datetime.now() + timedelta(hours=hours_remaining)
        
        return status
    
    def display_progress(self):
        """Display current training progress."""
        status = self.get_training_status()
        
        print("🔥 SUPERIOR VAE TRAINING MONITOR")
        print("=" * 60)
        print(f"⏰ Monitor Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Training Status
        if status['training_active']:
            print("🚀 STATUS: TRAINING IN PROGRESS")
            progress = (status['current_epoch'] / status['total_epochs']) * 100
            print(f"📊 Progress: {status['current_epoch']}/{status['total_epochs']} epochs ({progress:.1f}%)")
            
            # Progress bar
            bar_length = 40
            filled_length = int(bar_length * progress / 100)
            bar = "█" * filled_length + "░" * (bar_length - filled_length)
            print(f"▓ [{bar}] {progress:.1f}%")
            
            if status['estimated_completion']:
                print(f"⏳ Estimated Completion: {status['estimated_completion'].strftime('%H:%M:%S')}")
                remaining = status['estimated_completion'] - datetime.now()
                hours = int(remaining.total_seconds() // 3600)
                minutes = int((remaining.total_seconds() % 3600) // 60)
                print(f"⏱️  Time Remaining: ~{hours}h {minutes}m")
        
        elif status['current_epoch'] >= status['total_epochs']:
            print("✅ STATUS: TRAINING COMPLETED!")
            print(f"🏆 Final Epoch: {status['current_epoch']}")
        
        else:
            print("⏸️  STATUS: TRAINING PAUSED/STARTING")
        
        print()
        
        # Model Information
        if status['best_model_size'] > 0:
            print(f"💾 Best Model: superior_vae_ultimate.pth ({status['best_model_size']:.1f}MB)")
            if status['last_update']:
                print(f"🕒 Last Updated: {status['last_update'].strftime('%H:%M:%S')}")
        
        # List checkpoints
        if os.path.exists(self.models_dir):
            checkpoints = [f for f in os.listdir(self.models_dir) if "superior_vae_epoch_" in f]
            if checkpoints:
                print(f"📁 Checkpoints: {len(checkpoints)} saved")
                # Show latest few
                latest_checkpoints = sorted(checkpoints)[-3:]
                for cp in latest_checkpoints:
                    filepath = os.path.join(self.models_dir, cp)
                    size_mb = os.path.getsize(filepath) / (1024 * 1024)
                    print(f"   📦 {cp} ({size_mb:.1f}MB)")
        
        print()
        
        # Expected Quality Improvements
        if status['current_epoch'] > 0:
            self._show_expected_improvements(status['current_epoch'])
        
        return status
    
    def _show_expected_improvements(self, current_epoch):
        """Show expected quality improvements based on training progress."""
        progress = current_epoch / 500
        
        print("🎯 EXPECTED QUALITY EVOLUTION:")
        
        # Phase-based improvements
        if progress < 0.3:  # Warmup phase (0-150 epochs)
            print("   📚 Phase: Learning Basic Patterns")
            print("   🎨 Quality: Developing fundamental features")
            print("   🔄 Beta: Low (0.1) - Focus on reconstruction")
            
        elif progress < 0.7:  # Growth phase (150-350 epochs)  
            print("   🚀 Phase: Advanced Feature Learning")
            print("   🎨 Quality: Improving diversity and detail")
            print("   🔄 Beta: Increasing - Balancing reconstruction/generation")
            
        else:  # Refinement phase (350-500 epochs)
            print("   ✨ Phase: Fine-tuning and Perfection")
            print("   🎨 Quality: Achieving maximum sophistication")
            print("   🔄 Beta: High (→2.0) - Strong disentanglement")
        
        # Projected improvements
        expected_improvements = {
            0.1: "Basic shapes and textures",
            0.3: "Clear class distinctions", 
            0.5: "Fine details and realistic textures",
            0.7: "Exceptional diversity and quality",
            0.9: "Near-perfect generation capabilities",
            1.0: "Maximum quality achieved"
        }
        
        for threshold, improvement in expected_improvements.items():
            if progress >= threshold:
                print(f"   ✅ {improvement}")
            else:
                print(f"   ⏳ {improvement}")
                break
    
    def continuous_monitor(self, check_interval=300):  # 5 minutes
        """Run continuous monitoring."""
        print("🔍 STARTING CONTINUOUS MONITORING")
        print(f"   Checking every {check_interval//60} minutes")
        print("   Press Ctrl+C to stop monitoring")
        print("=" * 60)
        
        try:
            while True:
                status = self.display_progress()
                
                # Check if training completed
                if not status['training_active'] and status['current_epoch'] >= status['total_epochs']:
                    print("\n🎉 TRAINING COMPLETED - MONITORING FINISHED!")
                    break
                
                print(f"\n💤 Sleeping for {check_interval//60} minutes...")
                print("=" * 60)
                time.sleep(check_interval)
                
        except KeyboardInterrupt:
            print("\n🛑 Monitoring stopped by user")
    
    def generate_training_report(self):
        """Generate a comprehensive training report."""
        status = self.get_training_status()
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'training_status': status,
            'architecture_specs': {
                'parameters': '33,513,370',
                'latent_dimensions': 64,
                'attention_heads': 8,
                'batch_size': 256,
                'target_epochs': 500
            },
            'expected_outcomes': {
                'quality_grade': 'A+ EXCEPTIONAL or higher',
                'reconstruction_mse': '< 0.05 (previous: 0.0633)',
                'generation_diversity': '> 15.0 (previous: 13.32)',
                'overall_score': '> 3.0 (previous: 2.66)'
            }
        }
        
        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = f'results/training_report_{timestamp}.json'
        
        os.makedirs(self.results_dir, exist_ok=True)
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📄 Training report saved: {report_path}")
        return report


def main():
    """Main monitoring function."""
    monitor = TrainingMonitor()
    
    # Show current status
    status = monitor.display_progress()
    
    # Ask user for continuous monitoring
    if status['training_active']:
        print("\n🤔 Options:")
        print("   1. Single status check (current)")
        print("   2. Continuous monitoring (every 5 minutes)")
        print("   3. Generate training report")
        
        try:
            choice = input("\nEnter choice (1/2/3): ").strip()
            
            if choice == '2':
                monitor.continuous_monitor()
            elif choice == '3':
                monitor.generate_training_report()
            
        except KeyboardInterrupt:
            pass
    
    elif status['current_epoch'] >= status['total_epochs']:
        print("🎊 Training completed! Generating final report...")
        monitor.generate_training_report()


if __name__ == "__main__":
    main()