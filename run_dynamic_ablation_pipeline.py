#!/usr/bin/env python3
"""One-command pipeline for dynamic ablation study.

This script orchestrates:
1. Training all required models (Basic PPO, Dual-Att PPO, Full model)
2. Running evaluation (evaluate_5_experiments_dynamic.py)
3. Generating outputs (CSV + TXT) with full metadata

Usage:
    python run_dynamic_ablation_pipeline.py --train --evaluate
    python run_dynamic_ablation_pipeline.py --evaluate-only  # Skip training
    python run_dynamic_ablation_pipeline.py --train-only     # Skip evaluation

Options:
    --train-episodes: Number of training episodes per model (default: 3000)
    --skip-existing: Skip training if checkpoint already exists
"""

import argparse
import os
import sys
import subprocess
from datetime import datetime


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 100)
    print(f"  {title}")
    print("=" * 100 + "\n")


def run_command(cmd: list, description: str):
    """Run a command and handle errors."""
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print()
    
    result = subprocess.run(cmd, cwd=os.path.dirname(__file__))
    
    if result.returncode != 0:
        print(f"\n❌ Error: {description} failed with exit code {result.returncode}")
        return False
    
    print(f"\n✓ {description} completed successfully")
    return True


def check_checkpoint_exists(path: str) -> bool:
    """Check if a checkpoint file exists."""
    return os.path.exists(path)


def train_model(model_type: str, use_astar: bool, episodes: int, skip_existing: bool = False):
    """Train a single model."""
    # Determine checkpoint path
    suffix = 'astar' if use_astar else 'noastar'
    checkpoint_dir = f"models_{model_type}_{suffix}"
    checkpoint_path = os.path.join(checkpoint_dir, "model.pth")
    
    # Check if we should skip
    if skip_existing and check_checkpoint_exists(checkpoint_path):
        print(f"  ✓ Checkpoint already exists: {checkpoint_path}")
        print(f"  Skipping training for {model_type} ({suffix})")
        return True
    
    # Build command
    cmd = [
        sys.executable,
        "train_ablation.py",
        "--model", model_type,
        "--episodes", str(episodes),
    ]
    
    if not use_astar:
        cmd.append("--no-astar")
    
    desc = f"Training {model_type} ({'with' if use_astar else 'without'} A*) for {episodes} episodes"
    
    return run_command(cmd, desc)


def train_all_models(episodes: int, skip_existing: bool = False):
    """Train all required models."""
    print_section("TRAINING PHASE")
    
    models_to_train = [
        ("basic", True, "PPO Basic with A* guidance"),
        ("attention", False, "PPO Dual-Attention without A* guidance"),
        # Note: "full" model training would require the complete model.py from parent
        # For now, we'll assume it's trained separately or exists as best_navigation_model.pth
    ]
    
    success = True
    for model_type, use_astar, description in models_to_train:
        print(f"\n--- {description} ---")
        if not train_model(model_type, use_astar, episodes, skip_existing):
            success = False
            print(f"  ❌ Failed to train {model_type}")
            # Continue with other models even if one fails
    
    # Check for full model
    full_model_path = "best_navigation_model.pth"
    if not check_checkpoint_exists(full_model_path):
        print(f"\n⚠️  Warning: Full model checkpoint not found: {full_model_path}")
        print(f"   The 'Ours (Full)' method will use random initialization.")
        print(f"   To train the full model, use the main repository training script.")
    else:
        print(f"\n✓ Full model checkpoint found: {full_model_path}")
    
    return success


def run_evaluation():
    """Run the evaluation script."""
    print_section("EVALUATION PHASE")
    
    cmd = [sys.executable, "evaluate_5_experiments_dynamic.py"]
    desc = "Evaluating all 5 methods (dynamic ablation)"
    
    return run_command(cmd, desc)


def check_outputs():
    """Check that output files were created."""
    print_section("OUTPUT VERIFICATION")
    
    expected_files = [
        "ablation5_dynamic_metrics.csv",
        "ablation5_dynamic_metrics.txt",
    ]
    
    all_exist = True
    for fname in expected_files:
        if os.path.exists(fname):
            size = os.path.getsize(fname)
            print(f"  ✓ {fname} ({size} bytes)")
        else:
            print(f"  ❌ Missing: {fname}")
            all_exist = False
    
    return all_exist


def show_summary():
    """Show a summary of the pipeline results."""
    print_section("PIPELINE SUMMARY")
    
    print("Output files:")
    print("  - ablation5_dynamic_metrics.csv (machine-readable metrics)")
    print("  - ablation5_dynamic_metrics.txt (human-readable report)")
    print()
    print("Checkpoint locations:")
    print("  - models_basic_astar/model.pth (PPO Basic)")
    print("  - models_attention_noastar/model.pth (PPO Dual-Attention)")
    print("  - best_navigation_model.pth (Ours Full, if available)")
    print()
    print("To view results:")
    print("  cat ablation5_dynamic_metrics.txt")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="One-command pipeline for dynamic ablation study",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Pipeline control
    parser.add_argument("--train", action="store_true", 
                       help="Run training phase")
    parser.add_argument("--evaluate", action="store_true",
                       help="Run evaluation phase")
    parser.add_argument("--train-only", action="store_true",
                       help="Run only training (skip evaluation)")
    parser.add_argument("--evaluate-only", action="store_true",
                       help="Run only evaluation (skip training)")
    
    # Training options
    parser.add_argument("--train-episodes", type=int, default=3000,
                       help="Number of training episodes per model (default: 3000)")
    parser.add_argument("--skip-existing", action="store_true",
                       help="Skip training if checkpoint already exists")
    
    args = parser.parse_args()
    
    # Determine what to run
    if args.train_only:
        do_train = True
        do_evaluate = False
    elif args.evaluate_only:
        do_train = False
        do_evaluate = True
    elif args.train or args.evaluate:
        do_train = args.train
        do_evaluate = args.evaluate
    else:
        # Default: run both
        do_train = True
        do_evaluate = True
    
    print_section("DYNAMIC ABLATION PIPELINE")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Training: {'Yes' if do_train else 'No'}")
    print(f"Evaluation: {'Yes' if do_evaluate else 'No'}")
    
    # Execute pipeline
    success = True
    
    if do_train:
        if not train_all_models(args.train_episodes, args.skip_existing):
            print("\n⚠️  Warning: Some models failed to train")
            success = False
    
    if do_evaluate:
        if not run_evaluation():
            print("\n❌ Error: Evaluation failed")
            success = False
        else:
            check_outputs()
    
    # Show summary
    show_summary()
    
    # Final status
    print_section("PIPELINE STATUS")
    if success:
        print("✓ Pipeline completed successfully!")
        print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        return 0
    else:
        print("⚠️  Pipeline completed with warnings/errors")
        print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
