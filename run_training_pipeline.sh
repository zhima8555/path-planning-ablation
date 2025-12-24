#!/bin/bash
# Training and Evaluation Pipeline for Path Planning Ablation Study
#
# This script trains 3 models (basic, attention, full) for 6000 episodes each
# with fixed seeds, then runs dynamic evaluation to generate metrics.
#
# Requirements:
# - Python 3.x
# - PyTorch (optional - script will run in simulation mode if unavailable)
# - NumPy
#
# Usage:
#   ./run_training_pipeline.sh
#
# Outputs:
# - checkpoints/model_basic_6k.pth
# - checkpoints/model_attention_6k.pth
# - checkpoints/model_full_6k.pth
# - ablation5_dynamic_metrics.csv
# - ablation5_dynamic_metrics.txt

set -e  # Exit on error

echo "================================================================================"
echo "Path Planning Ablation Study - Training & Evaluation Pipeline"
echo "================================================================================"
echo "Training 3 models for 6000 episodes each with fixed seeds"
echo "Then running dynamic evaluation with K=20, N=10"
echo ""

# Create checkpoints directory
mkdir -p checkpoints

# Check if PyTorch is available
if python3 -c "import torch" 2>/dev/null; then
    echo "✓ PyTorch is available - will train actual models"
    TORCH_AVAILABLE=true
else
    echo "⚠ PyTorch not available - will create mock checkpoints"
    TORCH_AVAILABLE=false
fi

echo ""
echo "================================================================================"
echo "Step 1/4: Training Basic Model (6000 episodes, seed=42)"
echo "================================================================================"
if [ "$TORCH_AVAILABLE" = true ]; then
    python3 train_ablation.py --model basic --episodes 6000 --seed 42
else
    echo "Creating mock checkpoint for basic model..."
    echo "# Mock checkpoint - PyTorch not available" > checkpoints/model_basic_6k.pth
fi

echo ""
echo "================================================================================"
echo "Step 2/4: Training Attention Model (6000 episodes, seed=43)"
echo "  Note: Uses --no-astar to ablate A* path guidance (tests attention only)"
echo "================================================================================"
if [ "$TORCH_AVAILABLE" = true ]; then
    python3 train_ablation.py --model attention --no-astar --episodes 6000 --seed 43
else
    echo "Creating mock checkpoint for attention model..."
    echo "# Mock checkpoint - PyTorch not available" > checkpoints/model_attention_6k.pth
fi

echo ""
echo "================================================================================"
echo "Step 3/4: Training Full Model (6000 episodes, seed=44)"
echo "================================================================================"
if [ "$TORCH_AVAILABLE" = true ]; then
    python3 train_ablation.py --model full --episodes 6000 --seed 44
else
    echo "Creating mock checkpoint for full model..."
    echo "# Mock checkpoint - PyTorch not available" > checkpoints/model_full_6k.pth
fi

echo ""
echo "================================================================================"
echo "Step 4/4: Running Dynamic Evaluation (K=20, N=10)"
echo "================================================================================"
python3 evaluate_5_experiments_dynamic.py

echo ""
echo "================================================================================"
echo "Pipeline Complete!"
echo "================================================================================"
echo "Generated outputs:"
echo "  - checkpoints/model_basic_6k.pth"
echo "  - checkpoints/model_attention_6k.pth"
echo "  - checkpoints/model_full_6k.pth"
echo "  - ablation5_dynamic_metrics.csv"
echo "  - ablation5_dynamic_metrics.txt"
echo ""
echo "View results:"
echo "  cat ablation5_dynamic_metrics.txt"
echo "================================================================================"
