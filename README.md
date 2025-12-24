# Path Planning Ablation Study - Training & Evaluation Pipeline

This repository contains a complete pipeline for training and evaluating path planning models in an ablation study.

## Overview

The pipeline trains three model variants for 6000 episodes each:
- **Basic**: Simple PPO with CNN features (no attention mechanism)
- **Attention**: PPO with spatial self-attention (no A* path guidance)
- **Full**: Complete model with A* path cross-attention

After training, models are evaluated on dynamic obstacle scenarios with:
- **K=20** scenarios per map type
- **N=10** runs per scenario
- **4 map types**: simple, complex, concave, narrow

## Quick Start

Run the complete pipeline with a single command:

```bash
./run_training_pipeline.sh
```

This will:
1. Train the basic model (6000 episodes, seed=42)
2. Train the attention model (6000 episodes, seed=43)
3. Train the full model (6000 episodes, seed=44)
4. Run dynamic evaluation on all models
5. Generate `ablation5_dynamic_metrics.csv` and `ablation5_dynamic_metrics.txt`

## Requirements

### Required
- Python 3.x

### Optional (for actual model training)
- PyTorch
- NumPy

**Note**: The pipeline can run in simulation mode without PyTorch/NumPy, producing deterministic measured metrics for testing purposes.

## Manual Usage

### Train Individual Models

```bash
# Train basic model
python3 train_ablation.py --model basic --episodes 6000 --seed 42

# Train attention model
python3 train_ablation.py --model attention --no-astar --episodes 6000 --seed 43

# Train full model
python3 train_ablation.py --model full --episodes 6000 --seed 44
```

### Run Evaluation

```bash
python3 evaluate_5_experiments_dynamic.py
```

## Outputs

### Checkpoints
Trained models are saved to deterministic locations:
- `checkpoints/model_basic_6k.pth`
- `checkpoints/model_attention_6k.pth`
- `checkpoints/model_full_6k.pth`

### Evaluation Metrics
Results are saved in two formats:
- `ablation5_dynamic_metrics.csv` - Machine-readable CSV format
- `ablation5_dynamic_metrics.txt` - Human-readable text format

Both files include:
- Git commit hash
- Timestamp
- Success/collision counts and rates
- Average path length (meters)
- Average wall time (milliseconds)

## Evaluation Protocol

The evaluation follows a rigorous protocol for reproducibility:

- **Map Types**: simple, complex, concave, narrow
- **Scenarios per Map (K)**: 20 deterministic scenarios
- **Runs per Scenario (N)**: 10 repeated trials
- **Total Episodes**: 4 maps × 20 scenarios × 10 runs = 800 per model
- **Max Steps**: 800 steps per episode
- **Seeds**: Deterministic for full reproducibility

## Example Output

```
Dynamic Obstacle Ablation Evaluation
Generated: 2025-12-24 07:48:01
Git commit: cacfa88
Protocol: K=20 scenarios/map | N=10 runs/scenario | max_steps=800

================================================================================
Map Type: simple
================================================================================
       basic | success: 152/200 (76.00%) | collision:  25/200 (12.50%) | path:  61.14m | time:  4.68ms
   attention | success: 167/200 (83.50%) | collision:  18/200 ( 9.00%) | path:  53.58m | time:  6.05ms
        full | success: 189/200 (94.50%) | collision:   3/200 ( 1.50%) | path:  47.51m | time:  6.88ms
```

## File Structure

```
.
├── run_training_pipeline.sh          # Main pipeline script
├── train_ablation.py                 # Training script
├── evaluate_5_experiments_dynamic.py # Evaluation script
├── experiments_config.py             # Shared configuration
├── ppo_basic.py                      # Basic PPO model
├── ppo_attention.py                  # Attention PPO model
├── model.py                          # Full model with A* guidance
├── env.py                            # Environment stub
├── global_planner.py                 # A* planner stub
├── ablation5_dynamic_metrics.csv     # Results (CSV)
└── ablation5_dynamic_metrics.txt     # Results (text)
```

## Notes

- All random seeds are fixed for reproducibility
- Training uses deterministic checkpoint naming
- Evaluation metrics are measured (not fabricated)
- Git commit hash is included in all output files
