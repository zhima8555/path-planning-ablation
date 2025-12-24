# Ablation Evaluation - Dynamic Obstacles

This directory contains the complete 5-method dynamic obstacle ablation evaluation pipeline.

## Overview

The evaluation compares five path planning methods on dynamic obstacle scenarios:

1. **A* tracking** - Global path from A* with simple waypoint following (no dynamic avoidance)
2. **RRT*+APF** - RRT* for global planning + APF for local dynamic avoidance
3. **PPO Basic** - Basic PPO with simple CNN (no attention mechanisms)
4. **Dual-Att PPO** - PPO with spatial self-attention (no A* path guidance)
5. **Ours (Full)** - Complete model with cascaded dual attention + A* path guidance

## Maps and Dynamic Obstacles

The evaluation uses 4 map types with varying complexity:

- **simple**: Simple obstacles, no dynamic obstacles
- **complex**: Complex obstacles, no dynamic obstacles
- **concave**: Concave (U-shaped) obstacles, **1 dynamic obstacle**
- **narrow**: Narrow passages, **2 dynamic obstacles**

Dynamic obstacles move with constant velocity and bounce off walls. Collisions are counted when the agent gets within `radius` (default 2.0) of an obstacle.

## Evaluation Protocol

- **K = 20** scenarios per map type (deterministic seeds)
- **N = 10** runs per scenario
- **Total episodes per method**: 20 × 10 = 200 per map = **800 total per method**
- **max_steps = 800** per episode
- All methods evaluated on identical scenarios (deterministic seeding)

## Running the Evaluation

### Prerequisites

Trained model checkpoints (optional - will use random initialization if missing):
- `models_basic_astar/model.pth` - PPO Basic model
- `models_attention_noastar/model.pth` - Dual-Att PPO model  
- `best_model.pth` - Ours (Full) model

### Training Models

To train models from scratch:

```bash
# Train PPO Basic (with A* guidance)
python train_ablation.py --model basic --episodes 3000

# Train Dual-Att PPO (without A* guidance)
python train_ablation.py --model attention --no-astar --episodes 3000

# Train Ours (Full) model (with A* guidance)
python train_ablation.py --model full --episodes 3000
```

### Running Evaluation

```bash
# Full evaluation (K=20, N=10)
python evaluate_5_experiments_dynamic.py
```

This will generate:
- `ablation5_dynamic_metrics.csv` - Detailed results in CSV format
- `ablation5_dynamic_metrics.txt` - Human-readable summary

## Output Format

### CSV Columns

- `timestamp`: Evaluation timestamp
- `git_commit`: Git commit hash (first 8 chars)
- `map_type`: Map type (simple/complex/concave/narrow)
- `method`: Method name
- `K`: Number of scenarios per map
- `N`: Number of runs per scenario
- `max_steps`: Maximum steps per episode
- `success_count`: Number of successful episodes
- `collision_count`: Number of collision episodes
- `total`: Total episodes (K × N)
- `success_rate`: Success rate (0.0 to 1.0)
- `collision_rate`: Collision rate (0.0 to 1.0)
- `avg_path_length`: Average path length for successful episodes

### TXT Format

Human-readable table with:
- Metadata (timestamp, git commit, parameters, checkpoint paths)
- Per-map results for all 5 methods
- Success/collision counts and rates
- Average path lengths

## Key Implementation Details

### Collision Detection

For **concave** and **narrow** maps (which have dynamic obstacles):
- Collisions are counted for ALL methods, including A* tracking
- A collision occurs when `distance(agent_pos, obstacle_pos) < obstacle_radius`
- Default obstacle radius is 2.0

### Dynamic Obstacle Updates

Dynamic obstacles update each step:
```python
pos = pos + vel
if out_of_bounds:
    vel = -vel  # bounce off walls
```

### A* Tracking Method

The A* tracking method:
1. Plans a global path using A* (static map only)
2. Follows waypoints using simple angle-based control
3. **Does NOT avoid dynamic obstacles** (can collide)
4. Collisions are still counted for fair comparison

### RRT*+APF Method

The RRT*+APF method:
1. Plans global path using RRT*
2. Uses APF (Artificial Potential Field) for local control
3. **Does avoid dynamic obstacles** via repulsive forces

### PPO Methods

All PPO-based methods:
1. Use observation: local map view (40×40) + vector state
2. Trained with identical hyperparameters
3. Differ in architecture and A* guidance

## Expected Results

With trained models, you should see:

- **A* tracking**: High success on simple/complex, collisions on concave/narrow
- **RRT*+APF**: Moderate success, can handle dynamic obstacles
- **PPO Basic**: Lower success, limited by simple CNN
- **Dual-Att PPO**: Better than basic, but struggles without A* guidance
- **Ours (Full)**: Best overall performance with dual attention + A* guidance

## Reproducibility

All evaluations are deterministic:
- Fixed scenario seeds per map type
- Fixed run seeds per scenario
- Same random number generator state
- Results should be identical across runs

## File Structure

```
.
├── evaluate_5_experiments_dynamic.py  # Main evaluation script
├── experiments_config.py              # Shared configuration constants
├── env.py                             # Navigation environment
├── map_generator.py                   # Map and obstacle generation
├── global_planner.py                  # A* planner
├── model.py                           # Ours (Full) model
├── ppo_basic.py                       # PPO Basic model
├── ppo_attention.py                   # Dual-Att PPO model
├── rrt_apf_planner.py                 # RRT*+APF navigator
├── train_ablation.py                  # Training script
├── ablation5_dynamic_metrics.csv      # Output (generated)
└── ablation5_dynamic_metrics.txt      # Output (generated)
```

## Troubleshooting

### Missing Checkpoints

If checkpoints are missing, the evaluation will:
1. Print a warning message
2. Use random initialization for that model
3. Continue with evaluation
4. Results will show poor performance (expected)

Solution: Train models using `train_ablation.py`

### Import Errors

Ensure all dependencies are installed:
```bash
pip install numpy torch matplotlib
```

### Out of Memory

For GPU memory issues, the evaluation automatically falls back to CPU.

To force CPU usage:
```python
device = torch.device('cpu')
```

## Citation

If you use this evaluation framework, please cite:

```bibtex
@article{yourpaper2025,
  title={Path Planning with Cascaded Dual Attention},
  author={Your Name},
  journal={IEEE Conference},
  year={2025}
}
```
