# Path Planning Ablation Study

This repository contains an ablation study comparing 5 path planning methods with dynamic obstacles.

## Methods Evaluated

1. **A* tracking** - Pure A* path planning with waypoint tracking (no dynamic avoidance)
2. **RRT* + APF** - RRT* global planning with Artificial Potential Field local avoidance
3. **PPO (Basic)** - Basic PPO with A* guidance but no attention mechanism
4. **PPO (Dual-Att)** - PPO with Dual-Attention mechanism but without A* guidance
5. **Ours (Full)** - Complete model with A* guidance + Dual-Attention PPO

## Quick Reference

```bash
# Full pipeline (train + evaluate)
python run_dynamic_ablation_pipeline.py --train --evaluate

# Evaluate only (if models already trained)
python run_dynamic_ablation_pipeline.py --evaluate-only

# View results
cat ablation5_dynamic_metrics.txt

# Run validation tests
python test_evaluation_logic.py
```

## Evaluation Protocol

### Dynamic Evaluation
- **Maps**: 4 types (simple, complex, concave, narrow)
- **Scenarios**: K=20 scenarios per map
- **Runs**: N=10 runs per scenario
- **Total episodes**: K × N = 200 per (map, method)
- **Max steps**: 800 (configurable in `experiments_config.py`)
- **Reproducibility**: Fixed seeds for deterministic scenario generation

### Metrics Reported
- Success count & rate
- Collision count & rate
- Timeout count
- Average path length (meters, for successful episodes only)
- Average wall time (milliseconds)
- Metadata: timestamp, git commit hash, checkpoint paths

## Quick Start

### Prerequisites

**Important:** This ablation study requires the main path-planning repository to be properly set up. Ensure that the parent modules (`env.py`, `map_generator.py`, `global_planner.py`, `model.py`) are available in the parent directory before running the pipeline.

If you're setting up for the first time:
1. Clone/set up the main path-planning repository
2. Navigate to the ablation directory
3. Install Python dependencies: `pip install numpy torch matplotlib`

### Running the Complete Pipeline

The easiest way to run the complete ablation study:

```bash
# Run training + evaluation (recommended for first time)
python run_dynamic_ablation_pipeline.py --train --evaluate

# Run only evaluation (if models are already trained)
python run_dynamic_ablation_pipeline.py --evaluate-only

# Run only training
python run_dynamic_ablation_pipeline.py --train-only

# Skip training if checkpoints exist
python run_dynamic_ablation_pipeline.py --train --evaluate --skip-existing

# Custom training episodes
python run_dynamic_ablation_pipeline.py --train --train-episodes 5000
```

### Manual Training

Train individual models:

```bash
# Train PPO Basic (with A* guidance)
python train_ablation.py --model basic --episodes 3000

# Train PPO Dual-Attention (without A* guidance)
python train_ablation.py --model attention --no-astar --episodes 3000

# Train Full model (requires complete setup from parent repo)
# Use the main repository's training script
```

### Manual Evaluation

Run evaluation directly:

```bash
python evaluate_5_experiments_dynamic.py
```

## Output Files

After running the pipeline, the following files are generated in the repository root:

### Primary Outputs
- `ablation5_dynamic_metrics.csv` - Machine-readable metrics (CSV format)
- `ablation5_dynamic_metrics.txt` - Human-readable report with full metadata

### Example Output Format

**CSV fields:**
```
map_type,method,K,N,max_steps,success_count,collision_count,timeout_count,total,
success_rate,collision_rate,avg_path_length_m,avg_wall_time_ms,timestamp,git_commit
```

**TXT format:**
```
==================================================================================
DYNAMIC ABLATION STUDY - 5 METHODS
==================================================================================
Timestamp: 2025-12-24 12:00:00
Git commit: abc1234
Protocol: K=20 scenarios/map, N=10 runs/scenario, max_steps=800

Checkpoint paths:
  PPO (Basic): models_basic_astar/model.pth
  PPO (Dual-Att): models_attention_noastar/model.pth
  Ours (Full): best_navigation_model.pth

Results:
Map        Method               Succ   Coll Timeout Total SuccRate CollRate AvgLen(m)  AvgTime(ms)
--------------------------------------------------------------------------------------------------
simple     A* tracking           180     15       5   200    0.900    0.075      92.31       15.23
...
```

## Model Checkpoints

Models are saved to deterministic locations:

- `models_basic_astar/model.pth` - PPO Basic
- `models_attention_noastar/model.pth` - PPO Dual-Attention
- `best_navigation_model.pth` - Full model (Ours)

## Configuration

Evaluation parameters are centralized in `experiments_config.py`:

```python
MAP_TYPES = ["simple", "complex", "concave", "narrow"]
DYNAMIC_K = 20  # scenarios per map
DYNAMIC_N = 10  # runs per scenario
MAX_STEPS_DYNAMIC = 800  # max steps per episode
```

Modify these constants to adjust the evaluation protocol.

## Reproducibility

All evaluations use deterministic seeds:
- Scenario seeds: based on map type index (base: 20000 + map_index * 2000)
- Run seeds: based on map type and scenario index (base: 80000 + map_index * 10000 + scenario_idx * 100)

This ensures:
1. Same scenarios generated for all methods
2. Same random initialization for each run
3. Reproducible results across runs

## Dependencies

### Required Parent Modules
This ablation study package requires the main path-planning repository to be set up in the parent directory. The following modules must be available:

- `env.py` - AutonomousNavEnv (autonomous navigation environment)
- `map_generator.py` - MapGenerator (map generation utilities)
- `global_planner.py` - SmartAStarPlanner (A* path planning)
- `model.py` - CascadedDualAttentionActorCritic (full attention model)

**Repository Structure:**
```
path-planning-repo/          # Main repository
├── env.py
├── map_generator.py
├── global_planner.py
├── model.py
└── ablation/                 # This directory
    ├── ppo_basic.py
    ├── ppo_attention.py
    ├── rrt_apf_planner.py
    ├── train_ablation.py
    ├── evaluate_5_experiments_dynamic.py
    └── run_dynamic_ablation_pipeline.py
```

### Python Packages
```bash
pip install numpy torch matplotlib
```

If the main repository is not set up, the evaluation scripts will fail with module import errors.

## Additional Scripts

- `dynamic_obstacles_metrics.py` - Evaluation on concave/narrow maps only (3 methods)
- `run_ablation_evaluation.py` - Comprehensive ablation with visualizations
- `train_ablation.py` - Unified training script for all PPO variants
- `test_evaluation_logic.py` - Validation tests for evaluation logic

## Troubleshooting

### Import Errors
If you see `ModuleNotFoundError` for `env`, `map_generator`, `global_planner`, or `model`:
- Ensure the main path-planning repository is set up in the parent directory
- Check that you're running from the ablation subdirectory
- Verify the repository structure matches the expected layout (see Dependencies section)

### Missing Checkpoints
If you see warnings about missing checkpoint files:
- Run the training phase first: `python run_dynamic_ablation_pipeline.py --train-only`
- Or train individual models: `python train_ablation.py --model basic --episodes 3000`
- The evaluation will use random initialization if checkpoints are missing (results will be poor)

### Validation
To verify your setup is correct:
```bash
python test_evaluation_logic.py
```
All 6 tests should pass.

## Citation

If you use this ablation study, please cite:

```
@article{path-planning-ablation-2025,
  title={Path Planning with Dynamic Obstacles: An Ablation Study},
  author={...},
  year={2025}
}
```

## License

[Specify license here]
