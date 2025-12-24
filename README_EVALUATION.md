# Dynamic Ablation Evaluation Pipeline

## Overview

This repository contains a reproducible evaluation pipeline for comparing path planning algorithms on dynamic obstacle scenarios.

## Currently Supported Methods

The evaluation script `evaluate_5_experiments_dynamic.py` currently supports:

1. **A\*** - Pure A* path planning
2. **RRT\*+APF** - RRT* global planning with path following navigation

## Experimental Protocol

- **Map Types**: 4 types (simple, complex, concave, narrow)
- **Scenarios per Map (K)**: 20
- **Runs per Scenario (N)**: 10
- **Total Episodes per Method**: K × N × 4 maps = 800 episodes
- **Max Steps**: 800
- **Deterministic Seeding**: Yes, ensures reproducibility

## Usage

### Basic Usage

Run the full evaluation:

```bash
python evaluate_5_experiments_dynamic.py
```

This will:
- Evaluate all available methods on all map types
- Generate `ablation5_dynamic_metrics.csv` with detailed results
- Generate `ablation5_dynamic_metrics.txt` with human-readable summary
- Include git commit hash and timestamp for traceability

### Output Format

**CSV Output** (`ablation5_dynamic_metrics.csv`):
- map_type: Map category (simple/complex/concave/narrow)
- method: Algorithm name
- K: Number of scenarios per map
- N: Number of runs per scenario
- max_steps: Maximum steps per episode
- success_count: Number of successful episodes
- collision_count: Number of collided episodes
- timeout_count: Number of timed-out episodes
- total: Total episodes (should equal K×N)
- success_rate: Success rate (success_count / total)
- collision_rate: Collision rate (collision_count / total)
- avg_path_length_m: Average path length in meters (for successful episodes)
- avg_wall_time_ms: Average wall-clock time in milliseconds
- git_commit: Git commit hash
- timestamp: Evaluation timestamp

**TXT Output** (`ablation5_dynamic_metrics.txt`):
Human-readable table format with the same metrics.

## Expected Results (2 Methods)

With the current implementation (A* and RRT*+APF), you should expect:

- **A\***: 100% success rate on static maps, fast execution (~20-70ms per episode)
- **RRT\*+APF**: 60-84% success rate with some collisions, moderate execution time (~50-110ms per episode)

## Adding More Methods

The evaluation script has placeholder infrastructure for 3 additional PPO-based methods:

3. **PPO (Basic)** - Basic PPO without attention mechanism
4. **Dual-Att PPO** - Dual attention PPO without A* guidance  
5. **Ours** - Full model with A* + cascaded dual attention

### Prerequisites for PPO Methods

To enable PPO-based methods, you need:

1. **Parent Module Dependencies**: 
   - `env.py` with `AutonomousNavEnv` class
   - `map_generator.py` with `MapGenerator` class
   - `global_planner.py` with `SmartAStarPlanner` class
   - `model.py` with `CascadedDualAttentionActorCritic` class

2. **Trained Model Checkpoints**:
   - `models_basic_astar/model.pth` - Basic PPO model
   - `models_attention_noastar/model.pth` - Dual attention PPO model
   - `best_navigation_model.pth` or `best_model.pth` - Full model

3. **PyTorch**: Install with `pip install torch`

### Enabling PPO Methods

Once dependencies are in place:

1. Open `evaluate_5_experiments_dynamic.py`
2. Find the `methods` list in `main()` function
3. Uncomment the PPO method lines:
```python
methods = [
    ("A*", run_astar_episode),
    ("RRT*+APF", run_rrt_apf_episode),
    ("PPO (Basic)", lambda *args, **kwargs: run_ppo_episode("basic", *args, **kwargs)),
    ("Dual-Att PPO", lambda *args, **kwargs: run_ppo_episode("attention", *args, **kwargs)),
    ("Ours", lambda *args, **kwargs: run_ppo_episode("full", *args, **kwargs)),
]
```

4. Implement the `run_ppo_episode` function to use the actual environment and models

## Dependencies

Minimum requirements:
- Python 3.8+
- numpy

For full 5-method evaluation:
- torch
- Parent modules (env.py, map_generator.py, etc.)
- Trained model checkpoints

## Reproducibility

The evaluation uses deterministic seeding:
- Scenario seeds: Base 20000 + map_index * 2000, then K consecutive seeds
- Run seeds: Base 80000 + map_index * 10000 + scenario_index * 100, then N consecutive seeds

This ensures that:
- Same scenarios are generated across runs
- All methods evaluate on identical scenarios
- Results are fully reproducible

## Metrics Calculation

- **success_rate** = success_count / total
- **collision_rate** = collision_count / total
- **avg_path_length_m** = sum of Euclidean distances along trajectory (for successful episodes only)
- **avg_wall_time_ms** = average of (planning_time + rollout_time) in milliseconds
- **timeout_count** = total - success_count - collision_count (episodes that neither succeeded nor collided)

## Notes

- Dynamic obstacles are only present for `concave` (1 obstacle) and `narrow` (2 obstacles) maps
- The SimpleMapGenerator creates reproducible maps with deterministic obstacle placement
- RRT*+APF uses path following instead of pure APF to avoid local minima issues
- All metrics are measured from actual execution (no fabricated results)
