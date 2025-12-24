"""Dynamic evaluation (5 experiments) with reproducible metrics.

Evaluates 5 methods on dynamic obstacle scenarios:
1. A* tracking (global path, no dynamic avoidance)
2. RRT*+APF (local avoidance)
3. PPO Basic
4. Dual-Attention PPO
5. Ours (Full model)

Outputs:
- ablation5_dynamic_metrics.csv
- ablation5_dynamic_metrics.txt
"""

from __future__ import annotations

import csv
import os
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List

import numpy as np
import torch

from experiments_config import (
    MAP_TYPES,
    DYNAMIC_K,
    DYNAMIC_N,
    MAX_STEPS_DYNAMIC,
    get_seeds,
)

# Import actual repository modules
from env import AutonomousNavEnv
from map_generator import MapGenerator
from global_planner import SmartAStarPlanner
from rrt_apf_planner import RRTStarAPFNavigator
from ppo_basic import BasicPPOActorCritic
from ppo_attention import DualAttentionPPOActorCritic
from model import CascadedDualAttentionActorCritic


@dataclass
class Metrics:
    success_count: int = 0
    collision_count: int = 0
    total_count: int = 0
    sum_path_length: float = 0.0

    @property
    def success_rate(self) -> float:
        return self.success_count / self.total_count if self.total_count else 0.0

    @property
    def collision_rate(self) -> float:
        return self.collision_count / self.total_count if self.total_count else 0.0

    @property
    def avg_path_length(self) -> float:
        return self.sum_path_length / self.success_count if self.success_count else 0.0


def get_git_commit_hash():
    """Get current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(__file__)
        )
        return result.stdout.strip()[:8]
    except Exception:
        return "unknown"


def update_dynamic_obstacles(obstacles: list, map_size: int = 80):
    """Update dynamic obstacle positions."""
    for obs in obstacles:
        obs['pos'] = obs['pos'] + obs['vel']
        # Bounce off walls
        if not (1 < obs['pos'][0] < map_size - 1):
            obs['vel'][0] *= -1
        if not (1 < obs['pos'][1] < map_size - 1):
            obs['vel'][1] *= -1


def check_collision_dynamic(agent_pos: np.ndarray, obstacles: list) -> bool:
    """Check if agent collides with any dynamic obstacle."""
    for obs in obstacles:
        dist = np.linalg.norm(agent_pos - obs['pos'])
        if dist < obs.get('radius', 2.0):
            return True
    return False


def run_astar_tracking(map_type: str, seed: int, max_steps: int = 800) -> Dict[str, Any]:
    """Run A* tracking method (no dynamic avoidance)."""
    np.random.seed(seed)
    
    env = AutonomousNavEnv(map_type=map_type)
    env.reset(seed=seed)
    
    # Plan global path with A*
    planner = SmartAStarPlanner(env.static_map)
    global_path = planner.plan(env.start_pos, env.goal_pos)
    
    if global_path is None or len(global_path) == 0:
        global_path = np.array([env.start_pos, env.goal_pos])
    
    # Track the path (simple waypoint following)
    trajectory = [env.agent_pos.copy()]
    wp_idx = 0
    success = False
    collision = False
    
    # Get dynamic obstacles (for concave and narrow maps)
    dyn_obs = []
    for o in env.dynamic_obstacles:
        dyn_obs.append({
            'pos': np.array(o['pos'], dtype=np.float32).copy(),
            'vel': np.array(o.get('vel', [0.2, 0.2]), dtype=np.float32).copy(),
            'radius': float(o.get('radius', 2.0)),
        })
    
    for _ in range(max_steps):
        # Follow waypoints
        while wp_idx < len(global_path) - 1 and np.linalg.norm(env.agent_pos - global_path[wp_idx]) < 2.0:
            wp_idx += 1
        target = global_path[min(wp_idx, len(global_path) - 1)]
        
        # Compute direction to target
        desired_angle = np.arctan2(target[1] - env.agent_pos[1], target[0] - env.agent_pos[0])
        angle_diff = desired_angle - env.agent_dir
        angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi
        
        action = np.array([1.0, float(np.clip(angle_diff, -1.0, 1.0))], dtype=np.float32)
        _, _, done, _, info = env.step(action)
        trajectory.append(env.agent_pos.copy())
        
        # Update dynamic obstacles
        update_dynamic_obstacles(dyn_obs)
        
        # Check collision with dynamic obstacles (for concave and narrow)
        if map_type in ['concave', 'narrow']:
            if check_collision_dynamic(env.agent_pos, dyn_obs):
                collision = True
                done = True
        
        # Check static collision from env
        if info.get('collision', False):
            collision = True
        
        if done:
            success = info.get('success', False)
            break
    
    # Calculate path length
    path_len = sum(np.linalg.norm(trajectory[i+1] - trajectory[i]) 
                   for i in range(len(trajectory)-1))
    
    return {
        'success': success,
        'collision': collision,
        'path_length': path_len,
    }


def run_rrt_apf(map_type: str, seed: int, max_steps: int = 800) -> Dict[str, Any]:
    """Run RRT*+APF method."""
    np.random.seed(seed)
    
    mg = MapGenerator(80)
    grid, start, goal = mg.get_map(map_type)
    
    # Create navigator
    try:
        navigator = RRTStarAPFNavigator(grid, start, goal)
    except Exception as e:
        return {'success': False, 'collision': True, 'path_length': 0.0}
    
    # Get dynamic obstacles
    dyn_obs = []
    for o in (mg.get_dynamic_obstacles(map_type) or []):
        dyn_obs.append({
            'pos': np.array(o['pos'], dtype=np.float32).copy(),
            'vel': np.array(o.get('vel', [0.2, 0.2]), dtype=np.float32).copy(),
            'radius': float(o.get('radius', 2.0)),
        })
    
    trajectory = [navigator.pos.copy()]
    success = False
    collision = False
    
    for _ in range(max_steps):
        navigator.set_dynamic_obstacles(dyn_obs)
        pos, done, info = navigator.step()
        trajectory.append(pos.copy())
        
        # Update dynamic obstacles
        update_dynamic_obstacles(dyn_obs)
        
        # Check collision with dynamic obstacles
        if map_type in ['concave', 'narrow']:
            if check_collision_dynamic(pos, dyn_obs):
                collision = True
                done = True
        
        if done:
            success = info.get('success', False)
            collision = collision or info.get('collision', False)
            break
    
    path_len = sum(np.linalg.norm(trajectory[i+1] - trajectory[i]) 
                   for i in range(len(trajectory)-1))
    
    return {
        'success': success,
        'collision': collision,
        'path_length': path_len,
    }


def run_ppo_model(model: torch.nn.Module, map_type: str, seed: int, 
                  use_astar: bool = True, max_steps: int = 800) -> Dict[str, Any]:
    """Run PPO-based model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    np.random.seed(seed)
    env = AutonomousNavEnv(map_type=map_type)
    obs, _ = env.reset(seed=seed)
    
    # A* global path (if enabled)
    if use_astar:
        planner = SmartAStarPlanner(env.static_map)
        global_path = planner.plan(env.start_pos, env.goal_pos)
        if global_path is not None and len(global_path) > 0:
            env.set_global_path(global_path)
    
    # Get dynamic obstacles
    dyn_obs = []
    for o in env.dynamic_obstacles:
        dyn_obs.append({
            'pos': np.array(o['pos'], dtype=np.float32).copy(),
            'vel': np.array(o.get('vel', [0.2, 0.2]), dtype=np.float32).copy(),
            'radius': float(o.get('radius', 2.0)),
        })
    
    hidden_state = torch.zeros(1, 256).to(device)
    trajectory = [env.agent_pos.copy()]
    success = False
    collision = False
    
    with torch.no_grad():
        for _ in range(max_steps):
            img = torch.FloatTensor(obs['image']).unsqueeze(0).to(device)
            vec = torch.FloatTensor(obs['vector']).unsqueeze(0).to(device)
            
            mu, std, _, hidden_state = model({'image': img, 'vector': vec}, hidden_state)
            action = mu.cpu().numpy().flatten()
            
            obs, _, done, _, info = env.step(action.astype(np.float32))
            trajectory.append(env.agent_pos.copy())
            
            # Update dynamic obstacles
            update_dynamic_obstacles(dyn_obs)
            
            # Check collision with dynamic obstacles
            if map_type in ['concave', 'narrow']:
                if check_collision_dynamic(env.agent_pos, dyn_obs):
                    collision = True
                    done = True
            
            if done:
                success = info.get('success', False)
                collision = collision or info.get('collision', False)
                break
    
    path_len = sum(np.linalg.norm(trajectory[i+1] - trajectory[i]) 
                   for i in range(len(trajectory)-1))
    
    return {
        'success': success,
        'collision': collision,
        'path_length': path_len,
    }


def main() -> None:
    print("=" * 80)
    print("5-Method Dynamic Obstacle Ablation Evaluation")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Define the 5 methods
    methods = [
        'A* tracking',
        'RRT*+APF',
        'PPO Basic',
        'Dual-Att PPO',
        'Ours (Full)',
    ]
    
    # Load PPO models
    models = {}
    checkpoint_paths = {
        'PPO Basic': 'models_basic_astar/model.pth',
        'Dual-Att PPO': 'models_attention_noastar/model.pth',
        'Ours (Full)': 'best_model.pth',
    }
    
    for method_name, ckpt_path in checkpoint_paths.items():
        if method_name == 'PPO Basic':
            model = BasicPPOActorCritic(action_dim=2)
        elif method_name == 'Dual-Att PPO':
            model = DualAttentionPPOActorCritic(action_dim=2)
        else:  # Ours (Full)
            model = CascadedDualAttentionActorCritic(action_dim=2)
        
        if os.path.exists(ckpt_path):
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
            print(f"Loaded checkpoint: {ckpt_path}")
        else:
            print(f"WARNING: Checkpoint not found: {ckpt_path}")
            print(f"  Using random initialization for {method_name}")
            print(f"  To train: python train_ablation.py --model <model_type> --episodes 3000")
        
        models[method_name] = model
    
    # Get metadata
    git_hash = get_git_commit_hash()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    rows: List[Dict[str, Any]] = []
    txt_lines: List[str] = []
    txt_lines.append("=" * 80)
    txt_lines.append("5-Method Dynamic Obstacle Ablation Evaluation")
    txt_lines.append("=" * 80)
    txt_lines.append(f"Timestamp: {timestamp}")
    txt_lines.append(f"Git commit: {git_hash}")
    txt_lines.append(f"K={DYNAMIC_K} scenarios/map | N={DYNAMIC_N} runs/scenario | max_steps={MAX_STEPS_DYNAMIC}")
    txt_lines.append(f"Checkpoint paths: {checkpoint_paths}")
    txt_lines.append("=" * 80)
    
    for map_type in MAP_TYPES:
        print(f"\n{'='*80}")
        print(f"Map type: {map_type}")
        print(f"{'='*80}")
        txt_lines.append(f"\n=== Map: {map_type} ===")
        
        # Deterministic scenario seeds per map
        scenario_base = 20_000 + MAP_TYPES.index(map_type) * 2_000
        scenario_seeds = get_seeds(scenario_base, DYNAMIC_K)
        
        for method in methods:
            print(f"  Method: {method}")
            m = Metrics()
            
            for s_idx, scenario_seed in enumerate(scenario_seeds):
                run_seeds = get_seeds(80_000 + MAP_TYPES.index(map_type) * 10_000 + s_idx * 100, DYNAMIC_N)
                
                for run_seed in run_seeds:
                    # Run the appropriate method
                    if method == 'A* tracking':
                        result = run_astar_tracking(map_type, run_seed, MAX_STEPS_DYNAMIC)
                    elif method == 'RRT*+APF':
                        result = run_rrt_apf(map_type, run_seed, MAX_STEPS_DYNAMIC)
                    elif method == 'PPO Basic':
                        result = run_ppo_model(models['PPO Basic'], map_type, run_seed, 
                                              use_astar=True, max_steps=MAX_STEPS_DYNAMIC)
                    elif method == 'Dual-Att PPO':
                        result = run_ppo_model(models['Dual-Att PPO'], map_type, run_seed,
                                              use_astar=False, max_steps=MAX_STEPS_DYNAMIC)
                    else:  # Ours (Full)
                        result = run_ppo_model(models['Ours (Full)'], map_type, run_seed,
                                              use_astar=True, max_steps=MAX_STEPS_DYNAMIC)
                    
                    success = result['success']
                    collision = result['collision']
                    path_len = result['path_length']
                    
                    m.total_count += 1
                    if success:
                        m.success_count += 1
                        m.sum_path_length += float(path_len)
                    if collision:
                        m.collision_count += 1
            
            row = {
                'timestamp': timestamp,
                'git_commit': git_hash,
                'map_type': map_type,
                'method': method,
                'K': DYNAMIC_K,
                'N': DYNAMIC_N,
                'max_steps': MAX_STEPS_DYNAMIC,
                'success_count': m.success_count,
                'collision_count': m.collision_count,
                'total': m.total_count,
                'success_rate': m.success_rate,
                'collision_rate': m.collision_rate,
                'avg_path_length': m.avg_path_length,
            }
            rows.append(row)
            
            print(f"    success: {m.success_count}/{m.total_count} ({m.success_rate:.3f})")
            print(f"    collision: {m.collision_count}/{m.total_count} ({m.collision_rate:.3f})")
            print(f"    avg_path_len: {m.avg_path_length:.3f}")
            
            txt_lines.append(
                f"{method:>15s} | success {m.success_count:3d}/{m.total_count:3d} (rate={m.success_rate:.3f})"
                f" | collision {m.collision_count:3d}/{m.total_count:3d} (rate={m.collision_rate:.3f})"
                f" | avg_path_len={m.avg_path_length:7.3f}"
            )
    
    # Save outputs
    csv_path = os.path.join(os.path.dirname(__file__), "ablation5_dynamic_metrics.csv")
    txt_path = os.path.join(os.path.dirname(__file__), "ablation5_dynamic_metrics.txt")
    
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    
    with open(txt_path, "w") as f:
        f.write("\n".join(txt_lines) + "\n")
    
    print(f"\n{'='*80}")
    print(f"Results saved to:")
    print(f"  {csv_path}")
    print(f"  {txt_path}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
