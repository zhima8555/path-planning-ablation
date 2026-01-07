"""Dynamic evaluation of 5 methods with reproducible metrics.

Evaluates 5 methods across 4 map types with dynamic obstacles:
1. A* tracking (no dynamic avoidance)
2. RRT* + APF (local avoidance)
3. PPO (Basic) - trained with A* guidance
4. PPO (Dual-Att) - trained without A* guidance
5. Ours (Full) - A* + Dual-Att PPO

Protocol:
- K=20 scenarios per map
- N=10 runs per scenario
- Total: K*N = 200 episodes per (map, method)
- Reproducible seeds for deterministic evaluation

Outputs (in repository root):
- ablation5_dynamic_metrics.csv
- ablation5_dynamic_metrics.txt
"""

from __future__ import annotations

import csv
import os
import sys
import time
import subprocess
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Any, Optional

import numpy as np
import torch

# Ensure parent path for imports
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from env import AutonomousNavEnv
from map_generator import MapGenerator
from global_planner import SmartAStarPlanner
from model import CascadedDualAttentionActorCritic

from experiments_config import MAP_TYPES, DYNAMIC_K, DYNAMIC_N, MAX_STEPS_DYNAMIC, get_seeds
from ppo_basic import BasicPPOActorCritic
from ppo_attention import DualAttentionPPOActorCritic
from rrt_apf_planner import RRTStarAPFNavigator


@dataclass
class EpisodeResult:
    map_type: str
    method: str
    scenario_idx: int
    run_idx: int
    seed: int
    success: bool
    collision: bool
    timeout: bool
    steps: int
    path_length_m: float
    wall_time_ms: float


def path_length(traj: np.ndarray) -> float:
    """Compute path length in meters."""
    if traj is None or len(traj) < 2:
        return 0.0
    diffs = np.diff(traj, axis=0)
    return float(np.sum(np.linalg.norm(diffs, axis=1)))


def run_astar_tracking(map_type: str, seed: int, max_steps: int) -> EpisodeResult:
    """Run A* tracking (no dynamic avoidance)."""
    np.random.seed(seed)
    env = AutonomousNavEnv(map_type=map_type)
    env.reset(seed=seed)

    planner = SmartAStarPlanner(env.static_map)
    t0 = time.perf_counter()
    global_path = planner.plan(env.start_pos, env.goal_pos, env.dynamic_obstacles)
    t_plan = time.perf_counter() - t0

    if global_path is None or len(global_path) == 0:
        global_path = np.array([env.start_pos, env.goal_pos], dtype=np.float32)

    env.set_global_path(global_path)
    traj = [env.agent_pos.copy()]

    wp_idx = 0
    success = False
    collision = False
    timeout = False

    t1 = time.perf_counter()
    for step in range(max_steps):
        # Track waypoint
        while wp_idx < len(global_path) - 1 and np.linalg.norm(env.agent_pos - global_path[wp_idx]) < 2.0:
            wp_idx += 1
        target = global_path[min(wp_idx, len(global_path) - 1)]

        # Compute action
        desired = np.arctan2(target[1] - env.agent_pos[1], target[0] - env.agent_pos[0])
        ang = desired - env.agent_dir
        ang = (ang + np.pi) % (2 * np.pi) - np.pi
        action = np.array([1.0, float(np.clip(ang, -1.0, 1.0))], dtype=np.float32)

        _, _, done, _, info = env.step(action)
        traj.append(env.agent_pos.copy())
        if done:
            success = bool(info.get('success', False))
            collision = bool(info.get('collision', False))
            break
    else:
        timeout = True

    t_roll = time.perf_counter() - t1
    wall_ms = (t_plan + t_roll) * 1000.0

    return EpisodeResult(
        map_type=map_type,
        method='A* tracking',
        scenario_idx=-1,
        run_idx=-1,
        seed=seed,
        success=success,
        collision=collision,
        timeout=timeout,
        steps=len(traj),
        path_length_m=path_length(np.array(traj)),
        wall_time_ms=wall_ms,
    )


def run_rrt_apf(map_type: str, seed: int, max_steps: int) -> EpisodeResult:
    """Run RRT*+APF method."""
    np.random.seed(seed)
    mg = MapGenerator(80)
    grid, start, goal = mg.get_map(map_type)

    t0 = time.perf_counter()
    try:
        nav = RRTStarAPFNavigator(grid, start, goal)
    except Exception:
        # Planning failed
        return EpisodeResult(
            map_type=map_type,
            method='RRT*+APF',
            scenario_idx=-1,
            run_idx=-1,
            seed=seed,
            success=False,
            collision=False,
            timeout=False,
            steps=0,
            path_length_m=0.0,
            wall_time_ms=0.0,
        )
    t_plan = time.perf_counter() - t0

    # Setup dynamic obstacles
    dyn = []
    for o in (mg.get_dynamic_obstacles(map_type) or []):
        dyn.append({
            'pos': np.array(o['pos'], dtype=np.float32).copy(),
            'vel': np.array(o.get('vel', [0.2, 0.2]), dtype=np.float32).copy(),
            'radius': float(o.get('radius', 2.0)),
        })

    traj = [nav.pos.copy()]
    success = False
    collision = False
    timeout = False

    t1 = time.perf_counter()
    for step in range(max_steps):
        nav.set_dynamic_obstacles(dyn)
        result = nav.step()
        if result is None:
            break
        pos, done, info = result
        traj.append(pos.copy())

        # Update dynamic obstacles
        for obs in dyn:
            obs['pos'] = obs['pos'] + obs['vel']
            if not (1 < obs['pos'][0] < 79):
                obs['vel'][0] *= -1
            if not (1 < obs['pos'][1] < 79):
                obs['vel'][1] *= -1

        if done:
            success = bool(info.get('success', False))
            collision = bool(info.get('collision', False))
            break
    else:
        timeout = True

    t_roll = time.perf_counter() - t1
    wall_ms = (t_plan + t_roll) * 1000.0

    return EpisodeResult(
        map_type=map_type,
        method='RRT*+APF',
        scenario_idx=-1,
        run_idx=-1,
        seed=seed,
        success=success,
        collision=collision,
        timeout=timeout,
        steps=len(traj),
        path_length_m=path_length(np.array(traj)),
        wall_time_ms=wall_ms,
    )


class PPOPolicy:
    """Generic PPO policy wrapper."""

    def __init__(self, model, device: Optional[str] = None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def act(self, obs: dict, hidden: torch.Tensor):
        img = torch.as_tensor(obs['image'], dtype=torch.float32, device=self.device).unsqueeze(0)
        vec = torch.as_tensor(obs['vector'], dtype=torch.float32, device=self.device).unsqueeze(0)
        mu, std, val, next_hidden = self.model({'image': img, 'vector': vec}, hidden)
        action = mu.squeeze(0).detach().cpu().numpy()
        return action, next_hidden


def run_ppo_method(
    map_type: str,
    seed: int,
    policy: PPOPolicy,
    method_name: str,
    use_astar: bool,
    max_steps: int,
) -> EpisodeResult:
    """Run a PPO-based method."""
    np.random.seed(seed)
    env = AutonomousNavEnv(map_type=map_type)
    obs, _ = env.reset(seed=seed)

    t0 = time.perf_counter()
    if use_astar:
        planner = SmartAStarPlanner(env.static_map)
        try:
            gp = planner.plan(env.start_pos, env.goal_pos, env.dynamic_obstacles)
            if gp is not None and len(gp) > 0:
                env.set_global_path(gp)
        except Exception:
            pass
    t_plan = time.perf_counter() - t0

    hidden = torch.zeros(1, 256, device=policy.device)
    traj = [env.agent_pos.copy()]
    success = False
    collision = False
    timeout = False

    t1 = time.perf_counter()
    for step in range(max_steps):
        action, hidden = policy.act(obs, hidden)
        obs, _, done, _, info = env.step(action.astype(np.float32))
        traj.append(env.agent_pos.copy())
        if done:
            success = bool(info.get('success', False))
            collision = bool(info.get('collision', False))
            break
    else:
        timeout = True

    t_roll = time.perf_counter() - t1
    wall_ms = (t_plan + t_roll) * 1000.0

    return EpisodeResult(
        map_type=map_type,
        method=method_name,
        scenario_idx=-1,
        run_idx=-1,
        seed=seed,
        success=success,
        collision=collision,
        timeout=timeout,
        steps=len(traj),
        path_length_m=path_length(np.array(traj)),
        wall_time_ms=wall_ms,
    )


def aggregate_results(results: List[EpisodeResult]) -> List[Dict[str, Any]]:
    """Aggregate results by (map_type, method)."""
    groups: Dict[tuple, List[EpisodeResult]] = {}
    for r in results:
        key = (r.map_type, r.method)
        groups.setdefault(key, []).append(r)

    summary = []
    for (mt, method), rows in sorted(groups.items()):
        total = len(rows)
        success_count = sum(1 for r in rows if r.success)
        collision_count = sum(1 for r in rows if r.collision)
        timeout_count = sum(1 for r in rows if r.timeout)
        
        success_rate = success_count / total if total else 0.0
        collision_rate = collision_count / total if total else 0.0
        
        # Average path length over successful episodes
        successful_paths = [r.path_length_m for r in rows if r.success]
        avg_path_length_m = float(np.mean(successful_paths)) if successful_paths else 0.0
        
        # Average wall time
        avg_wall_time_ms = float(np.mean([r.wall_time_ms for r in rows]))

        summary.append({
            'map_type': mt,
            'method': method,
            'K': DYNAMIC_K,
            'N': DYNAMIC_N,
            'max_steps': MAX_STEPS_DYNAMIC,
            'success_count': success_count,
            'collision_count': collision_count,
            'timeout_count': timeout_count,
            'total': total,
            'success_rate': success_rate,
            'collision_rate': collision_rate,
            'avg_path_length_m': avg_path_length_m,
            'avg_wall_time_ms': avg_wall_time_ms,
        })
    
    return summary


def get_git_commit() -> str:
    """Get current git commit hash."""
    try:
        return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], 
                                      stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return 'unknown'


def save_outputs(summary: List[Dict[str, Any]], checkpoint_info: Dict[str, str]):
    """Save CSV and TXT outputs."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    git_commit = get_git_commit()

    # CSV
    csv_path = 'ablation5_dynamic_metrics.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        if summary:
            fieldnames = list(summary[0].keys()) + ['timestamp', 'git_commit']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in summary:
                row_copy = row.copy()
                row_copy['timestamp'] = timestamp
                row_copy['git_commit'] = git_commit
                writer.writerow(row_copy)

    # TXT
    txt_path = 'ablation5_dynamic_metrics.txt'
    lines = []
    lines.append('=' * 100)
    lines.append('DYNAMIC ABLATION STUDY - 5 METHODS')
    lines.append('=' * 100)
    lines.append(f'Timestamp: {timestamp}')
    lines.append(f'Git commit: {git_commit}')
    lines.append(f'Protocol: K={DYNAMIC_K} scenarios/map, N={DYNAMIC_N} runs/scenario, max_steps={MAX_STEPS_DYNAMIC}')
    lines.append('')
    lines.append('Checkpoint paths:')
    for method, path in checkpoint_info.items():
        lines.append(f'  {method}: {path}')
    lines.append('')
    lines.append('Results:')
    lines.append('')

    hdr = (f"{'Map':<10} {'Method':<20} {'Succ':>6} {'Coll':>6} {'Timeout':>7} {'Total':>5} "
           f"{'SuccRate':>8} {'CollRate':>8} {'AvgLen(m)':>10} {'AvgTime(ms)':>12}")
    lines.append(hdr)
    lines.append('-' * len(hdr))

    for row in summary:
        lines.append(
            f"{row['map_type']:<10} {row['method']:<20} {row['success_count']:>6} "
            f"{row['collision_count']:>6} {row['timeout_count']:>7} {row['total']:>5} "
            f"{row['success_rate']:>8.3f} {row['collision_rate']:>8.3f} "
            f"{row['avg_path_length_m']:>10.2f} {row['avg_wall_time_ms']:>12.2f}"
        )

    lines.append('=' * 100)

    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')

    print(f'Saved: {csv_path}')
    print(f'Saved: {txt_path}')


def load_or_create_policy(model_class, checkpoint_path: str, device: str) -> PPOPolicy:
    """Load a policy from checkpoint or create with random weights."""
    model = model_class(action_dim=2)
    if os.path.exists(checkpoint_path):
        print(f'  Loading checkpoint: {checkpoint_path}')
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    else:
        print(f'  Warning: Checkpoint not found: {checkpoint_path}')
        print(f'  Using random initialization (results will be poor!)')
    return PPOPolicy(model, device=device)


def main():
    """Main evaluation function."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')
    print('=' * 100)
    print('DYNAMIC ABLATION EVALUATION - 5 METHODS')
    print('=' * 100)

    # Define checkpoint paths
    checkpoint_info = {
        'PPO (Basic)': 'models_basic_astar/model.pth',
        'PPO (Dual-Att)': 'models_attention_noastar/model.pth',
        'Ours (Full)': 'best_navigation_model.pth',
    }

    # Load PPO policies
    print('\nLoading models...')
    ppo_basic_policy = load_or_create_policy(BasicPPOActorCritic, checkpoint_info['PPO (Basic)'], device)
    ppo_attention_policy = load_or_create_policy(DualAttentionPPOActorCritic, checkpoint_info['PPO (Dual-Att)'], device)
    ours_policy = load_or_create_policy(CascadedDualAttentionActorCritic, checkpoint_info['Ours (Full)'], device)

    # Run evaluations
    results: List[EpisodeResult] = []
    
    for map_type in MAP_TYPES:
        print(f'\nEvaluating on map: {map_type}')
        
        # Generate scenario seeds
        scenario_base = 20_000 + MAP_TYPES.index(map_type) * 2_000
        scenario_seeds = get_seeds(scenario_base, DYNAMIC_K)
        
        for s_idx, scenario_seed in enumerate(scenario_seeds):
            # Generate run seeds
            run_base = 80_000 + MAP_TYPES.index(map_type) * 10_000 + s_idx * 100
            run_seeds = get_seeds(run_base, DYNAMIC_N)
            
            for r_idx, run_seed in enumerate(run_seeds):
                # Run all 5 methods
                result = run_astar_tracking(map_type, run_seed, MAX_STEPS_DYNAMIC)
                result.scenario_idx = s_idx
                result.run_idx = r_idx
                results.append(result)
                
                result = run_rrt_apf(map_type, run_seed, MAX_STEPS_DYNAMIC)
                result.scenario_idx = s_idx
                result.run_idx = r_idx
                results.append(result)
                
                result = run_ppo_method(map_type, run_seed, ppo_basic_policy, 
                                       'PPO (Basic)', use_astar=True, max_steps=MAX_STEPS_DYNAMIC)
                result.scenario_idx = s_idx
                result.run_idx = r_idx
                results.append(result)
                
                result = run_ppo_method(map_type, run_seed, ppo_attention_policy,
                                       'PPO (Dual-Att)', use_astar=False, max_steps=MAX_STEPS_DYNAMIC)
                result.scenario_idx = s_idx
                result.run_idx = r_idx
                results.append(result)
                
                result = run_ppo_method(map_type, run_seed, ours_policy,
                                       'Ours (Full)', use_astar=True, max_steps=MAX_STEPS_DYNAMIC)
                result.scenario_idx = s_idx
                result.run_idx = r_idx
                results.append(result)
            
            print(f'  Completed scenario {s_idx + 1}/{DYNAMIC_K}')

    # Aggregate and save
    print('\nAggregating results...')
    summary = aggregate_results(results)
    save_outputs(summary, checkpoint_info)
    
    print('\nEvaluation complete!')


if __name__ == '__main__':
    main()
