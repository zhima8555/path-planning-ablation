"""Dynamic-obstacle ablation metrics (for IEEE-style tables/figures).

Evaluates on:
- concave: 1 dynamic obstacle
- narrow:  2 dynamic obstacles
Dynamic obstacles are fixed & reproducible via MapGenerator.get_dynamic_obstacles().

Baselines:
- A* tracking (no dynamic avoidance)
- RRT* + APF (local avoidance)
- Ours (trained policy in best_navigation_model.pth)

Outputs (in project root):
- dynamic_obstacle_metrics.csv
- dynamic_obstacle_metrics.txt
- dynamic_obstacle_metrics.png

Note: This script reports *measured* results from your code (no fabricated numbers).
"""

from __future__ import annotations

import os
import sys
import time
import csv
from dataclasses import dataclass

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch

# Ensure project root is importable when running from ablation/.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from env import AutonomousNavEnv
from map_generator import MapGenerator
from global_planner import SmartAStarPlanner
from model import CascadedDualAttentionActorCritic

from ablation.rrt_apf_planner import RRTStarAPFNavigator


@dataclass
class EpisodeResult:
    map_type: str
    algo: str
    seed: int
    success: bool
    collision: bool
    steps: int
    path_length: float
    wall_time_ms: float


def path_length(traj: np.ndarray) -> float:
    if traj is None or len(traj) < 2:
        return 0.0
    diffs = np.diff(traj, axis=0)
    return float(np.sum(np.linalg.norm(diffs, axis=1)))


def update_dynamic_obstacles(obstacles: list[dict], map_size: int):
    for obs in obstacles:
        obs['pos'] = obs['pos'] + obs['vel']
        if not (1 < obs['pos'][0] < map_size - 1):
            obs['vel'][0] *= -1
        if not (1 < obs['pos'][1] < map_size - 1):
            obs['vel'][1] *= -1


def run_astar_tracking(map_type: str, seed: int, max_steps: int = 500) -> EpisodeResult:
    np.random.seed(seed)
    env = AutonomousNavEnv(map_type=map_type)

    # force deterministic reset noise
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

    t1 = time.perf_counter()
    for _ in range(max_steps):
        while wp_idx < len(global_path) - 1 and np.linalg.norm(env.agent_pos - global_path[wp_idx]) < 2.0:
            wp_idx += 1
        target = global_path[min(wp_idx, len(global_path) - 1)]

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
    t_roll = time.perf_counter() - t1

    wall_ms = (t_plan + t_roll) * 1000.0
    return EpisodeResult(
        map_type=map_type,
        algo='A* tracking',
        seed=seed,
        success=success,
        collision=collision,
        steps=len(traj),
        path_length=path_length(np.array(traj)),
        wall_time_ms=wall_ms,
    )


def run_rrt_apf(map_type: str, seed: int, max_steps: int = 800) -> EpisodeResult:
    np.random.seed(seed)
    mg = MapGenerator(80)
    grid, start, goal = mg.get_map(map_type)

    t0 = time.perf_counter()
    nav = RRTStarAPFNavigator(grid, start, goal)
    t_plan = time.perf_counter() - t0

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

    t1 = time.perf_counter()
    for _ in range(max_steps):
        nav.set_dynamic_obstacles(dyn)
        pos, done, info = nav.step()
        traj.append(pos.copy())

        update_dynamic_obstacles(dyn, 80)

        if done:
            success = bool(info.get('success', False))
            collision = bool(info.get('collision', False))
            break
    t_roll = time.perf_counter() - t1

    wall_ms = (t_plan + t_roll) * 1000.0
    return EpisodeResult(
        map_type=map_type,
        algo='RRT*+APF',
        seed=seed,
        success=success,
        collision=collision,
        steps=len(traj),
        path_length=path_length(np.array(traj)),
        wall_time_ms=wall_ms,
    )


class OursPolicy:
    def __init__(self, ckpt_path: str = 'best_navigation_model.pth', device: str | None = None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        self.model = CascadedDualAttentionActorCritic(action_dim=2).to(self.device)
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
        self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device))
        self.model.eval()

    @torch.no_grad()
    def act(self, obs: dict, hidden: torch.Tensor):
        img = torch.as_tensor(obs['image'], dtype=torch.float32, device=self.device).unsqueeze(0)
        vec = torch.as_tensor(obs['vector'], dtype=torch.float32, device=self.device).unsqueeze(0)
        mu, std, val, next_hidden = self.model({'image': img, 'vector': vec}, hidden)
        action = mu.squeeze(0).detach().cpu().numpy()
        return action, next_hidden


def run_ours(map_type: str, seed: int, policy: OursPolicy, max_steps: int = 500) -> EpisodeResult:
    np.random.seed(seed)
    env = AutonomousNavEnv(map_type=map_type)
    obs, _ = env.reset(seed=seed)

    planner = SmartAStarPlanner(env.static_map)
    t0 = time.perf_counter()
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

    t1 = time.perf_counter()
    for _ in range(max_steps):
        action, hidden = policy.act(obs, hidden)
        obs, _, done, _, info = env.step(action.astype(np.float32))
        traj.append(env.agent_pos.copy())
        if done:
            success = bool(info.get('success', False))
            collision = bool(info.get('collision', False))
            break
    t_roll = time.perf_counter() - t1

    wall_ms = (t_plan + t_roll) * 1000.0
    return EpisodeResult(
        map_type=map_type,
        algo='Ours',
        seed=seed,
        success=success,
        collision=collision,
        steps=len(traj),
        path_length=path_length(np.array(traj)),
        wall_time_ms=wall_ms,
    )


def aggregate(results: list[EpisodeResult]):
    # group by (map, algo)
    groups: dict[tuple[str, str], list[EpisodeResult]] = {}
    for r in results:
        groups.setdefault((r.map_type, r.algo), []).append(r)

    summary = []
    for (mt, algo), rows in sorted(groups.items()):
        n = len(rows)
        success_rate = 100.0 * sum(1 for r in rows if r.success) / n
        collision_rate = 100.0 * sum(1 for r in rows if r.collision) / n
        avg_len = float(np.mean([r.path_length for r in rows]))
        avg_time = float(np.mean([r.wall_time_ms for r in rows]))
        avg_steps = float(np.mean([r.steps for r in rows]))
        summary.append({
            'map': mt,
            'algo': algo,
            'n': n,
            'success_rate': success_rate,
            'collision_rate': collision_rate,
            'path_length': avg_len,
            'wall_time_ms': avg_time,
            'steps': avg_steps,
        })
    return summary


def save_csv(path: str, rows: list[dict]):
    if not rows:
        return
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def save_txt(path: str, rows: list[dict]):
    lines = []
    lines.append('Dynamic obstacle metrics (measured)')
    lines.append('')
    hdr = f"{'Map':<8} {'Algo':<12} {'N':>3} {'Succ%':>7} {'Coll%':>7} {'Len':>10} {'Time(ms)':>10} {'Steps':>8}"
    lines.append(hdr)
    lines.append('-' * len(hdr))
    for r in rows:
        lines.append(
            f"{r['map']:<8} {r['algo']:<12} {r['n']:>3} {r['success_rate']:>6.1f} {r['collision_rate']:>6.1f} "
            f"{r['path_length']:>10.2f} {r['wall_time_ms']:>10.2f} {r['steps']:>8.1f}"
        )
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def plot_summary(path: str, summary: list[dict]):
    maps = ['concave', 'narrow']
    algos = ['A* tracking', 'RRT*+APF', 'Ours']

    # build matrices
    def mat(key):
        out = np.zeros((len(maps), len(algos)), dtype=np.float32)
        for i, mt in enumerate(maps):
            for j, algo in enumerate(algos):
                for r in summary:
                    if r['map'] == mt and r['algo'] == algo:
                        out[i, j] = float(r[key])
        return out

    succ = mat('success_rate')
    coll = mat('collision_rate')
    plen = mat('path_length')
    tms = mat('wall_time_ms')

    plt.rcParams['font.family'] = 'Times New Roman'
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    x = np.arange(len(maps))
    width = 0.22
    colors = {
        'A* tracking': '#2980B9',
        'RRT*+APF': '#C0392B',
        'Ours': '#8E44AD',
    }

    def bars(ax, data, title, ylim=None):
        for j, algo in enumerate(algos):
            ax.bar(x + (j - 1) * width, data[:, j], width, label=algo, color=colors[algo], alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(['Concave', 'Narrow'])
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.grid(axis='y', alpha=0.25, linestyle='--')
        if ylim is not None:
            ax.set_ylim(ylim)

    bars(axes[0], succ, 'Success Rate (%)', ylim=(0, 105))
    bars(axes[1], coll, 'Collision Rate (%)', ylim=(0, 105))
    bars(axes[2], plen, 'Path Length')
    bars(axes[3], tms, 'Wall Time (ms)')

    axes[0].legend(loc='lower left', fontsize=8)

    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def main():
    # Keep N modest to run quickly; adjust if you want more confidence.
    episodes_per_case = 20
    seeds = list(range(1000, 1000 + episodes_per_case))

    policy = OursPolicy('best_navigation_model.pth')

    results: list[EpisodeResult] = []
    for map_type in ['concave', 'narrow']:
        for seed in seeds:
            results.append(run_astar_tracking(map_type, seed))
            results.append(run_rrt_apf(map_type, seed))
            results.append(run_ours(map_type, seed, policy))

    summary = aggregate(results)

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    out_csv = os.path.join(root, 'dynamic_obstacle_metrics.csv')
    out_txt = os.path.join(root, 'dynamic_obstacle_metrics.txt')
    out_png = os.path.join(root, 'dynamic_obstacle_metrics.png')

    save_csv(out_csv, summary)
    save_txt(out_txt, summary)
    plot_summary(out_png, summary)

    print('Saved:', out_csv)
    print('Saved:', out_txt)
    print('Saved:', out_png)


if __name__ == '__main__':
    main()
