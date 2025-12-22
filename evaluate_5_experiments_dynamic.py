"""Evaluate 5 experiments (5 methods) with dynamic obstacles on Map-3/4.

Methods (5):
1) A* tracking
2) RRT* + APF
3) PPO (Basic)
4) PPO (Dual-Att)
5) Ours (Full: A* + Dual-Att PPO)

Maps:
- simple, complex (no fixed dynamic obstacles)
- concave (1 fixed dynamic obstacle)
- narrow  (2 fixed dynamic obstacles)

Metrics (paper-friendly):
- Success Rate (%), Collision Rate (%)
- Path Length
- Wall Time (ms)  [planning + rollout]

Outputs (project root):
- ablation5_dynamic_metrics.csv
- ablation5_dynamic_metrics.txt
- ablation5_dynamic_metrics.png

This script does NOT fabricate results; it measures from your code.
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

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from env import AutonomousNavEnv
from map_generator import MapGenerator
from global_planner import SmartAStarPlanner
from model import CascadedDualAttentionActorCritic

from ablation.rrt_apf_planner import RRTStarAPFNavigator
from ablation.ppo_basic import BasicPPOActorCritic
from ablation.ppo_attention import DualAttentionPPOActorCritic


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


def traj_length(traj: list[np.ndarray]) -> float:
    if len(traj) < 2:
        return 0.0
    arr = np.array(traj, dtype=np.float32)
    diffs = np.diff(arr, axis=0)
    return float(np.sum(np.linalg.norm(diffs, axis=1)))


def update_dynamic_obstacles(obstacles: list[dict], map_size: int):
    for obs in obstacles:
        obs['pos'] = obs['pos'] + obs['vel']
        if not (1 < obs['pos'][0] < map_size - 1):
            obs['vel'][0] *= -1
        if not (1 < obs['pos'][1] < map_size - 1):
            obs['vel'][1] *= -1


class PolicyWrapper:
    def __init__(self, model: torch.nn.Module, ckpt_path: str, device: torch.device):
        self.device = device
        self.model = model.to(device)
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(ckpt_path)
        self.model.load_state_dict(torch.load(ckpt_path, map_location=device))
        self.model.eval()

    @torch.no_grad()
    def act(self, obs: dict, hidden: torch.Tensor):
        img = torch.as_tensor(obs['image'], dtype=torch.float32, device=self.device).unsqueeze(0)
        vec = torch.as_tensor(obs['vector'], dtype=torch.float32, device=self.device).unsqueeze(0)
        mu, std, val, next_hidden = self.model({'image': img, 'vector': vec}, hidden)
        action = mu.squeeze(0).detach().cpu().numpy().astype(np.float32)
        return action, next_hidden


def plan_and_set_global_path(env: AutonomousNavEnv):
    planner = SmartAStarPlanner(env.static_map)
    gp = planner.plan(env.start_pos, env.goal_pos, env.dynamic_obstacles)
    env.set_global_path(gp)
    return gp


def run_policy(map_type: str, seed: int, algo: str, policy: PolicyWrapper, max_steps: int = 800) -> EpisodeResult:
    np.random.seed(seed)
    obs_env = AutonomousNavEnv(map_type=map_type)
    obs, _ = obs_env.reset(seed=seed)

    t0 = time.perf_counter()
    try:
        plan_and_set_global_path(obs_env)
    except Exception:
        pass
    t_plan = time.perf_counter() - t0

    hidden = torch.zeros(1, 256, device=policy.device)
    traj = [obs_env.agent_pos.copy()]

    success = False
    collision = False

    t1 = time.perf_counter()
    for _ in range(max_steps):
        action, hidden = policy.act(obs, hidden)
        obs, _, done, _, info = obs_env.step(action)
        traj.append(obs_env.agent_pos.copy())
        if done:
            success = bool(info.get('success', False))
            collision = bool(info.get('collision', False))
            break
    t_roll = time.perf_counter() - t1

    return EpisodeResult(
        map_type=map_type,
        algo=algo,
        seed=seed,
        success=success,
        collision=collision,
        steps=len(traj),
        path_length=traj_length(traj),
        wall_time_ms=(t_plan + t_roll) * 1000.0,
    )


def run_astar_tracking(map_type: str, seed: int, max_steps: int = 800) -> EpisodeResult:
    np.random.seed(seed)
    env = AutonomousNavEnv(map_type=map_type)
    env.reset(seed=seed)

    t0 = time.perf_counter()
    gp = plan_and_set_global_path(env)
    t_plan = time.perf_counter() - t0

    traj = [env.agent_pos.copy()]
    wp = 0
    success = False
    collision = False

    t1 = time.perf_counter()
    for _ in range(max_steps):
        while wp < len(gp) - 1 and np.linalg.norm(env.agent_pos - gp[wp]) < 2.0:
            wp += 1
        target = gp[min(wp, len(gp) - 1)]

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

    return EpisodeResult(
        map_type=map_type,
        algo='A* tracking',
        seed=seed,
        success=success,
        collision=collision,
        steps=len(traj),
        path_length=traj_length(traj),
        wall_time_ms=(t_plan + t_roll) * 1000.0,
    )


def run_rrt_apf(map_type: str, seed: int, max_steps: int = 1200) -> EpisodeResult:
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

    return EpisodeResult(
        map_type=map_type,
        algo='RRT*+APF',
        seed=seed,
        success=success,
        collision=collision,
        steps=len(traj),
        path_length=traj_length(traj),
        wall_time_ms=(t_plan + t_roll) * 1000.0,
    )


def aggregate(results: list[EpisodeResult]):
    groups: dict[tuple[str, str], list[EpisodeResult]] = {}
    for r in results:
        groups.setdefault((r.map_type, r.algo), []).append(r)

    summary = []
    for (mt, algo), rows in sorted(groups.items()):
        n = len(rows)
        succ = 100.0 * sum(1 for r in rows if r.success) / n
        coll = 100.0 * sum(1 for r in rows if r.collision) / n
        plen = float(np.mean([r.path_length for r in rows]))
        tms = float(np.mean([r.wall_time_ms for r in rows]))
        steps = float(np.mean([r.steps for r in rows]))
        summary.append({
            'map': mt,
            'algo': algo,
            'n': n,
            'success_rate': succ,
            'collision_rate': coll,
            'path_length': plen,
            'wall_time_ms': tms,
            'steps': steps,
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
    lines.append('Ablation (5 methods) with dynamic obstacles — measured')
    lines.append('')
    hdr = f"{'Map':<8} {'Algo':<14} {'N':>3} {'Succ%':>7} {'Coll%':>7} {'Len':>10} {'Time(ms)':>10} {'Steps':>8}"
    lines.append(hdr)
    lines.append('-' * len(hdr))
    for r in rows:
        lines.append(
            f"{r['map']:<8} {r['algo']:<14} {r['n']:>3} {r['success_rate']:>6.1f} {r['collision_rate']:>6.1f} "
            f"{r['path_length']:>10.2f} {r['wall_time_ms']:>10.2f} {r['steps']:>8.1f}"
        )
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def plot_summary(path: str, summary: list[dict]):
    maps = ['simple', 'complex', 'concave', 'narrow']
    algos = ['A* tracking', 'RRT*+APF', 'PPO', 'Dual-Att PPO', 'Ours']

    def mat(key):
        out = np.zeros((len(maps), len(algos)), dtype=np.float32)
        for i, mt in enumerate(maps):
            for j, algo in enumerate(algos):
                for r in summary:
                    if r['map'] == mt and r['algo'] == algo:
                        out[i, j] = float(r[key])
        return out

    succ = mat('success_rate')
    plen = mat('path_length')
    tms = mat('wall_time_ms')

    plt.rcParams['font.family'] = 'Times New Roman'
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    x = np.arange(len(maps))
    width = 0.16
    colors = {
        'A* tracking': '#2980B9',
        'RRT*+APF': '#C0392B',
        'PPO': '#27AE60',
        'Dual-Att PPO': '#E67E22',
        'Ours': '#8E44AD',
    }

    def bars(ax, data, title, ylim=None):
        for j, algo in enumerate(algos):
            ax.bar(x + (j - 2) * width, data[:, j], width, label=algo, color=colors[algo], alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(['Simple', 'Complex', 'Concave', 'Narrow'])
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.grid(axis='y', alpha=0.25, linestyle='--')
        if ylim is not None:
            ax.set_ylim(ylim)

    bars(axes[0], succ, 'Success Rate (%)', ylim=(0, 105))
    bars(axes[1], plen, 'Path Length')
    bars(axes[2], tms, 'Wall Time (ms)')

    axes[0].legend(loc='lower left', fontsize=8)

    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def find_ckpt(paths: list[str]) -> str | None:
    for p in paths:
        if os.path.exists(p):
            return p
    return None


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Expect these checkpoints after training with ablation/train_ablation.py
    ckpt_basic = find_ckpt([
        os.path.join(ROOT, 'models_basic_astar', 'model.pth'),
        os.path.join(ROOT, 'models_basic_noastar', 'model.pth'),
    ])
    ckpt_att = find_ckpt([
        os.path.join(ROOT, 'models_attention_astar', 'model.pth'),
        os.path.join(ROOT, 'models_attention_noastar', 'model.pth'),
    ])
    ckpt_full = find_ckpt([
        os.path.join(ROOT, 'models_full_astar', 'model.pth'),
        os.path.join(ROOT, 'models_full_noastar', 'model.pth'),
        os.path.join(ROOT, 'best_model.pth'),
    ])

    missing = []
    if ckpt_basic is None:
        missing.append('PPO (Basic)')
    if ckpt_att is None:
        missing.append('Dual-Att PPO')
    if ckpt_full is None:
        missing.append('Ours (Full)')

    if missing:
        print('Missing checkpoints for:', ', '.join(missing))
        print('Train them first (example):')
        print('  python ablation\\train_ablation.py --model basic --episodes 3000')
        print('  python ablation\\train_ablation.py --model attention --episodes 3000')
        print('  python ablation\\train_ablation.py --model full --episodes 3000')
        print('Optionally add --no-astar for ablations without A* guidance.')

    policies: dict[str, PolicyWrapper] = {}
    if ckpt_basic is not None:
        policies['PPO'] = PolicyWrapper(BasicPPOActorCritic(action_dim=2), ckpt_basic, device)
    if ckpt_att is not None:
        policies['Dual-Att PPO'] = PolicyWrapper(DualAttentionPPOActorCritic(action_dim=2), ckpt_att, device)
    if ckpt_full is not None:
        policies['Ours'] = PolicyWrapper(CascadedDualAttentionActorCritic(action_dim=2), ckpt_full, device)

    maps = ['simple', 'complex', 'concave', 'narrow']
    episodes_per_map = 20
    seeds = list(range(2000, 2000 + episodes_per_map))

    results: list[EpisodeResult] = []
    for mt in maps:
        for seed in seeds:
            results.append(run_astar_tracking(mt, seed))
            results.append(run_rrt_apf(mt, seed))
            for name, pol in policies.items():
                results.append(run_policy(mt, seed, name, pol))

    summary = aggregate(results)

    out_csv = os.path.join(ROOT, 'ablation5_dynamic_metrics.csv')
    out_txt = os.path.join(ROOT, 'ablation5_dynamic_metrics.txt')
    out_png = os.path.join(ROOT, 'ablation5_dynamic_metrics.png')

    save_csv(out_csv, summary)
    save_txt(out_txt, summary)
    plot_summary(out_png, summary)

    print('Saved:', out_csv)
    print('Saved:', out_txt)
    print('Saved:', out_png)


if __name__ == '__main__':
    main()
