"""动态障碍物对比（concave: 1个；narrow: 2个）

说明：该脚本用同一套动态障碍物配置（来自 map_generator.py）对比
- A* 跟踪（无动态避障）
- RRT*+APF（带动态避障）
- Ours（best_navigation_model.pth）

输出：dynamic_obstacles_compare.png / .pdf
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from env import AutonomousNavEnv
from map_generator import MapGenerator
from global_planner import SmartAStarPlanner

from ablation.rrt_apf_planner import RRTStarAPFNavigator
from visualize_paths import PathVisualizer


def update_dynamic_obstacles(obstacles: list[dict], map_size: int):
    for obs in obstacles:
        obs['pos'] = obs['pos'] + obs['vel']
        if not (1 < obs['pos'][0] < map_size - 1):
            obs['vel'][0] *= -1
        if not (1 < obs['pos'][1] < map_size - 1):
            obs['vel'][1] *= -1


def run_astar_tracking(map_type: str, max_steps: int = 500):
    env = AutonomousNavEnv(map_type=map_type)
    planner = SmartAStarPlanner(env.static_map)
    global_path = planner.plan(env.start_pos, env.goal_pos, env.dynamic_obstacles)
    if global_path is not None and len(global_path) > 0:
        env.set_global_path(global_path)

    obs, _ = env.reset()

    # 重新注入 global_path（reset 后会清空）
    env.set_global_path(global_path)

    traj = [env.agent_pos.copy()]
    dyn_hist = []

    if env.dynamic_obstacles:
        dyn_hist.append([
            {'pos': o['pos'].copy(), 'radius': float(o.get('radius', 2.0))}
            for o in env.dynamic_obstacles
        ])

    wp_idx = 0
    success = False
    collision = False

    for _ in range(max_steps):
        # 选择前方目标点
        while wp_idx < len(global_path) - 1 and np.linalg.norm(env.agent_pos - global_path[wp_idx]) < 2.0:
            wp_idx += 1
        target = global_path[min(wp_idx, len(global_path) - 1)]

        # 简单朝向控制
        desired = np.arctan2(target[1] - env.agent_pos[1], target[0] - env.agent_pos[0])
        ang = desired - env.agent_dir
        ang = (ang + np.pi) % (2 * np.pi) - np.pi

        v_cmd = 1.0
        w_cmd = float(np.clip(ang, -1.0, 1.0))

        # env action: a0 in [-1,1] maps to v_cmd in [0,1]
        action = np.array([1.0, w_cmd], dtype=np.float32)

        obs, r, done, _, info = env.step(action)
        traj.append(env.agent_pos.copy())

        if env.dynamic_obstacles:
            dyn_hist.append([
                {'pos': o['pos'].copy(), 'radius': float(o.get('radius', 2.0))}
                for o in env.dynamic_obstacles
            ])

        if done:
            success = bool(info.get('success', False))
            collision = bool(info.get('collision', False))
            break

    return np.array(traj), dyn_hist, success, collision


def run_rrt_apf(map_type: str, max_steps: int = 800):
    mg = MapGenerator(80)
    grid, start, goal = mg.get_map(map_type)

    nav = RRTStarAPFNavigator(grid, start, goal)

    # 拷贝动态障碍物（固定配置）
    dyn = []
    for o in (mg.get_dynamic_obstacles(map_type) or []):
        dyn.append({
            'pos': np.array(o['pos'], dtype=np.float32).copy(),
            'vel': np.array(o.get('vel', [0.2, 0.2]), dtype=np.float32).copy(),
            'radius': float(o.get('radius', 2.0)),
        })

    traj = [nav.pos.copy()]
    dyn_hist = []

    if dyn:
        dyn_hist.append([
            {'pos': o['pos'].copy(), 'radius': float(o.get('radius', 2.0))}
            for o in dyn
        ])

    success = False
    collision = False

    for _ in range(max_steps):
        nav.set_dynamic_obstacles(dyn)
        pos, done, info = nav.step()
        traj.append(pos.copy())

        # 更新动态障碍物
        update_dynamic_obstacles(dyn, 80)

        if dyn:
            dyn_hist.append([
                {'pos': o['pos'].copy(), 'radius': float(o.get('radius', 2.0))}
                for o in dyn
            ])

        if done:
            success = bool(info.get('success', False))
            collision = bool(info.get('collision', False))
            break

    return np.array(traj), dyn_hist, success, collision


def run_ours(map_type: str, max_steps: int = 500):
    env = AutonomousNavEnv(map_type=map_type)
    planner = SmartAStarPlanner(env.static_map)
    try:
        gp = planner.plan(env.start_pos, env.goal_pos, env.dynamic_obstacles)
        if gp is not None and len(gp) > 0:
            env.set_global_path(gp)
    except:
        pass

    vis = PathVisualizer('best_navigation_model.pth')
    traj, dyn_hist, success = vis.generate_path_with_dynamic_obs(env, max_steps=max_steps)

    # env 里 info 已包含 collision，但 PathVisualizer 只返回 success；
    # 这里用最后一步是否越界/撞墙/撞动态障碍做一个保守判定
    collision = False
    if len(traj) > 0:
        p = traj[-1]
        cx, cy = int(p[0]), int(p[1])
        if not (0 <= cx < 80 and 0 <= cy < 80) or env.static_map[cx, cy] == 1:
            collision = True
        else:
            for o in env.dynamic_obstacles:
                if np.linalg.norm(p - o['pos']) < float(o.get('radius', 2.0)):
                    collision = True
                    break

    return traj, dyn_hist, bool(success), bool(collision)


def plot_panel(ax, grid, traj, dyn_hist, title):
    ax.imshow(grid.T, cmap='Greys', origin='lower', extent=[0, 80, 0, 80], alpha=0.9)

    if dyn_hist and len(dyn_hist) > 0:
        for obs_idx in range(len(dyn_hist[0])):
            obs_traj = np.array([frame[obs_idx]['pos'] for frame in dyn_hist])
            radius = float(dyn_hist[0][obs_idx].get('radius', 2.0))
            ax.plot(obs_traj[:, 0], obs_traj[:, 1], color='#00BCD4', linestyle='--', linewidth=1.2, alpha=0.6)
            ax.add_patch(plt.Circle((obs_traj[0, 0], obs_traj[0, 1]), radius, color='#1565C0', alpha=0.65))

    ax.plot(traj[:, 0], traj[:, 1], color='#8E44AD', linewidth=2.6)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlim(0, 80)
    ax.set_ylim(0, 80)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])


def main():
    plt.rcParams['font.family'] = 'Times New Roman'

    cases = [
        ('concave', '(c) Concave + 1 dynamic obstacle'),
        ('narrow', '(d) Narrow + 2 dynamic obstacles'),
    ]

    algos = [
        ('A* tracking', run_astar_tracking),
        ('RRT*+APF', run_rrt_apf),
        ('Ours', run_ours),
    ]

    fig, axes = plt.subplots(len(cases), len(algos), figsize=(14, 8))
    if len(cases) == 1:
        axes = np.expand_dims(axes, axis=0)

    mg = MapGenerator(80)

    summary = []
    for r, (map_type, map_label) in enumerate(cases):
        grid, start, goal = mg.get_map(map_type)
        for c, (algo_name, runner) in enumerate(algos):
            traj, dyn_hist, success, collision = runner(map_type)
            status = 'SUCCESS' if success else ('COLLISION' if collision else 'TIMEOUT')
            title = f"{map_label}\n{algo_name}: {status}"
            plot_panel(axes[r, c], grid, traj, dyn_hist, title)
            summary.append((map_type, algo_name, status, len(traj)))

    plt.tight_layout()
    out_png = os.path.join(os.path.dirname(__file__), '..', 'dynamic_obstacles_compare.png')
    out_pdf = os.path.join(os.path.dirname(__file__), '..', 'dynamic_obstacles_compare.pdf')
    plt.savefig(out_png, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(out_pdf, bbox_inches='tight', facecolor='white')
    plt.close()

    print('Saved:', out_png)
    print('Saved:', out_pdf)
    print('\nSummary:')
    for row in summary:
        print(row)


if __name__ == '__main__':
    main()
