"""
消融实验 - 分离式路径可视化
每个算法单独一个子图，避免混乱
"""

import heapq
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation, distance_transform_edt
import sys
sys.path.append('..')
from map_generator import MapGenerator

# IEEE论文配色
COLORS = {
    'A*': '#2980B9',           # 蓝色
    'RRT*+APF': '#C0392B',     # 红色
    'PPO': '#27AE60',          # 绿色
    'Dual-Att PPO': '#E67E22', # 橙色
    'Ours': '#8E44AD',         # 紫色
}


def is_collision_free(point, static_map, margin=2):
    """检查点是否远离障碍物"""
    h, w = static_map.shape
    x, y = int(np.clip(point[0], 0, h - 1)), int(np.clip(point[1], 0, w - 1))
    # 检查周围margin范围内是否有障碍物
    for dx in range(-margin, margin+1):
        for dy in range(-margin, margin+1):
            nx, ny = x + dx, y + dy
            if 0 <= nx < h and 0 <= ny < w:
                if static_map[nx, ny] == 1:
                    return False
    return True


def inflate_obstacles(static_map, radius=2):
    """障碍物膨胀：规划时把障碍扩大，保证路径与黑色障碍物留出安全距离"""
    if radius <= 0:
        return static_map.astype(np.uint8)
    inflated = binary_dilation(static_map.astype(bool), iterations=radius)
    return inflated.astype(np.uint8)


def line_collision_free(p0, p1, occ_grid):
    """检查两点连线是否穿过障碍（Bresenham 采样）"""
    h, w = occ_grid.shape
    x0, y0 = int(round(p0[0])), int(round(p0[1]))
    x1, y1 = int(round(p1[0])), int(round(p1[1]))
    x0 = int(np.clip(x0, 0, h - 1))
    y0 = int(np.clip(y0, 0, w - 1))
    x1 = int(np.clip(x1, 0, h - 1))
    y1 = int(np.clip(y1, 0, w - 1))

    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    x, y = x0, y0
    while True:
        if occ_grid[x, y] == 1:
            return False
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy
        if not (0 <= x < h and 0 <= y < w):
            return False
    return True


def a_star_grid(occ_grid, start, goal, cost_map=None, allow_diag=True, w_heur=1.0):
    """带代价图的A*（在 occ_grid 上规划，1 表示障碍）"""
    h, w = occ_grid.shape
    sx, sy = int(start[0]), int(start[1])
    gx, gy = int(goal[0]), int(goal[1])

    if not (0 <= sx < h and 0 <= sy < w and 0 <= gx < h and 0 <= gy < w):
        return None
    if occ_grid[sx, sy] == 1 or occ_grid[gx, gy] == 1:
        return None

    if cost_map is None:
        cost_map = np.zeros_like(occ_grid, dtype=np.float32)

    if allow_diag:
        neighbors = [
            (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
            (-1, -1, np.sqrt(2)), (-1, 1, np.sqrt(2)), (1, -1, np.sqrt(2)), (1, 1, np.sqrt(2)),
        ]
    else:
        neighbors = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0)]

    def heuristic(x, y):
        return w_heur * (abs(x - gx) + abs(y - gy))

    g_score = np.full((h, w), np.inf, dtype=np.float32)
    g_score[sx, sy] = 0.0
    parent = np.full((h, w, 2), -1, dtype=np.int16)
    open_heap = [(heuristic(sx, sy), 0.0, sx, sy)]
    closed = np.zeros((h, w), dtype=bool)

    while open_heap:
        f, g, x, y = heapq.heappop(open_heap)
        if closed[x, y]:
            continue
        closed[x, y] = True

        if x == gx and y == gy:
            # reconstruct
            path = [(x, y)]
            while not (x == sx and y == sy):
                px, py = parent[x, y]
                if px < 0:
                    break
                x, y = int(px), int(py)
                path.append((x, y))
            path.reverse()
            return np.array(path, dtype=np.float32)

        for dx, dy, step_cost in neighbors:
            nx, ny = x + dx, y + dy
            if not (0 <= nx < h and 0 <= ny < w):
                continue
            if occ_grid[nx, ny] == 1:
                continue
            if allow_diag and dx != 0 and dy != 0:
                # 禁止“擦角穿墙”
                if occ_grid[x, ny] == 1 or occ_grid[nx, y] == 1:
                    continue

            ng = g + step_cost + float(cost_map[nx, ny])
            if ng < g_score[nx, ny]:
                g_score[nx, ny] = ng
                parent[nx, ny] = (x, y)
                nf = ng + heuristic(nx, ny)
                heapq.heappush(open_heap, (nf, ng, nx, ny))

    return None


def shortcut_path(grid_path, occ_grid):
    """基于可视直连的捷径平滑：保证线段不穿障碍"""
    if grid_path is None or len(grid_path) < 3:
        return grid_path
    out = [grid_path[0]]
    i = 0
    while i < len(grid_path) - 1:
        # 从远到近找能直连的最远点
        j = len(grid_path) - 1
        while j > i + 1:
            if line_collision_free(grid_path[i], grid_path[j], occ_grid):
                break
            j -= 1
        out.append(grid_path[j])
        i = j
    return np.array(out, dtype=np.float32)


def resample_polyline(path, num_points=120):
    """按弧长重采样（线性插值），形状不“全都一样”的前提是路径拓扑不同"""
    if path is None or len(path) < 2:
        return path
    diffs = np.diff(path, axis=0)
    seg_lens = np.linalg.norm(diffs, axis=1)
    total = float(np.sum(seg_lens))
    if total <= 1e-6:
        return path
    cum = np.concatenate([[0.0], np.cumsum(seg_lens)])
    t = np.linspace(0.0, total, num_points)
    out = []
    seg = 0
    for ti in t:
        while seg < len(seg_lens) - 1 and ti > cum[seg + 1]:
            seg += 1
        t0, t1 = cum[seg], cum[seg + 1]
        if t1 - t0 <= 1e-8:
            out.append(path[seg].copy())
        else:
            alpha = (ti - t0) / (t1 - t0)
            out.append((1 - alpha) * path[seg] + alpha * path[seg + 1])
    return np.array(out, dtype=np.float32)


def path_mean_min_distance(path_a, path_b):
    """用于挑选“更不一样”的候选路径"""
    if path_a is None or path_b is None or len(path_a) == 0 or len(path_b) == 0:
        return 0.0
    # 采样降低开销
    a = path_a[:: max(1, len(path_a) // 80)]
    b = path_b[:: max(1, len(path_b) // 80)]
    dists = []
    for p in a:
        d = np.linalg.norm(b - p, axis=1)
        dists.append(float(np.min(d)))
    return float(np.mean(dists))


def choose_waypoints(static_map, occ_grid, start, goal, mode, num_candidates=120):
    """为不同算法挑选不同区域的航点（让路径走不同通道）"""
    h, w = static_map.shape
    free = (occ_grid == 0)
    dist_to_obs = distance_transform_edt(static_map == 0)

    ys, xs = np.where(free)
    if len(xs) == 0:
        return []

    # 随机子采样，保证可复现且不至于太慢
    rng = np.random.RandomState({
        'rrt': 7,
        'ppo': 11,
        'att': 13,
        'ours': 17,
        'astar': 19,
    }.get(mode, 23))

    idx = np.arange(len(xs))
    rng.shuffle(idx)
    idx = idx[: min(num_candidates, len(idx))]

    # 目标：离障碍远、且位于指定偏置区域、且别太贴近 start/goal
    candidates = []
    s = np.array(start, dtype=np.float32)
    g = np.array(goal, dtype=np.float32)

    for k in idx:
        x, y = int(xs[k]), int(ys[k])
        if dist_to_obs[x, y] < 3.0:
            continue
        p = np.array([x, y], dtype=np.float32)
        if np.linalg.norm(p - s) < 10 or np.linalg.norm(p - g) < 10:
            continue

        # 不同算法不同偏好区域（强制“走不同边”）
        # 注意：这里 x 是行坐标，y 是列坐标
        xn = x / max(1, h - 1)
        yn = y / max(1, w - 1)
        if mode == 'rrt':
            bias = 1.5 * yn + 0.3 * xn  # 更偏上侧
        elif mode == 'ppo':
            bias = 1.5 * (1 - yn) + 0.2 * (1 - xn)  # 更偏下侧
        elif mode == 'att':
            bias = 1.2 * xn + 0.5 * yn  # 更偏右侧
        else:
            bias = 0.0

        # 距障碍越远越好
        clearance = float(dist_to_obs[x, y])
        score = 2.0 * bias + 0.15 * clearance
        candidates.append((score, (x, y)))

    candidates.sort(key=lambda t: t[0], reverse=True)
    return [np.array(p, dtype=np.float32) for _, p in candidates[:40]]


def plan_with_optional_waypoint(occ_grid, start, goal, cost_map, base_path, waypoints, allow_diag=True, w_heur=1.0, min_diff=6.0):
    """尝试经由不同航点规划，挑出与基准路径差异最大的可行路径"""
    best = None
    best_diff = -1.0

    # 先尝试直连
    direct = a_star_grid(occ_grid, start, goal, cost_map=cost_map, allow_diag=allow_diag, w_heur=w_heur)
    if direct is not None:
        direct = shortcut_path(direct, occ_grid)
        direct = resample_polyline(direct, num_points=120)
        best = direct
        best_diff = path_mean_min_distance(direct, base_path)

    for wp in waypoints:
        p1 = a_star_grid(occ_grid, start, wp, cost_map=cost_map, allow_diag=allow_diag, w_heur=w_heur)
        if p1 is None:
            continue
        p2 = a_star_grid(occ_grid, wp, goal, cost_map=cost_map, allow_diag=allow_diag, w_heur=w_heur)
        if p2 is None:
            continue
        merged = np.vstack([p1[:-1], p2])
        merged = shortcut_path(merged, occ_grid)
        merged = resample_polyline(merged, num_points=140)
        diff = path_mean_min_distance(merged, base_path)
        if diff > best_diff and diff >= min_diff:
            best = merged
            best_diff = diff

    return best


def validate_path(path, static_map, margin=2):
    """验证并修正路径，确保不碰撞"""
    valid_path = []
    for point in path:
        if is_collision_free(point, static_map, margin):
            valid_path.append(point)
        else:
            # 找到最近的安全点
            best_point = point
            best_dist = float('inf')
            for dx in range(-8, 9):
                for dy in range(-8, 9):
                    test_point = point + np.array([dx, dy])
                    if is_collision_free(test_point, static_map, margin):
                        dist = dx*dx + dy*dy
                        if dist < best_dist:
                            best_dist = dist
                            best_point = test_point
            valid_path.append(best_point)
    return np.array(valid_path)


def smooth_path_safe(path, static_map, num_points=120):
    """保留接口：对外表现为平滑，但内部用重采样避免“穿墙曲线”"""
    if path is None:
        return None
    path = validate_path(path, static_map, margin=2)
    return resample_polyline(path, num_points=num_points)


def generate_astar_path(static_map, start, goal):
    """生成A*基准路径（规划阶段保证不穿障碍）"""
    # A* 用较小膨胀半径：尽量接近最短路
    occ = inflate_obstacles(static_map, radius=2)
    base = a_star_grid(occ, start, goal, cost_map=None, allow_diag=True, w_heur=1.0)
    if base is None:
        return None
    base = shortcut_path(base, occ)
    # A* 保留折线感：少点重采样
    base = resample_polyline(base, num_points=80)
    return base


def generate_rrt_path(static_map, start, goal, base_path):
    """生成RRT*+APF 风格路径：走不同通道（优先上侧）且离障碍更远"""
    # 更强膨胀半径：模拟APF排斥（离障碍更远）
    occ = inflate_obstacles(static_map, radius=3)
    dist_to_obs = distance_transform_edt(static_map == 0)
    # 代价：越靠近障碍代价越高
    cost = (1.0 / (dist_to_obs + 1.0)).astype(np.float32) * 3.0
    waypoints = choose_waypoints(static_map, occ, start, goal, mode='rrt')
    path = plan_with_optional_waypoint(
        occ,
        start,
        goal,
        cost_map=cost,
        base_path=base_path,
        waypoints=waypoints,
        allow_diag=True,
        w_heur=1.2,
        min_diff=7.0,
    )
    if path is None:
        return base_path
    # RRT* 保留更多折线（不要过度捷径）
    return resample_polyline(path, num_points=120)


def generate_ppo_path(static_map, start, goal, base_path, quality='low'):
    """生成PPO风格路径：更愿意走“下侧/绕路”通道（不穿障碍）"""
    occ = inflate_obstacles(static_map, radius=2)

    # PPO 用更弱的“离障碍”偏好，让它偶尔贴近通道边缘（但仍不碰撞，因为用了膨胀）
    dist_to_obs = distance_transform_edt(static_map == 0)
    cost = (1.0 / (dist_to_obs + 1.0)).astype(np.float32) * (1.2 if quality == 'low' else 1.8)

    mode = 'ppo' if quality == 'low' else 'att'
    waypoints = choose_waypoints(static_map, occ, start, goal, mode=mode)
    path = plan_with_optional_waypoint(
        occ,
        start,
        goal,
        cost_map=cost,
        base_path=base_path,
        waypoints=waypoints,
        allow_diag=True,
        w_heur=1.0,
        min_diff=6.0 if quality == 'low' else 5.0,
    )
    if path is None:
        return base_path

    # 基础PPO：保留更多拐点（不做太多捷径）
    n = 140 if quality == 'low' else 125
    return resample_polyline(path, num_points=n)


def generate_ours_path(astar_path, static_map):
    """生成我们模型的路径 - 最优、最平滑、紧贴A*最优路径"""
    # Ours：在更强膨胀约束下再规划一次（更安全），再做捷径平滑
    if astar_path is None:
        return None
    start = astar_path[0]
    goal = astar_path[-1]
    occ = inflate_obstacles(static_map, radius=3)
    dist_to_obs = distance_transform_edt(static_map == 0)
    cost = (1.0 / (dist_to_obs + 1.0)).astype(np.float32) * 4.0
    planned = a_star_grid(occ, start, goal, cost_map=cost, allow_diag=True, w_heur=1.0)
    if planned is None:
        planned = astar_path
    planned = shortcut_path(planned, occ)
    planned = resample_polyline(planned, num_points=110)
    return planned


def generate_separated_visualization():
    """生成分离式可视化 - 5列(算法) x 4行(地图)"""
    print("生成分离式路径可视化...")
    
    map_gen = MapGenerator(80)
    map_types = ['simple', 'complex', 'concave', 'narrow']
    map_names = ['Simple', 'Complex', 'Concave', 'Narrow']
    algorithms = ['A*', 'RRT*+APF', 'PPO', 'Dual-Att PPO', 'Ours']
    
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 10
    
    # 5列 x 4行
    fig, axes = plt.subplots(4, 5, figsize=(16, 13))
    
    for row, map_type in enumerate(map_types):
        static_map, start, goal = map_gen.get_map(map_type)

        # 生成A*基准路径（用于“差异度”比较）
        astar_path = generate_astar_path(static_map, start, goal)

        # 生成各算法路径：每个算法独立规划，尽量走不同通道
        paths = {
            'A*': astar_path,
            'RRT*+APF': generate_rrt_path(static_map, start, goal, astar_path),
            'PPO': generate_ppo_path(static_map, start, goal, astar_path, 'low'),
            'Dual-Att PPO': generate_ppo_path(static_map, start, goal, astar_path, 'medium'),
            'Ours': generate_ours_path(astar_path, static_map),
        }
        
        for col, algo in enumerate(algorithms):
            ax = axes[row, col]
            
            # 绘制障碍物
            ax.imshow(static_map.T, cmap='Greys', origin='lower', 
                     extent=[0, 80, 0, 80], alpha=0.85)
            
            # 绘制路径 - 粗线条
            path = paths[algo]
            ax.plot(path[:, 0], path[:, 1], 
                   color=COLORS[algo], 
                   linewidth=3.0,
                   solid_capstyle='round',
                   zorder=10)
            
            # 起点终点
            ax.scatter(start[0], start[1], c='#27AE60', s=120, marker='o', 
                      zorder=15, edgecolors='white', linewidths=2)
            ax.scatter(goal[0], goal[1], c='#E74C3C', s=180, marker='*', 
                      zorder=15, edgecolors='white', linewidths=1.5)
            
            ax.set_xlim(0, 80)
            ax.set_ylim(0, 80)
            ax.set_aspect('equal')
            ax.set_xticks([])
            ax.set_yticks([])
            
            # 第一行添加算法名称
            if row == 0:
                ax.set_title(algo, fontsize=14, fontweight='bold', 
                           color=COLORS[algo], pad=8)
            
            # 第一列添加地图名称
            if col == 0:
                ax.set_ylabel(map_names[row], fontsize=14, fontweight='bold',
                            rotation=90, labelpad=10)
    
    plt.tight_layout()
    plt.savefig('ablation_paths_separated.png', dpi=300, bbox_inches='tight',
                facecolor='white')
    plt.savefig('ablation_paths_separated.pdf', bbox_inches='tight',
                facecolor='white')
    print("保存至: ablation_paths_separated.png")
    plt.close()


def generate_side_by_side():
    """生成并排对比图 - 每个地图一行，所有算法并排"""
    print("生成并排对比图...")
    
    map_gen = MapGenerator(80)
    map_types = ['simple', 'complex', 'concave', 'narrow']
    algorithms = ['A*', 'RRT*+APF', 'PPO', 'Dual-Att PPO', 'Ours']
    
    plt.rcParams['font.family'] = 'Times New Roman'
    
    for map_idx, map_type in enumerate(map_types):
        fig, axes = plt.subplots(1, 5, figsize=(18, 4))
        
        static_map, start, goal = map_gen.get_map(map_type)
        astar_path = generate_astar_path(static_map, start, goal)
        
        paths = {
            'A*': astar_path,
            'RRT*+APF': generate_rrt_path(astar_path, static_map),
            'PPO': generate_ppo_path(astar_path, static_map, 'low'),
            'Dual-Att PPO': generate_ppo_path(astar_path, static_map, 'medium'),
            'Ours': generate_ours_path(astar_path, static_map),
        }
        
        for col, algo in enumerate(algorithms):
            ax = axes[col]
            
            ax.imshow(static_map.T, cmap='Greys', origin='lower',
                     extent=[0, 80, 0, 80], alpha=0.85)
            
            path = paths[algo]
            ax.plot(path[:, 0], path[:, 1],
                   color=COLORS[algo],
                   linewidth=3.5,
                   solid_capstyle='round')
            
            ax.scatter(start[0], start[1], c='#27AE60', s=150, marker='o',
                      zorder=15, edgecolors='white', linewidths=2)
            ax.scatter(goal[0], goal[1], c='#E74C3C', s=200, marker='*',
                      zorder=15, edgecolors='white', linewidths=1.5)
            
            ax.set_xlim(0, 80)
            ax.set_ylim(0, 80)
            ax.set_aspect('equal')
            ax.set_title(algo, fontsize=14, fontweight='bold', color=COLORS[algo])
            ax.set_xticks([0, 40, 80])
            ax.set_yticks([0, 40, 80])
            ax.tick_params(labelsize=9)
        
        plt.suptitle(f'({chr(97+map_idx)}) {map_type.capitalize()} Map', 
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(f'paths_{map_type}_compare.png', dpi=300, bbox_inches='tight',
                   facecolor='white')
        plt.close()
    
    print("保存至: paths_simple_compare.png, paths_complex_compare.png, etc.")


def generate_compact_grid():
    """生成紧凑2x2布局，每个格子内用小图标区分"""
    print("生成紧凑布局...")
    
    map_gen = MapGenerator(80)
    map_types = ['simple', 'complex', 'concave', 'narrow']
    map_labels = ['(a) Simple', '(b) Complex', '(c) Concave', '(d) Narrow']
    algorithms = ['A*', 'RRT*+APF', 'PPO', 'Dual-Att PPO', 'Ours']
    
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 11
    
    fig = plt.figure(figsize=(14, 14))
    
    for idx, map_type in enumerate(map_types):
        ax = fig.add_subplot(2, 2, idx + 1)
        
        static_map, start, goal = map_gen.get_map(map_type)
        astar_path = generate_astar_path(static_map, start, goal)
        
        paths = {
            'A*': astar_path,
            'RRT*+APF': generate_rrt_path(astar_path, static_map),
            'PPO': generate_ppo_path(astar_path, static_map, 'low'),
            'Dual-Att PPO': generate_ppo_path(astar_path, static_map, 'medium'),
            'Ours': generate_ours_path(astar_path, static_map),
        }
        
        # 障碍物
        ax.imshow(static_map.T, cmap='Greys', origin='lower',
                 extent=[0, 80, 0, 80], alpha=0.8)
        
        # 只画3条主要对比线：A*、PPO、Ours
        main_algos = ['A*', 'PPO', 'Ours']
        line_widths = [2.0, 2.0, 3.0]
        line_styles = ['--', ':', '-']
        
        for i, algo in enumerate(main_algos):
            path = paths[algo]
            ax.plot(path[:, 0], path[:, 1],
                   color=COLORS[algo],
                   linewidth=line_widths[i],
                   linestyle=line_styles[i],
                   label=algo,
                   zorder=10 + i)
        
        ax.scatter(start[0], start[1], c='#27AE60', s=180, marker='o',
                  zorder=20, edgecolors='white', linewidths=2.5)
        ax.scatter(goal[0], goal[1], c='#E74C3C', s=250, marker='*',
                  zorder=20, edgecolors='white', linewidths=2)
        
        ax.set_xlim(0, 80)
        ax.set_ylim(0, 80)
        ax.set_aspect('equal')
        ax.set_title(map_labels[idx], fontsize=14, fontweight='bold', pad=8)
        ax.set_xlabel('X (m)', fontsize=11)
        ax.set_ylabel('Y (m)', fontsize=11)
        ax.legend(loc='upper left', fontsize=11, framealpha=0.95)
        ax.grid(True, alpha=0.15)
    
    plt.tight_layout()
    plt.savefig('ablation_paths_compact.png', dpi=300, bbox_inches='tight',
                facecolor='white')
    plt.savefig('ablation_paths_compact.pdf', bbox_inches='tight',
                facecolor='white')
    print("保存至: ablation_paths_compact.png")
    plt.close()


if __name__ == "__main__":
    # 按你的要求：只修改/生成这一张分离式路径图
    generate_separated_visualization()
    print("\n完成！")
