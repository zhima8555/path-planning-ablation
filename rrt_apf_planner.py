"""
消融实验 - RRT* + APF (人工势场法) 路径规划
RRT*用于全局路径规划，APF用于局部避障
"""

import numpy as np
import sys
sys.path.append('..')


class RRTStarPlanner:
    """RRT*路径规划器"""
    
    def __init__(self, grid_map, step_size=3.0, max_iter=3000, goal_sample_rate=0.1):
        self.grid_map = grid_map
        self.rows, self.cols = grid_map.shape
        self.step_size = step_size
        self.max_iter = max_iter
        self.goal_sample_rate = goal_sample_rate
        self.neighbor_radius = step_size * 2
        
    def plan(self, start, goal):
        """
        RRT*路径规划
        
        Args:
            start: 起点 (x, y)
            goal: 终点 (x, y)
            
        Returns:
            path: 路径点列表
        """
        start = np.array(start, dtype=np.float32)
        goal = np.array(goal, dtype=np.float32)
        
        # 节点列表: [(位置, 父节点索引, 代价)]
        nodes = [(start, -1, 0.0)]
        
        for i in range(self.max_iter):
            # 随机采样（有概率直接采样目标点）
            if np.random.random() < self.goal_sample_rate:
                sample = goal
            else:
                sample = np.array([
                    np.random.uniform(0, self.rows),
                    np.random.uniform(0, self.cols)
                ])
            
            # 找最近节点
            nearest_idx = self._find_nearest(nodes, sample)
            nearest_pos = nodes[nearest_idx][0]
            
            # 向采样点扩展
            direction = sample - nearest_pos
            dist = np.linalg.norm(direction)
            if dist < 1e-6:
                continue
                
            direction = direction / dist
            new_pos = nearest_pos + direction * min(self.step_size, dist)
            
            # 碰撞检测
            if not self._is_collision_free(nearest_pos, new_pos):
                continue
            
            # RRT*: 在邻域内寻找最优父节点
            near_indices = self._find_near_nodes(nodes, new_pos)
            
            # 选择代价最小的父节点
            best_parent = nearest_idx
            best_cost = nodes[nearest_idx][2] + np.linalg.norm(new_pos - nearest_pos)
            
            for idx in near_indices:
                node_pos = nodes[idx][0]
                new_cost = nodes[idx][2] + np.linalg.norm(new_pos - node_pos)
                if new_cost < best_cost and self._is_collision_free(node_pos, new_pos):
                    best_parent = idx
                    best_cost = new_cost
            
            # 添加新节点
            new_idx = len(nodes)
            nodes.append((new_pos, best_parent, best_cost))
            
            # RRT*: 重新布线
            for idx in near_indices:
                node_pos, node_parent, node_cost = nodes[idx]
                new_cost = best_cost + np.linalg.norm(node_pos - new_pos)
                if new_cost < node_cost and self._is_collision_free(new_pos, node_pos):
                    nodes[idx] = (node_pos, new_idx, new_cost)
            
            # 检查是否到达目标
            if np.linalg.norm(new_pos - goal) < self.step_size:
                if self._is_collision_free(new_pos, goal):
                    final_cost = best_cost + np.linalg.norm(goal - new_pos)
                    nodes.append((goal, new_idx, final_cost))
                    return self._extract_path(nodes, len(nodes) - 1)
        
        # 未找到完整路径，返回最接近目标的路径
        closest_idx = self._find_nearest(nodes, goal)
        return self._extract_path(nodes, closest_idx)
    
    def _find_nearest(self, nodes, point):
        """找最近节点"""
        distances = [np.linalg.norm(node[0] - point) for node in nodes]
        return np.argmin(distances)
    
    def _find_near_nodes(self, nodes, point):
        """找邻域内的节点"""
        indices = []
        for i, node in enumerate(nodes):
            if np.linalg.norm(node[0] - point) < self.neighbor_radius:
                indices.append(i)
        return indices
    
    def _is_collision_free(self, p1, p2):
        """检查两点之间是否无碰撞"""
        dist = np.linalg.norm(p2 - p1)
        if dist < 1e-6:
            return True
            
        num_checks = int(dist / 0.5) + 1
        for i in range(num_checks + 1):
            t = i / num_checks
            point = p1 + t * (p2 - p1)
            ix, iy = int(point[0]), int(point[1])
            
            if not (0 <= ix < self.rows and 0 <= iy < self.cols):
                return False
            if self.grid_map[ix, iy] == 1:
                return False
                
        return True
    
    def _extract_path(self, nodes, goal_idx):
        """从节点树中提取路径"""
        path = []
        idx = goal_idx
        while idx != -1:
            path.append(nodes[idx][0])
            idx = nodes[idx][1]
        return np.array(path[::-1])


class APFController:
    """人工势场法(APF)局部控制器"""
    
    def __init__(self, grid_map, attractive_gain=1.0, repulsive_gain=100.0, 
                 influence_distance=10.0):
        self.grid_map = grid_map
        self.rows, self.cols = grid_map.shape
        self.k_att = attractive_gain      # 引力增益
        self.k_rep = repulsive_gain       # 斥力增益
        self.d0 = influence_distance      # 障碍物影响距离
        
    def compute_force(self, pos, goal, dynamic_obstacles=[]):
        """
        计算合力
        
        Args:
            pos: 当前位置
            goal: 目标位置
            dynamic_obstacles: 动态障碍物列表 [{'pos': [x,y]}, ...]
            
        Returns:
            force: 合力向量
        """
        pos = np.array(pos, dtype=np.float32)
        goal = np.array(goal, dtype=np.float32)
        
        # 引力（指向目标）
        f_att = self._attractive_force(pos, goal)
        
        # 斥力（远离障碍物）
        f_rep = self._repulsive_force(pos, goal)
        
        # 动态障碍物斥力
        for obs in dynamic_obstacles:
            obs_pos = np.array(obs['pos'])
            dist = np.linalg.norm(pos - obs_pos)
            if dist < self.d0 and dist > 0.1:
                direction = (pos - obs_pos) / dist
                # 动态障碍物斥力更强
                magnitude = self.k_rep * 2 * (1/dist - 1/self.d0) * (1/dist**2)
                f_rep += magnitude * direction
        
        # 合力
        total_force = f_att + f_rep
        
        # 限制最大力
        max_force = 5.0
        force_magnitude = np.linalg.norm(total_force)
        if force_magnitude > max_force:
            total_force = total_force / force_magnitude * max_force
            
        return total_force
    
    def _attractive_force(self, pos, goal):
        """计算引力"""
        direction = goal - pos
        dist = np.linalg.norm(direction)
        
        if dist < 0.1:
            return np.zeros(2)
            
        # 线性引力场
        return self.k_att * direction / dist
    
    def _repulsive_force(self, pos, goal):
        """计算障碍物斥力"""
        f_rep = np.zeros(2)
        
        # 在pos周围搜索障碍物
        search_range = int(self.d0) + 1
        ix, iy = int(pos[0]), int(pos[1])
        
        for dx in range(-search_range, search_range + 1):
            for dy in range(-search_range, search_range + 1):
                nx, ny = ix + dx, iy + dy
                
                if not (0 <= nx < self.rows and 0 <= ny < self.cols):
                    continue
                    
                if self.grid_map[nx, ny] == 1:  # 是障碍物
                    obs_pos = np.array([nx + 0.5, ny + 0.5])
                    dist = np.linalg.norm(pos - obs_pos)
                    
                    if dist < self.d0 and dist > 0.1:
                        direction = (pos - obs_pos) / dist
                        # 斥力公式
                        dist_to_goal = np.linalg.norm(goal - pos)
                        magnitude = self.k_rep * (1/dist - 1/self.d0) * (1/dist**2) * (dist_to_goal**2)
                        f_rep += magnitude * direction
        
        return f_rep


class RRTStarAPFNavigator:
    """RRT* + APF 导航器"""
    
    def __init__(self, grid_map, start, goal):
        self.grid_map = grid_map
        self.start = np.array(start, dtype=np.float32)
        self.goal = np.array(goal, dtype=np.float32)
        self.pos = self.start.copy()
        self.vel = np.zeros(2)
        
        # RRT*全局规划
        print("  RRT*规划中...")
        self.global_planner = RRTStarPlanner(grid_map)
        self.global_path = self.global_planner.plan(start, goal)
        print(f"  RRT*路径点数: {len(self.global_path)}")
        
        # APF局部控制
        self.apf = APFController(grid_map)
        
        # 路径跟踪
        self.path_idx = 0
        self.lookahead = 5  # 前瞻点数
        
        # 导航参数
        self.max_speed = 1.0
        self.goal_threshold = 2.0
        
        # 动态障碍物
        self.dynamic_obstacles = []
        
    def set_dynamic_obstacles(self, obstacles):
        """设置动态障碍物"""
        self.dynamic_obstacles = obstacles
        
    def step(self):
        """
        执行一步导航
        
        Returns:
            pos: 当前位置
            done: 是否完成
            info: 额外信息
        """
        # 检查是否到达目标
        if np.linalg.norm(self.pos - self.goal) < self.goal_threshold:
            return self.pos, True, {'success': True, 'collision': False}
        
        # 获取局部目标点（沿全局路径的前瞻点）
        local_goal = self._get_local_goal()
        
        # 使用APF计算控制力
        force = self.apf.compute_force(self.pos, local_goal, self.dynamic_obstacles)
        
        # 速度更新
        self.vel = 0.7 * self.vel + 0.3 * force
        speed = np.linalg.norm(self.vel)
        if speed > self.max_speed:
            self.vel = self.vel / speed * self.max_speed
        
        # 位置更新
        self.pos = self.pos + self.vel
        
        # 更新路径索引
        self._update_path_index()
        
        # 检查碰撞
        ix, iy = int(self.pos[0]), int(self.pos[1])
        if not (0 <= ix < self.grid_map.shape[0] and 0 <= iy < self.grid_map.shape[1]):
            return self.pos, True, {'success': False, 'collision': True}
        if self.grid_map[ix, iy] == 1:
            return self.pos, True, {'success': False, 'collision': True}
        
        # 检查动态障碍物碰撞
        for obs in self.dynamic_obstacles:
            if np.linalg.norm(self.pos - np.array(obs['pos'])) < 2.0:
                return self.pos, True, {'success': False, 'collision': True}
            
        return self.pos, False, {'success': False, 'collision': False}
    
    def _get_local_goal(self):
        """获取局部目标点"""
        lookahead_idx = min(self.path_idx + self.lookahead, len(self.global_path) - 1)
        return self.global_path[lookahead_idx]
    
    def _update_path_index(self):
        """更新路径索引"""
        while self.path_idx < len(self.global_path) - 1:
            dist_to_next = np.linalg.norm(self.pos - self.global_path[self.path_idx + 1])
            if dist_to_next < 3.0:
                self.path_idx += 1
            else:
                break
    
    def get_path_length(self):
        """计算路径长度"""
        if len(self.global_path) < 2:
            return 0
        length = 0
        for i in range(len(self.global_path) - 1):
            length += np.linalg.norm(self.global_path[i+1] - self.global_path[i])
        return length


if __name__ == "__main__":
    # 测试
    from map_generator import MapGenerator
    
    map_gen = MapGenerator(80)
    static_map, start, goal = map_gen.get_map('complex')
    
    navigator = RRTStarAPFNavigator(static_map, start, goal)
    
    print(f"起点: {start}")
    print(f"终点: {goal}")
    print(f"路径长度: {navigator.get_path_length():.2f}")
    
    # 模拟导航
    steps = 0
    max_steps = 500
    while steps < max_steps:
        pos, done, info = navigator.step()
        steps += 1
        if done:
            if info['success']:
                print(f"成功到达目标！步数: {steps}")
            else:
                print(f"导航失败: {info}")
            break
    else:
        print(f"超过最大步数 {max_steps}")
