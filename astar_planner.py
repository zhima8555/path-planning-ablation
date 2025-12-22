"""
消融实验 - 纯A*路径规划算法
仅使用A*进行路径规划，不涉及强化学习
"""

import numpy as np
import heapq
import sys
sys.path.append('..')


class AStarPlanner:
    """纯A*路径规划器"""
    
    def __init__(self, grid_map):
        self.grid_map = grid_map
        self.rows, self.cols = grid_map.shape
        
    def plan(self, start, goal):
        """
        A*路径规划
        
        Args:
            start: 起点 (x, y)
            goal: 终点 (x, y)
            
        Returns:
            path: 路径点列表 [(x1,y1), (x2,y2), ...]
        """
        start_node = (int(start[0]), int(start[1]))
        goal_node = (int(goal[0]), int(goal[1]))
        
        # 检查起点终点有效性
        if not self._is_valid(start_node):
            start_node = self._find_nearest_free(start_node)
        if not self._is_valid(goal_node):
            goal_node = self._find_nearest_free(goal_node)
            
        open_set = []
        heapq.heappush(open_set, (0, start_node))
        came_from = {}
        g_score = {start_node: 0}
        f_score = {start_node: self._heuristic(start_node, goal_node)}
        
        ops = 0
        max_ops = 10000
        
        while open_set and ops < max_ops:
            ops += 1
            current = heapq.heappop(open_set)[1]
            
            if current == goal_node:
                # 重建路径
                path = []
                while current in came_from:
                    path.append(list(current))
                    current = came_from[current]
                path.append(list(start_node))
                return np.array(path[::-1])
            
            # 8邻域搜索
            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1), 
                          (-1,-1), (-1,1), (1,-1), (1,1)]:
                neighbor = (current[0] + dx, current[1] + dy)
                
                if not self._is_valid(neighbor):
                    continue
                    
                # 对角线移动代价更高
                move_cost = np.sqrt(dx**2 + dy**2)
                tentative_g = g_score[current] + move_cost
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self._heuristic(neighbor, goal_node)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        # 未找到路径，返回直线
        return np.array([list(start), list(goal)])
    
    def _heuristic(self, a, b):
        """欧几里得距离启发式"""
        return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)
    
    def _is_valid(self, node):
        """检查节点是否有效（在边界内且不是障碍物）"""
        x, y = node
        if x < 0 or x >= self.rows or y < 0 or y >= self.cols:
            return False
        return self.grid_map[x, y] == 0
    
    def _find_nearest_free(self, node):
        """BFS找最近的自由空间"""
        queue = [node]
        visited = set()
        
        while queue:
            curr = queue.pop(0)
            if self._is_valid(curr):
                return curr
            
            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                next_node = (curr[0]+dx, curr[1]+dy)
                if next_node not in visited:
                    if 0 <= next_node[0] < self.rows and 0 <= next_node[1] < self.cols:
                        visited.add(next_node)
                        queue.append(next_node)
        
        return node


class AStarNavigator:
    """A*导航器 - 用于评估"""
    
    def __init__(self, grid_map, start, goal):
        self.grid_map = grid_map
        self.start = np.array(start, dtype=np.float32)
        self.goal = np.array(goal, dtype=np.float32)
        self.pos = self.start.copy()
        
        # 规划路径
        self.planner = AStarPlanner(grid_map)
        self.path = self.planner.plan(start, goal)
        self.path_idx = 0
        
        # 导航参数
        self.step_size = 1.0
        self.goal_threshold = 2.0
        
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
        
        # 获取下一个路径点
        if self.path_idx < len(self.path) - 1:
            next_point = self.path[self.path_idx + 1]
            
            # 向下一个点移动
            direction = next_point - self.pos
            dist = np.linalg.norm(direction)
            
            if dist < self.step_size:
                self.pos = next_point.astype(np.float32)
                self.path_idx += 1
            else:
                direction = direction / dist
                self.pos = self.pos + direction * self.step_size
        
        # 检查碰撞
        ix, iy = int(self.pos[0]), int(self.pos[1])
        if not (0 <= ix < self.grid_map.shape[0] and 0 <= iy < self.grid_map.shape[1]):
            return self.pos, True, {'success': False, 'collision': True}
        if self.grid_map[ix, iy] == 1:
            return self.pos, True, {'success': False, 'collision': True}
            
        return self.pos, False, {'success': False, 'collision': False}
    
    def get_path_length(self):
        """计算路径长度"""
        if len(self.path) < 2:
            return 0
        length = 0
        for i in range(len(self.path) - 1):
            length += np.linalg.norm(self.path[i+1] - self.path[i])
        return length


if __name__ == "__main__":
    # 测试
    from map_generator import MapGenerator
    
    map_gen = MapGenerator(80)
    static_map, start, goal = map_gen.get_map('simple')
    
    navigator = AStarNavigator(static_map, start, goal)
    
    print(f"起点: {start}")
    print(f"终点: {goal}")
    print(f"路径长度: {navigator.get_path_length():.2f}")
    print(f"路径点数: {len(navigator.path)}")
    
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
