"""
Global path planner for ablation experiments.
Provides A* planning functionality.
"""

import numpy as np
import heapq


class SmartAStarPlanner:
    """A* path planner for global path planning."""
    
    def __init__(self, grid_map):
        """
        Args:
            grid_map: np.ndarray of shape (H, W), 0=free, 1=obstacle
        """
        self.grid_map = grid_map
        self.rows, self.cols = grid_map.shape
        
    def plan(self, start, goal, dynamic_obstacles=None):
        """
        Plan a path from start to goal using A* algorithm.
        
        Args:
            start: np.ndarray or tuple/list, starting position (x, y)
            goal: np.ndarray or tuple/list, goal position (x, y)
            dynamic_obstacles: Optional list of dynamic obstacles (ignored in static planning)
            
        Returns:
            np.ndarray of shape (N, 2), path from start to goal
            Returns None if no path found
        """
        start = np.array(start, dtype=np.float32)
        goal = np.array(goal, dtype=np.float32)
        
        start_node = (int(start[0]), int(start[1]))
        goal_node = (int(goal[0]), int(goal[1]))
        
        # Check validity
        if not self._is_valid(start_node):
            start_node = self._find_nearest_free(start_node)
        if not self._is_valid(goal_node):
            goal_node = self._find_nearest_free(goal_node)
        
        if start_node is None or goal_node is None:
            return np.array([start, goal])
            
        open_set = []
        heapq.heappush(open_set, (0, start_node))
        came_from = {}
        g_score = {start_node: 0}
        f_score = {start_node: self._heuristic(start_node, goal_node)}
        
        ops = 0
        max_ops = 50000
        
        while open_set and ops < max_ops:
            ops += 1
            current = heapq.heappop(open_set)[1]
            
            if current == goal_node:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(np.array(current, dtype=np.float32))
                    current = came_from[current]
                path.append(np.array(start_node, dtype=np.float32))
                return np.array(path[::-1])
            
            # 8-connected neighbors
            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1), 
                          (-1,-1), (-1,1), (1,-1), (1,1)]:
                neighbor = (current[0] + dx, current[1] + dy)
                
                if not self._is_valid(neighbor):
                    continue
                    
                # Diagonal movement costs more
                move_cost = np.sqrt(dx**2 + dy**2)
                tentative_g = g_score[current] + move_cost
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self._heuristic(neighbor, goal_node)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        # No path found, return straight line
        return np.array([start, goal])
    
    def _heuristic(self, a, b):
        """Euclidean distance heuristic."""
        return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)
    
    def _is_valid(self, node):
        """Check if node is valid (within bounds and not an obstacle)."""
        x, y = node
        if x < 0 or x >= self.rows or y < 0 or y >= self.cols:
            return False
        return self.grid_map[x, y] == 0
    
    def _find_nearest_free(self, node):
        """Find nearest free cell using BFS."""
        queue = [node]
        visited = set()
        
        max_search = 1000
        count = 0
        
        while queue and count < max_search:
            count += 1
            curr = queue.pop(0)
            if self._is_valid(curr):
                return curr
            
            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                next_node = (curr[0]+dx, curr[1]+dy)
                if next_node not in visited:
                    if 0 <= next_node[0] < self.rows and 0 <= next_node[1] < self.cols:
                        visited.add(next_node)
                        queue.append(next_node)
        
        return None


if __name__ == "__main__":
    # Test
    from map_generator import MapGenerator
    
    mg = MapGenerator(80)
    grid, start, goal = mg.get_map('simple')
    
    planner = SmartAStarPlanner(grid)
    path = planner.plan(start, goal)
    
    print(f"Start: {start}")
    print(f"Goal: {goal}")
    if path is not None:
        print(f"Path length: {len(path)} points")
        total_dist = sum(np.linalg.norm(path[i+1] - path[i]) for i in range(len(path)-1))
        print(f"Total distance: {total_dist:.2f}")
    else:
        print("No path found")
