import heapq
import numpy as np

class SmartAStarPlanner:
    def __init__(self, grid_map):
        self.grid_map = grid_map
        self.rows, self.cols = grid_map.shape

    def plan(self, start, goal, dynamic_obstacles=[]):
        start_node = (int(start[0]), int(start[1]))
        goal_node = (int(goal[0]), int(goal[1]))
        
        # 兜底：如果出生在墙里，找最近的空地
        if not self._is_valid(start_node): start_node = self._find_nearest_free(start_node)
        if not self._is_valid(goal_node): goal_node = self._find_nearest_free(goal_node)

        open_set = []
        heapq.heappush(open_set, (0, start_node))
        came_from = {}
        g_score = {start_node: 0}
        f_score = {start_node: self._heuristic(start_node, goal_node)}
        
        path_found = False
        ops = 0
        
        # 限制搜索步数，防止死循环
        while open_set and ops < 5000:
            ops += 1
            current = heapq.heappop(open_set)[1]
            if current == goal_node:
                path_found = True
                break
            
            # 8 邻域
            for dx, dy in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
                neighbor = (current[0] + dx, current[1] + dy)
                if self._is_valid(neighbor):
                    dist = np.sqrt(dx**2 + dy**2)
                    tentative_g = g_score[current] + dist
                    
                    if neighbor not in g_score or tentative_g < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g
                        f_score[neighbor] = tentative_g + self._heuristic(neighbor, goal_node)
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        if path_found:
            path = []
            curr = goal_node
            while curr in came_from:
                path.append(list(curr))
                curr = came_from[curr]
            path.append(list(start_node))
            return np.array(path[::-1]) # 倒序
        
        return np.array([start, goal])

    def _heuristic(self, a, b):
        return np.linalg.norm(np.array(a) - np.array(b))

    def _is_valid(self, node):
        return (0 <= node[0] < self.rows and 0 <= node[1] < self.cols and self.grid_map[node[0]][node[1]] == 0)

    def _find_nearest_free(self, node):
        q = [node]
        visited = set()
        while q:
            curr = q.pop(0)
            if self._is_valid(curr): return curr
            for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                nxt = (curr[0]+dx, curr[1]+dy)
                if nxt not in visited and 0<=nxt[0]<self.rows and 0<=nxt[1]<self.cols:
                    visited.add(nxt)
                    q.append(nxt)
        return node