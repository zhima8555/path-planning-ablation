"""
Map generator for path planning ablation experiments.
Provides static maps and dynamic obstacles for evaluation.
"""

import numpy as np


class MapGenerator:
    """Map generator for creating various types of obstacle maps."""
    
    def __init__(self, map_size=80):
        """
        Args:
            map_size: Size of the square map grid
        """
        self.map_size = map_size
        self._map_cache = {}
        
    def get_map(self, map_type='simple'):
        """
        Get a map of the specified type with start and goal positions.
        
        Args:
            map_type: One of 'simple', 'complex', 'concave', 'narrow'
            
        Returns:
            Tuple of (grid_map, start_pos, goal_pos)
            - grid_map: np.ndarray of shape (map_size, map_size), 0=free, 1=obstacle
            - start_pos: np.ndarray of shape (2,), starting position
            - goal_pos: np.ndarray of shape (2,), goal position
        """
        # Use deterministic generation based on map type
        seed = {'simple': 42, 'complex': 100, 'concave': 200, 'narrow': 300}.get(map_type, 42)
        np.random.seed(seed)
        
        grid = np.zeros((self.map_size, self.map_size), dtype=np.int32)
        
        if map_type == 'simple':
            # Simple map with a few scattered obstacles
            num_obstacles = 20
            for _ in range(num_obstacles):
                x = np.random.randint(10, self.map_size - 10)
                y = np.random.randint(10, self.map_size - 10)
                size = np.random.randint(2, 5)
                grid[x:x+size, y:y+size] = 1
                
            start = np.array([10, 10], dtype=np.float32)
            goal = np.array([70, 70], dtype=np.float32)
            
        elif map_type == 'complex':
            # Complex map with more obstacles and irregular shapes
            num_obstacles = 40
            for _ in range(num_obstacles):
                x = np.random.randint(5, self.map_size - 5)
                y = np.random.randint(5, self.map_size - 5)
                size = np.random.randint(2, 8)
                grid[x:x+size, y:y+size] = 1
                
            start = np.array([10, 10], dtype=np.float32)
            goal = np.array([70, 70], dtype=np.float32)
            
        elif map_type == 'concave':
            # Concave obstacle (C-shaped or U-shaped)
            # Create a U-shaped obstacle in the center
            grid[25:55, 25:30] = 1  # left wall
            grid[25:55, 50:55] = 1  # right wall
            grid[50:55, 25:55] = 1  # bottom wall
            
            # Add some additional obstacles
            for _ in range(10):
                x = np.random.randint(5, self.map_size - 5)
                y = np.random.randint(5, self.map_size - 5)
                if not (20 < x < 60 and 20 < y < 60):  # avoid center
                    size = np.random.randint(2, 5)
                    grid[x:x+size, y:y+size] = 1
                    
            start = np.array([15, 40], dtype=np.float32)
            goal = np.array([65, 40], dtype=np.float32)
            
        elif map_type == 'narrow':
            # Narrow passage map
            # Create walls with a narrow passage
            grid[0:35, 35:45] = 1  # left barrier
            grid[45:80, 35:45] = 1  # right barrier
            
            # Add additional obstacles to make it challenging
            for _ in range(15):
                x = np.random.randint(5, self.map_size - 5)
                y = np.random.randint(5, self.map_size - 5)
                if not (30 < x < 50 and 30 < y < 50):  # avoid passage area
                    size = np.random.randint(2, 4)
                    grid[x:x+size, y:y+size] = 1
                    
            start = np.array([10, 60], dtype=np.float32)
            goal = np.array([70, 20], dtype=np.float32)
            
        else:
            # Default to simple
            start = np.array([10, 10], dtype=np.float32)
            goal = np.array([70, 70], dtype=np.float32)
        
        # Ensure start and goal are not in obstacles
        grid[int(start[0])-2:int(start[0])+3, int(start[1])-2:int(start[1])+3] = 0
        grid[int(goal[0])-2:int(goal[0])+3, int(goal[1])-2:int(goal[1])+3] = 0
        
        return grid, start, goal
    
    def get_dynamic_obstacles(self, map_type='simple'):
        """
        Get dynamic obstacles for the specified map type.
        
        Args:
            map_type: One of 'simple', 'complex', 'concave', 'narrow'
            
        Returns:
            List of dict, each containing:
            - 'pos': [x, y] initial position
            - 'vel': [vx, vy] velocity
            - 'radius': float, collision radius (default 2.0)
        """
        # Only concave and narrow maps have dynamic obstacles per problem statement
        if map_type == 'concave':
            return [
                {
                    'pos': [40.0, 40.0],
                    'vel': [0.3, 0.2],
                    'radius': 2.0,
                }
            ]
        elif map_type == 'narrow':
            return [
                {
                    'pos': [35.0, 40.0],
                    'vel': [0.2, 0.3],
                    'radius': 2.0,
                },
                {
                    'pos': [45.0, 40.0],
                    'vel': [-0.2, 0.2],
                    'radius': 2.0,
                }
            ]
        else:
            # Simple and complex maps have no dynamic obstacles
            return []


if __name__ == "__main__":
    # Test
    mg = MapGenerator(80)
    for map_type in ['simple', 'complex', 'concave', 'narrow']:
        grid, start, goal = mg.get_map(map_type)
        dyn_obs = mg.get_dynamic_obstacles(map_type)
        print(f"{map_type}: grid shape={grid.shape}, start={start}, goal={goal}, "
              f"dynamic_obs={len(dyn_obs)}")
