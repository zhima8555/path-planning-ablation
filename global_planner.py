"""
Minimal stub global planner.
This allows train_ablation.py to run without external dependencies.
"""

import numpy as np


class SmartAStarPlanner:
    """Stub A* planner for path planning."""
    
    def __init__(self, static_map):
        self.static_map = static_map
        
    def plan(self, start, goal, dynamic_obstacles=None):
        """Returns a simple straight-line path from start to goal."""
        if start is None or goal is None:
            return []
            
        # Simple straight-line path
        num_waypoints = 10
        path = []
        for i in range(num_waypoints + 1):
            t = i / num_waypoints
            waypoint = start * (1 - t) + goal * t
            path.append(waypoint)
            
        return np.array(path, dtype=np.float32)
