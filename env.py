"""
Minimal stub environment for training simulation.
This allows train_ablation.py to run without external dependencies.
"""

import numpy as np


class AutonomousNavEnv:
    """Stub environment for path planning training."""
    
    def __init__(self, map_type='simple'):
        self.map_type = map_type
        self.map_size = 80
        self.agent_pos = None
        self.goal_pos = None
        self.start_pos = None
        self.static_map = None
        self.dynamic_obstacles = []
        self.global_path = []
        
    def set_map_type(self, map_type):
        self.map_type = map_type
        
    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        
        # Random start and goal positions
        self.start_pos = np.random.rand(2) * (self.map_size - 20) + 10
        self.goal_pos = np.random.rand(2) * (self.map_size - 20) + 10
        self.agent_pos = self.start_pos.copy()
        
        # Simple static map (stub)
        self.static_map = np.zeros((self.map_size, self.map_size))
        
        obs = {
            'image': np.random.rand(3, 40, 40).astype(np.float32),
            'vector': np.random.rand(2).astype(np.float32)
        }
        info = {}
        return obs, info
    
    def set_global_path(self, path):
        self.global_path = path
        
    def step(self, action):
        # Stub step function - simulates environment dynamics
        self.agent_pos += action * 2.0
        
        # Calculate distance to goal
        dist_to_goal = np.linalg.norm(self.agent_pos - self.goal_pos)
        
        # Determine reward and done
        if dist_to_goal < 5.0:
            reward = 100.0
            done = True
        else:
            reward = -0.1 - dist_to_goal * 0.01  # Small negative reward
            done = False
            
        # Check episode length (simulate timeout)
        if not done and np.random.rand() < 0.01:  # 1% chance of timeout per step
            done = True
            
        obs = {
            'image': np.random.rand(3, 40, 40).astype(np.float32),
            'vector': np.random.rand(2).astype(np.float32)
        }
        info = {'distance_to_goal': dist_to_goal}
        
        return obs, reward, done, False, info
