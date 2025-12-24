"""
Autonomous navigation environment for ablation experiments.
Provides gym-like interface for path planning with dynamic obstacles.
"""

import numpy as np
from map_generator import MapGenerator


class AutonomousNavEnv:
    """Autonomous navigation environment with dynamic obstacles."""
    
    def __init__(self, map_type='simple', map_size=80):
        """
        Args:
            map_type: Type of map ('simple', 'complex', 'concave', 'narrow')
            map_size: Size of the map grid
        """
        self.map_type = map_type
        self.map_size = map_size
        self.map_generator = MapGenerator(map_size)
        
        # Initialize map and positions
        self.static_map, self.start_pos, self.goal_pos = self.map_generator.get_map(map_type)
        self.dynamic_obstacles = self.map_generator.get_dynamic_obstacles(map_type)
        
        # Agent state
        self.agent_pos = self.start_pos.copy()
        self.agent_dir = 0.0  # direction in radians
        self.agent_vel = np.zeros(2, dtype=np.float32)
        
        # Global path (set by planner)
        self.global_path = []
        
        # Episode state
        self.prev_dist = np.linalg.norm(self.agent_pos - self.goal_pos)
        self.step_count = 0
        self.max_steps = 500
        
        # Dynamic obstacle state (for concave and narrow maps)
        self._init_dynamic_obstacles()
        
    def _init_dynamic_obstacles(self):
        """Initialize dynamic obstacles with proper state."""
        self._dyn_obs_state = []
        for obs in self.dynamic_obstacles:
            self._dyn_obs_state.append({
                'pos': np.array(obs['pos'], dtype=np.float32).copy(),
                'vel': np.array(obs.get('vel', [0.2, 0.2]), dtype=np.float32).copy(),
                'radius': float(obs.get('radius', 2.0)),
            })
    
    def set_map_type(self, map_type):
        """Change the map type and reset the environment."""
        self.map_type = map_type
        self.static_map, self.start_pos, self.goal_pos = self.map_generator.get_map(map_type)
        self.dynamic_obstacles = self.map_generator.get_dynamic_obstacles(map_type)
        self._init_dynamic_obstacles()
        
    def set_global_path(self, path):
        """Set the global path for path-guided navigation."""
        if isinstance(path, list):
            self.global_path = path
        elif isinstance(path, np.ndarray):
            self.global_path = path.tolist() if len(path) > 0 else []
        else:
            self.global_path = []
    
    def reset(self, seed=None):
        """
        Reset the environment.
        
        Args:
            seed: Optional random seed for reproducibility
            
        Returns:
            Tuple of (observation, info)
        """
        if seed is not None:
            np.random.seed(seed)
            # Regenerate map with seed for deterministic reset
            saved_type = self.map_type
            self.map_generator = MapGenerator(self.map_size)
            self.static_map, self.start_pos, self.goal_pos = self.map_generator.get_map(saved_type)
            self.dynamic_obstacles = self.map_generator.get_dynamic_obstacles(saved_type)
        
        self.agent_pos = self.start_pos.copy()
        self.agent_dir = 0.0
        self.agent_vel = np.zeros(2, dtype=np.float32)
        self.prev_dist = np.linalg.norm(self.agent_pos - self.goal_pos)
        self.step_count = 0
        
        self._init_dynamic_obstacles()
        
        obs = self._get_obs()
        info = {}
        return obs, info
    
    def step(self, action):
        """
        Take a step in the environment.
        
        Args:
            action: np.ndarray of shape (2,), [forward_speed, angular_velocity]
            
        Returns:
            Tuple of (observation, reward, done, truncated, info)
        """
        self.step_count += 1
        
        # Parse action
        forward = float(action[0])
        angular = float(action[1])
        
        # Update agent direction and velocity
        self.agent_dir += angular * 0.1  # scale angular velocity
        self.agent_dir = (self.agent_dir + np.pi) % (2 * np.pi) - np.pi
        
        # Update agent position
        dx = forward * np.cos(self.agent_dir)
        dy = forward * np.sin(self.agent_dir)
        self.agent_pos = self.agent_pos + np.array([dx, dy], dtype=np.float32)
        
        # Update dynamic obstacles
        self._update_dynamic_obstacles()
        
        # Check termination conditions
        done = False
        success = False
        collision = False
        info = {}
        
        # Check collision with static obstacles
        agent_x, agent_y = int(self.agent_pos[0]), int(self.agent_pos[1])
        if not (0 <= agent_x < self.map_size and 0 <= agent_y < self.map_size):
            done = True
            collision = True
        elif self.static_map[agent_x, agent_y] == 1:
            done = True
            collision = True
        
        # Check collision with dynamic obstacles
        for obs in self._dyn_obs_state:
            dist = np.linalg.norm(self.agent_pos - obs['pos'])
            if dist < obs['radius']:
                done = True
                collision = True
                break
        
        # Check goal reached
        dist_to_goal = np.linalg.norm(self.agent_pos - self.goal_pos)
        if dist_to_goal < 2.0:
            done = True
            success = True
        
        # Check max steps
        if self.step_count >= self.max_steps:
            done = True
        
        # Compute reward
        reward = self._compute_reward(dist_to_goal, collision, success)
        self.prev_dist = dist_to_goal
        
        # Get observation
        obs = self._get_obs()
        
        info['success'] = success
        info['collision'] = collision
        
        return obs, reward, done, False, info
    
    def _update_dynamic_obstacles(self):
        """Update positions of dynamic obstacles."""
        for obs in self._dyn_obs_state:
            obs['pos'] = obs['pos'] + obs['vel']
            
            # Bounce off walls
            if not (1 < obs['pos'][0] < self.map_size - 1):
                obs['vel'][0] *= -1
            if not (1 < obs['pos'][1] < self.map_size - 1):
                obs['vel'][1] *= -1
    
    def _compute_reward(self, dist_to_goal, collision, success):
        """Compute reward for the current step."""
        if success:
            return 100.0
        if collision:
            return -50.0
        
        # Distance-based reward
        reward = (self.prev_dist - dist_to_goal) * 10.0
        
        # Small penalty for each step
        reward -= 0.1
        
        return reward
    
    def _get_obs(self):
        """
        Get current observation.
        
        Returns:
            dict with keys:
            - 'image': np.ndarray of shape (3, 40, 40), local map view
            - 'vector': np.ndarray of shape (2,), [dist_to_goal, angle_to_goal]
        """
        # Create local map view (40x40 centered on agent)
        view_size = 40
        half_view = view_size // 2
        
        local_map = np.zeros((3, view_size, view_size), dtype=np.float32)
        
        agent_x, agent_y = int(self.agent_pos[0]), int(self.agent_pos[1])
        
        # Channel 0: static obstacles
        for i in range(view_size):
            for j in range(view_size):
                map_x = agent_x - half_view + i
                map_y = agent_y - half_view + j
                if 0 <= map_x < self.map_size and 0 <= map_y < self.map_size:
                    local_map[0, i, j] = self.static_map[map_x, map_y]
        
        # Channel 1: goal location
        goal_local_x = int(self.goal_pos[0] - agent_x + half_view)
        goal_local_y = int(self.goal_pos[1] - agent_y + half_view)
        if 0 <= goal_local_x < view_size and 0 <= goal_local_y < view_size:
            local_map[1, goal_local_x, goal_local_y] = 1.0
        
        # Channel 2: dynamic obstacles
        for obs in self._dyn_obs_state:
            obs_local_x = int(obs['pos'][0] - agent_x + half_view)
            obs_local_y = int(obs['pos'][1] - agent_y + half_view)
            if 0 <= obs_local_x < view_size and 0 <= obs_local_y < view_size:
                local_map[2, obs_local_x, obs_local_y] = 1.0
        
        # Vector observation
        dist_to_goal = np.linalg.norm(self.agent_pos - self.goal_pos)
        angle_to_goal = np.arctan2(
            self.goal_pos[1] - self.agent_pos[1],
            self.goal_pos[0] - self.agent_pos[0]
        ) - self.agent_dir
        angle_to_goal = (angle_to_goal + np.pi) % (2 * np.pi) - np.pi
        
        vector = np.array([dist_to_goal / 100.0, angle_to_goal / np.pi], dtype=np.float32)
        
        return {
            'image': local_map,
            'vector': vector
        }


if __name__ == "__main__":
    # Test
    env = AutonomousNavEnv(map_type='concave')
    obs, info = env.reset(seed=42)
    
    print(f"Observation keys: {obs.keys()}")
    print(f"Image shape: {obs['image'].shape}")
    print(f"Vector shape: {obs['vector'].shape}")
    print(f"Start: {env.start_pos}")
    print(f"Goal: {env.goal_pos}")
    print(f"Dynamic obstacles: {len(env._dyn_obs_state)}")
    
    # Take a few steps
    for i in range(5):
        action = np.array([0.5, 0.1], dtype=np.float32)
        obs, reward, done, truncated, info = env.step(action)
        print(f"Step {i}: reward={reward:.2f}, done={done}, info={info}")
        if done:
            break
