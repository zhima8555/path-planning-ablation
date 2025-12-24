import numpy as np
import gymnasium as gym
from gymnasium import spaces
from map_generator import MapGenerator

class AutonomousNavEnv(gym.Env):
    def __init__(self, map_type='simple'):
        super(AutonomousNavEnv, self).__init__()
        self.map_size = 80
        
        # 40x40 高分辨率视野
        self.bev_size = 40   
        self.bev_res = 1.0   
        
        self.map_gen = MapGenerator(self.map_size)
        self.current_map_type = map_type
        
        # 多模态输入 (图片 + 速度向量)
        self.observation_space = spaces.Dict({
            "image": spaces.Box(low=0, high=1, shape=(3, self.bev_size, self.bev_size), dtype=np.float32),
            "vector": spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        })
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        
        self.reset()

    def set_map_type(self, map_type):
        self.current_map_type = map_type

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.static_map, self.start_pos, self.goal_pos = self.map_gen.get_map(self.current_map_type)
        
        self.agent_pos = np.array(self.start_pos) + np.random.uniform(-1, 1, 2)
        self.agent_dir = 0.0
        self.agent_vel = np.zeros(2) 
        
        self.prev_dist = np.linalg.norm(self.agent_pos - self.goal_pos)
        
        # 动态障碍物：优先使用地图的固定配置（用于复现实验）
        self.dynamic_obstacles = []
        fixed = []
        if hasattr(self.map_gen, 'get_dynamic_obstacles'):
            fixed = self.map_gen.get_dynamic_obstacles(self.current_map_type) or []

        if len(fixed) > 0:
            for obs in fixed:
                pos = np.array(obs['pos'], dtype=np.float32)
                vel = np.array(obs.get('vel', [0.2, 0.2]), dtype=np.float32)
                radius = float(obs.get('radius', 2.0))
                # 确保生成在空地
                px, py = int(pos[0]), int(pos[1])
                if 0 <= px < self.map_size and 0 <= py < self.map_size and self.static_map[px, py] == 0:
                    self.dynamic_obstacles.append({'pos': pos, 'vel': vel, 'radius': radius})
        else:
            # 兼容旧逻辑：随机动态障碍物（数量较少）
            if self.current_map_type == 'simple':
                num_obs = 0
            elif self.current_map_type == 'complex':
                num_obs = 2
            else:
                num_obs = 1

            for _ in range(num_obs):
                pos = np.random.uniform(20, self.map_size-20, 2)
                if self.static_map[int(pos[0]), int(pos[1])] == 0:
                    self.dynamic_obstacles.append({
                        'pos': pos.astype(np.float32),
                        'vel': np.random.uniform(-0.2, 0.2, 2).astype(np.float32),
                        'radius': 2.0,
                    })
        
        self.global_path = []
        return self._get_obs(), {}

    def set_global_path(self, path):
        self.global_path = path

    def step(self, action):
        # 动作映射: [-1, 1] -> [0, 1]
        v_cmd = (action[0] + 1.0) / 2.0 
        w_cmd = np.clip(action[1], -1.0, 1.0)
        
        self.agent_vel[0] = 0.7 * self.agent_vel[0] + 0.3 * v_cmd
        self.agent_vel[1] = 0.7 * self.agent_vel[1] + 0.3 * w_cmd
        v, omega = self.agent_vel
        
        self.agent_dir += omega * 0.6
        self.agent_pos[0] += v * np.cos(self.agent_dir)
        self.agent_pos[1] += v * np.sin(self.agent_dir)
        
        for obs in self.dynamic_obstacles:
            obs['pos'] += obs['vel']
            if not (1 < obs['pos'][0] < self.map_size-1): obs['vel'][0] *= -1
            if not (1 < obs['pos'][1] < self.map_size-1): obs['vel'][1] *= -1

        reward, done, info = self._compute_reward(v, omega)
        
        return self._get_obs(), reward, done, False, info

    def _compute_reward(self, v, omega):
        cur_dist = np.linalg.norm(self.agent_pos - self.goal_pos)

        # 极简奖励函数：只关注距离变化
        # 1. 进展奖励：靠近目标给正奖励，远离给负奖励
        r_progress = (self.prev_dist - cur_dist) * 2.0

        # 2. 额外的进展奖励：如果比开始位置更近，给予额外奖励
        if cur_dist < self.prev_dist:
            r_progress += 0.5

        # 3. 成功奖励
        r_success = 0
        if cur_dist < 1.0:
            r_success = 100.0

        # 4. 碰撞惩罚（轻微）
        r_collision = 0
        cx, cy = int(self.agent_pos[0]), int(self.agent_pos[1])
        if not (0 <= cx < self.map_size and 0 <= cy < self.map_size) or self.static_map[cx, cy] == 1:
            r_collision = -5.0

        # 5. 动态障碍物碰撞
        if r_collision == 0 and self.dynamic_obstacles:
            for obs in self.dynamic_obstacles:
                radius = float(obs.get('radius', 2.0))
                if np.linalg.norm(self.agent_pos - obs['pos']) < radius:
                    r_collision = -5.0
                    break

        # 累计奖励
        total = r_progress + r_success + r_collision

        # 更新之前距离
        self.prev_dist = cur_dist

        # 终止条件
        done = (r_success > 0) or (r_collision < 0)
        info = {'success': r_success > 0, 'collision': r_collision < 0}

        return total, done, info

    def _get_obs(self):
        bev = np.zeros((3, self.bev_size, self.bev_size), dtype=np.float32)
        cos_t, sin_t = np.cos(self.agent_dir), np.sin(self.agent_dir)
        center = self.bev_size // 2
        
        def draw_point(gx, gy, ch):
            dx, dy = gx - self.agent_pos[0], gy - self.agent_pos[1]
            lx = dx*cos_t + dy*sin_t
            ly = -dx*sin_t + dy*cos_t
            r = center - int(lx / self.bev_res)
            c = center - int(ly / self.bev_res)
            if 0 <= r < self.bev_size and 0 <= c < self.bev_size:
                bev[ch, r, c] = 1.0

        # 画局部地图
        start_x, end_x = int(self.agent_pos[0]-25), int(self.agent_pos[0]+25)
        start_y, end_y = int(self.agent_pos[1]-25), int(self.agent_pos[1]+25)
        for i in range(start_x, end_x):
            for j in range(start_y, end_y):
                if 0<=i<self.map_size and 0<=j<self.map_size and self.static_map[i,j] == 1:
                    draw_point(i, j, 0)
        
        if len(self.global_path) > 0:
            for pt in self.global_path[::2]:
                draw_point(pt[0], pt[1], 1)

        for obs in self.dynamic_obstacles:
            ox, oy = obs['pos']
            for i in [-0.5, 0, 0.5]:
                for j in [-0.5, 0, 0.5]:
                    draw_point(ox+i, oy+j, 2)

        return {
            "image": bev, 
            "vector": self.agent_vel.astype(np.float32)
        }