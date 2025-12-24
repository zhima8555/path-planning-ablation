import numpy as np
import matplotlib.pyplot as plt

class MapGenerator:
    def __init__(self, map_size=80):
        self.map_size = map_size

    def get_dynamic_obstacles(self, map_type):
        """Return fixed dynamic obstacles for specific maps.

        Returns a list of dicts: {'pos': np.array([x,y]), 'vel': np.array([vx,vy]), 'radius': float}
        """
        # 按你的要求：第3张(concave) 1个；第4张(narrow) 2个
        if map_type == 'concave':
            return [
                {
                    'pos': np.array([28.0, 36.0], dtype=np.float32),
                    'vel': np.array([0.25, -0.18], dtype=np.float32),
                    'radius': 2.0,
                }
            ]

        if map_type == 'narrow':
            return [
                {
                    'pos': np.array([39.0, 29.0], dtype=np.float32),
                    'vel': np.array([0.22, 0.25], dtype=np.float32),
                    'radius': 2.0,
                },
                {
                    'pos': np.array([55.0, 41.0], dtype=np.float32),
                    'vel': np.array([-0.18, 0.20], dtype=np.float32),
                    'radius': 2.0,
                }
            ]

        return []

    def get_map(self, map_type):
        """
        Generate specific map types strictly matching the reference image.
        Returns:
            grid: np.array (80x80), 1=obstacle, 0=free
            start_pos: np.array [x, y]
            goal_pos: np.array [x, y]
        """
        grid = np.zeros((self.map_size, self.map_size), dtype=np.float32)
        
        # 1. Basic Borders (Always present)
        grid[0:2, :] = 1.0; grid[-2:, :] = 1.0
        grid[:, 0:2] = 1.0; grid[:, -2:] = 1.0

        # Default positions
        start_pos = np.array([5.0, 5.0])
        goal_pos = np.array([70.0, 70.0])

        if map_type == 'simple':
            # Map 1: 简单静态环境 (图a) - 少量分散的小方块（统一大小）
            # 约20-25个小障碍物，随机分布，统一为4x4大小
            np.random.seed(42)
            
            # 生成分散的统一尺寸方块 (4x4)
            for _ in range(22):
                rx = np.random.randint(8, self.map_size-12)
                ry = np.random.randint(8, self.map_size-12)
                grid[rx:rx+4, ry:ry+4] = 1.0
            
            # 在对角线附近添加3-4个障碍物（起点5,5到终点70,70的直线路径）
            # 对角线方程: y = x (因为起点和终点在对角线上)
            grid[20:24, 20:24] = 1.0  # 对角线附近
            grid[37:41, 37:41] = 1.0  # 中部对角线
            grid[52:56, 52:56] = 1.0  # 对角线附近
            
        elif map_type == 'complex':
            # Map 2: 简单复杂环境 (图b) - 更多更密集但规整的方块
            # 23个4x4小方块 + 7个6x6大方块
            np.random.seed(101)
            
            # 第一批：统一的4x4小方块，规整分布
            positions_4x4 = [
                (10, 15), (10, 35), (10, 55), (10, 70),
                (18, 25), (18, 45), (18, 65),
                (28, 12), (28, 32), (28, 52),
                (38, 20), (38, 40), (38, 60),
                (48, 15), (48, 35), (48, 55), (48, 70),
                (58, 25), (58, 45), (58, 65),
                (68, 18), (68, 38), (68, 58)
            ]
            for rx, ry in positions_4x4:
                grid[rx:rx+4, ry:ry+4] = 1.0
            
            # 第二批：统一的6x6大方块（形成聚集区域）
            positions_6x6 = [
                (15, 60), (25, 20), (35, 68),
                (50, 28), (60, 50), (70, 12),
                (42, 48)
            ]
            for rx, ry in positions_6x6:
                grid[rx:rx+6, ry:ry+6] = 1.0

        elif map_type == 'concave':
            # Map 3: 复杂动态环境凹型 - 规整的L型、凹型障碍物
            
            # 左上角小方块
            grid[8:13, 70:75] = 1.0
            
            # 上部区域障碍物
            grid[18:24, 62:68] = 1.0
            grid[26:32, 72:78] = 1.0
            
            # 左侧规整L型障碍物（开口向右）
            grid[18:30, 8:14] = 1.0   # 竖
            grid[24:30, 14:22] = 1.0  # 横
            
            # 必经之路凹形障碍物1 - 对角线前段（开口向右下）
            grid[20:26, 18:28] = 1.0  # 上横
            grid[30:36, 18:28] = 1.0  # 下横
            grid[20:36, 18:24] = 1.0  # 左竖（凹形背面）
            
            # 左下规整L型障碍物
            grid[8:14, 42:52] = 1.0   # 横
            grid[8:18, 46:52] = 1.0   # 竖
            
            # 标准对称T型障碍物
            grid[38:44, 28:48] = 1.0  # 横（T的上部，宽度20）
            grid[44:58, 35:41] = 1.0  # 竖（T的下部主干，宽度6，居中）
            
            # 必经之路凹形障碍物2 - 对角线中后段（开口向左上）
            grid[48:54, 50:60] = 1.0  # 上横
            grid[58:64, 50:60] = 1.0  # 下横
            grid[48:64, 54:60] = 1.0  # 右竖（凹形背面）
            
            # 右下角规整L型（开口向左上）
            grid[66:72, 68:78] = 1.0  # 右竖
            grid[66:72, 62:78] = 1.0  # 底横
            
            # 散落的规整小方块（统一4x4）
            grid[12:16, 32:36] = 1.0
            grid[32:36, 62:66] = 1.0
            grid[42:46, 20:24] = 1.0
            grid[58:62, 30:34] = 1.0
            grid[68:72, 42:46] = 1.0
            grid[48:52, 70:74] = 1.0
            
            start_pos = np.array([5.0, 5.0])
            goal_pos = np.array([75.0, 75.0])

        elif map_type == 'narrow':
            # Map 4: 复杂静态环境狭窄障碍物
            
            # 左上角巨大斜三角形
            for i in range(30):
                grid[2:2+i, 78-i:78] = 1.0
            
            # 左侧大T型障碍物
            grid[18:24, 8:28] = 1.0   # 横（T的上部）
            grid[24:48, 15:21] = 1.0  # 竖（T的下部）
            
            # 左下小方块（调整位置避免紧贴）
            grid[8:12, 30:34] = 1.0
            grid[52:56, 8:12] = 1.0
            
            # 中部区域散落小方块（间隔分布，移除凹形上方紧贴的方块）
            grid[32:36, 36:40] = 1.0
            grid[44:48, 42:46] = 1.0
            grid[48:52, 32:36] = 1.0
            
            # 中下部L型障碍物（统一尺寸）
            grid[56:68, 28:34] = 1.0  # 竖（长度12）
            grid[62:68, 34:44] = 1.0  # 横（长度10）
            
            # 右侧中部凹型障碍物（开口向左）
            grid[38:44, 50:60] = 1.0  # 上横
            grid[50:56, 50:60] = 1.0  # 下横
            grid[38:56, 54:60] = 1.0  # 右竖
            
            # 右侧L型障碍物（与左侧统一尺寸）
            grid[58:70, 48:54] = 1.0  # 竖（长度12）
            grid[64:70, 54:64] = 1.0  # 横（长度10）
            
            # 右上角方块（不覆盖目标点）
            grid[68:74, 65:71] = 1.0
            
            # 上部小方块群（间隔分布，远离凹型）
            grid[30:34, 64:68] = 1.0
            grid[43:47, 68:72] = 1.0
            grid[52:56, 62:66] = 1.0
            
            # 中右小方块（远离凹型）
            grid[28:32, 44:48] = 1.0
            
            # 下部散落小方块（间隔分布）
            grid[68:72, 38:42] = 1.0
            grid[72:76, 50:54] = 1.0
            
            start_pos = np.array([5.0, 5.0])
            goal_pos = np.array([75.0, 75.0])

        else:
            print(f"Unknown map type: {map_type}, using simple.")
            
        return grid, start_pos, goal_pos

if __name__ == "__main__":
    gen = MapGenerator()
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    types = ['simple', 'complex', 'concave', 'narrow']
    
    for i, t in enumerate(types):
        grid, s, g = gen.get_map(t)
        axes[i].imshow(grid.T, cmap='Greys', origin='lower')
        axes[i].scatter(s[0], s[1], c='green', s=100, label='Start')
        axes[i].scatter(g[0], g[1], c='red', marker='*', s=200, label='Goal')
        axes[i].set_title(f"Type: {t}")
        axes[i].legend()
    
    plt.tight_layout()
    plt.savefig('map_preview_final.png')
    print("Final map preview saved to: map_preview_final.png")