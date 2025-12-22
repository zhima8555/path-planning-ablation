"""
消融实验模块

包含以下对比算法：
1. A* - 纯A*路径规划
2. RRT* + APF - RRT*全局规划 + 人工势场局部避障
3. PPO (Basic) - 基础PPO，无注意力机制
4. Dual-Attention PPO - 双重注意力PPO，无A*引导
5. A* + Dual-Attention PPO (Ours) - 完整模型

使用方法：
---------
1. 训练单个模型:
   python train_ablation.py --model basic --episodes 3000
   python train_ablation.py --model attention --no-astar --episodes 3000

2. 批量训练所有模型:
   python run_all_experiments.py

3. 评估所有算法:
   python evaluate_all.py --trials 50

4. 直接测试传统算法:
   python astar_planner.py
   python rrt_apf_planner.py
"""

from .astar_planner import AStarPlanner, AStarNavigator
from .rrt_apf_planner import RRTStarPlanner, APFController, RRTStarAPFNavigator
from .ppo_basic import BasicPPOActorCritic
from .ppo_attention import DualAttentionPPOActorCritic

__all__ = [
    'AStarPlanner',
    'AStarNavigator', 
    'RRTStarPlanner',
    'APFController',
    'RRTStarAPFNavigator',
    'BasicPPOActorCritic',
    'DualAttentionPPOActorCritic',
]
