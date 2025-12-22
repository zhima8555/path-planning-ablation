"""
消融实验 - 统一评估脚本
评估所有算法在不同地图上的性能
"""

import torch
import numpy as np
import os
import sys
import time
import argparse
from collections import defaultdict
import matplotlib.pyplot as plt

sys.path.append('..')
from env import AutonomousNavEnv
from global_planner import SmartAStarPlanner
from map_generator import MapGenerator

# 导入所有算法
from astar_planner import AStarNavigator
from rrt_apf_planner import RRTStarAPFNavigator
from ppo_basic import BasicPPOActorCritic
from ppo_attention import DualAttentionPPOActorCritic
sys.path.append('..')
from model import CascadedDualAttentionActorCritic

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AlgorithmEvaluator:
    """算法评估器"""
    
    def __init__(self):
        self.map_gen = MapGenerator(80)
        self.map_types = ['simple', 'complex', 'concave', 'narrow']
        self.results = defaultdict(lambda: defaultdict(list))
        
    def evaluate_astar(self, num_trials=50):
        """评估纯A*算法"""
        print("\n评估: A* 算法")
        
        for map_type in self.map_types:
            successes = 0
            path_lengths = []
            times = []
            steps_list = []
            
            for trial in range(num_trials):
                static_map, start, goal = self.map_gen.get_map(map_type)
                
                start_time = time.time()
                navigator = AStarNavigator(static_map, start, goal)
                plan_time = time.time() - start_time
                
                # 执行导航
                steps = 0
                max_steps = 500
                
                while steps < max_steps:
                    pos, done, info = navigator.step()
                    steps += 1
                    if done:
                        break
                
                if info.get('success', False):
                    successes += 1
                    path_lengths.append(navigator.get_path_length())
                    times.append(plan_time)
                    steps_list.append(steps)
            
            success_rate = successes / num_trials * 100
            avg_path_length = np.mean(path_lengths) if path_lengths else 0
            avg_time = np.mean(times) if times else 0
            avg_steps = np.mean(steps_list) if steps_list else 0
            
            self.results['A*'][map_type] = {
                'success_rate': success_rate,
                'path_length': avg_path_length,
                'planning_time': avg_time * 1000,  # ms
                'steps': avg_steps
            }
            
            print(f"  {map_type}: 成功率={success_rate:.1f}%, 路径长度={avg_path_length:.1f}, 步数={avg_steps:.1f}")
    
    def evaluate_rrt_apf(self, num_trials=50):
        """评估RRT* + APF算法"""
        print("\n评估: RRT* + APF 算法")
        
        for map_type in self.map_types:
            successes = 0
            path_lengths = []
            times = []
            steps_list = []
            
            for trial in range(num_trials):
                static_map, start, goal = self.map_gen.get_map(map_type)
                
                start_time = time.time()
                try:
                    navigator = RRTStarAPFNavigator(static_map, start, goal)
                    plan_time = time.time() - start_time
                except Exception as e:
                    continue
                
                # 执行导航
                steps = 0
                max_steps = 500
                
                while steps < max_steps:
                    pos, done, info = navigator.step()
                    steps += 1
                    if done:
                        break
                
                if info.get('success', False):
                    successes += 1
                    path_lengths.append(navigator.get_path_length())
                    times.append(plan_time)
                    steps_list.append(steps)
            
            success_rate = successes / num_trials * 100
            avg_path_length = np.mean(path_lengths) if path_lengths else 0
            avg_time = np.mean(times) if times else 0
            avg_steps = np.mean(steps_list) if steps_list else 0
            
            self.results['RRT*+APF'][map_type] = {
                'success_rate': success_rate,
                'path_length': avg_path_length,
                'planning_time': avg_time * 1000,
                'steps': avg_steps
            }
            
            print(f"  {map_type}: 成功率={success_rate:.1f}%, 路径长度={avg_path_length:.1f}, 步数={avg_steps:.1f}")
    
    def evaluate_ppo_model(self, model, model_name, use_astar=True, num_trials=50):
        """评估PPO模型"""
        print(f"\n评估: {model_name}")
        
        model.eval()
        env = AutonomousNavEnv(map_type='simple')
        
        for map_type in self.map_types:
            successes = 0
            path_lengths = []
            steps_list = []
            
            for trial in range(num_trials):
                env.set_map_type(map_type)
                obs, _ = env.reset()
                
                # A*路径规划（如果启用）
                if use_astar:
                    planner = SmartAStarPlanner(env.static_map)
                    global_path = planner.plan(env.start_pos, env.goal_pos)
                    env.set_global_path(global_path)
                else:
                    env.set_global_path([])
                
                hidden_state = torch.zeros(1, 256).to(device)
                trajectory = [env.agent_pos.copy()]
                
                steps = 0
                max_steps = 500
                
                with torch.no_grad():
                    while steps < max_steps:
                        img = torch.FloatTensor(obs['image']).unsqueeze(0).to(device)
                        vec = torch.FloatTensor(obs['vector']).unsqueeze(0).to(device)
                        
                        mu, std, val, hidden_state = model({'image': img, 'vector': vec}, hidden_state)
                        action = mu.cpu().numpy().flatten()
                        
                        obs, reward, done, _, info = env.step(action)
                        trajectory.append(env.agent_pos.copy())
                        steps += 1
                        
                        if done:
                            break
                
                if info.get('success', False):
                    successes += 1
                    # 计算实际行走路径长度
                    traj_length = sum(np.linalg.norm(np.array(trajectory[i+1]) - np.array(trajectory[i])) 
                                     for i in range(len(trajectory)-1))
                    path_lengths.append(traj_length)
                    steps_list.append(steps)
            
            success_rate = successes / num_trials * 100
            avg_path_length = np.mean(path_lengths) if path_lengths else 0
            avg_steps = np.mean(steps_list) if steps_list else 0
            
            self.results[model_name][map_type] = {
                'success_rate': success_rate,
                'path_length': avg_path_length,
                'steps': avg_steps
            }
            
            print(f"  {map_type}: 成功率={success_rate:.1f}%, 路径长度={avg_path_length:.1f}, 步数={avg_steps:.1f}")
    
    def print_summary(self):
        """打印汇总结果"""
        print("\n" + "=" * 80)
        print("消融实验结果汇总")
        print("=" * 80)
        
        # 表头
        print(f"\n{'算法':<25} {'地图类型':<12} {'成功率(%)':<12} {'路径长度':<12} {'步数':<10}")
        print("-" * 80)
        
        for algo_name in self.results:
            for map_type in self.map_types:
                if map_type in self.results[algo_name]:
                    r = self.results[algo_name][map_type]
                    print(f"{algo_name:<25} {map_type:<12} {r['success_rate']:<12.1f} "
                          f"{r.get('path_length', 0):<12.1f} {r.get('steps', 0):<10.1f}")
        
        # 计算平均值
        print("-" * 80)
        print("\n平均成功率:")
        for algo_name in self.results:
            avg_success = np.mean([self.results[algo_name][mt]['success_rate'] 
                                  for mt in self.map_types if mt in self.results[algo_name]])
            print(f"  {algo_name}: {avg_success:.1f}%")
    
    def plot_results(self, save_path='ablation_results.png'):
        """绘制结果对比图"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        algorithms = list(self.results.keys())
        x = np.arange(len(self.map_types))
        width = 0.15
        
        colors = ['#3498DB', '#E74C3C', '#2ECC71', '#F39C12', '#9B59B6']
        
        # 成功率对比
        ax = axes[0]
        for i, algo in enumerate(algorithms):
            values = [self.results[algo].get(mt, {}).get('success_rate', 0) for mt in self.map_types]
            ax.bar(x + i * width, values, width, label=algo, color=colors[i % len(colors)])
        ax.set_ylabel('Success Rate (%)')
        ax.set_title('Success Rate Comparison')
        ax.set_xticks(x + width * (len(algorithms) - 1) / 2)
        ax.set_xticklabels(self.map_types)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # 路径长度对比
        ax = axes[1]
        for i, algo in enumerate(algorithms):
            values = [self.results[algo].get(mt, {}).get('path_length', 0) for mt in self.map_types]
            ax.bar(x + i * width, values, width, label=algo, color=colors[i % len(colors)])
        ax.set_ylabel('Path Length')
        ax.set_title('Path Length Comparison')
        ax.set_xticks(x + width * (len(algorithms) - 1) / 2)
        ax.set_xticklabels(self.map_types)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # 步数对比
        ax = axes[2]
        for i, algo in enumerate(algorithms):
            values = [self.results[algo].get(mt, {}).get('steps', 0) for mt in self.map_types]
            ax.bar(x + i * width, values, width, label=algo, color=colors[i % len(colors)])
        ax.set_ylabel('Steps')
        ax.set_title('Navigation Steps Comparison')
        ax.set_xticks(x + width * (len(algorithms) - 1) / 2)
        ax.set_xticklabels(self.map_types)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n结果图保存至: {save_path}")
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='消融实验评估')
    parser.add_argument('--trials', type=int, default=50, help='每种地图的测试次数')
    parser.add_argument('--skip-training', action='store_true', help='跳过传统算法评估')
    args = parser.parse_args()
    
    evaluator = AlgorithmEvaluator()
    
    # 评估传统算法
    if not args.skip_training:
        evaluator.evaluate_astar(num_trials=args.trials)
        evaluator.evaluate_rrt_apf(num_trials=args.trials)
    
    # 评估PPO模型（需要先训练）
    # 1. 基础PPO
    if os.path.exists('models_basic_astar/model.pth'):
        model = BasicPPOActorCritic(action_dim=2).to(device)
        model.load_state_dict(torch.load('models_basic_astar/model.pth'))
        evaluator.evaluate_ppo_model(model, 'PPO (Basic)', use_astar=True, num_trials=args.trials)
    
    # 2. 双重注意力PPO（无A*）
    if os.path.exists('models_attention_noastar/model.pth'):
        model = DualAttentionPPOActorCritic(action_dim=2).to(device)
        model.load_state_dict(torch.load('models_attention_noastar/model.pth'))
        evaluator.evaluate_ppo_model(model, 'Dual-Attention PPO', use_astar=False, num_trials=args.trials)
    
    # 3. 完整模型（A* + 双重注意力PPO）
    if os.path.exists('../best_model.pth'):
        model = CascadedDualAttentionActorCritic(action_dim=2).to(device)
        model.load_state_dict(torch.load('../best_model.pth'))
        evaluator.evaluate_ppo_model(model, 'A* + Dual-Attention PPO (Ours)', use_astar=True, num_trials=args.trials)
    
    # 打印结果
    evaluator.print_summary()
    
    # 绘制结果
    evaluator.plot_results()


if __name__ == "__main__":
    main()
