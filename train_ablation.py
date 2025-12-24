"""
消融实验 - 统一训练脚本
支持训练不同的PPO变体模型
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import os
import sys
import argparse
import matplotlib.pyplot as plt
from datetime import datetime

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from env import AutonomousNavEnv
from global_planner import SmartAStarPlanner

# 导入不同的模型
from ppo_basic import BasicPPOActorCritic
from ppo_attention import DualAttentionPPOActorCritic
from model import CascadedDualAttentionActorCritic  # 完整模型

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# PPO超参数
LR = 3e-4
GAMMA = 0.99
EPS_CLIP = 0.2
K_EPOCHS = 10
BATCH_SIZE = 128
UPDATE_TIMESTEP = 512
MAX_EPISODES = 6000  # Updated to 6000 episodes as per requirements
ENTROPY_COEF = 0.01
GRAD_CLIP_NORM = 0.5
VALUE_LOSS_COEF = 0.5


class Memory:
    def __init__(self):
        self.clear()
    
    def clear(self):
        self.imgs = []
        self.vecs = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.dones = []
        self.hidden_states = []


class PPOTrainer:
    """PPO训练器 - 支持不同模型"""
    
    def __init__(self, model_type='full', use_astar=True):
        """
        Args:
            model_type: 'basic' | 'attention' | 'full'
            use_astar: 是否使用A*路径引导
        """
        self.model_type = model_type
        self.use_astar = use_astar
        
        # 选择模型
        if model_type == 'basic':
            self.policy = BasicPPOActorCritic(action_dim=2).to(device)
            self.policy_old = BasicPPOActorCritic(action_dim=2).to(device)
        elif model_type == 'attention':
            self.policy = DualAttentionPPOActorCritic(action_dim=2).to(device)
            self.policy_old = DualAttentionPPOActorCritic(action_dim=2).to(device)
        else:  # full
            self.policy = CascadedDualAttentionActorCritic(action_dim=2).to(device)
            self.policy_old = CascadedDualAttentionActorCritic(action_dim=2).to(device)
        
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.policy_old.eval()
        
        self.optimizer = optim.Adam(self.policy.parameters(), lr=LR)
        self.mse_loss = nn.MSELoss()
        
    def select_action(self, obs_dict, hidden_state, training=True):
        self.policy_old.eval()
        with torch.no_grad():
            img = torch.FloatTensor(obs_dict['image']).unsqueeze(0).to(device)
            vec = torch.FloatTensor(obs_dict['vector']).unsqueeze(0).to(device)
            mu, std, val, next_hidden = self.policy_old({'image': img, 'vector': vec}, hidden_state)
            
            if training:
                dist = Normal(mu, std)
                action = dist.sample()
                action_logprob = dist.log_prob(action).sum(dim=-1)
            else:
                action = mu
                action_logprob = torch.zeros(1)
                
        return action.cpu().numpy().flatten(), action_logprob.cpu().numpy().flatten(), next_hidden
    
    def update(self, memory):
        # 准备数据
        old_imgs = torch.FloatTensor(np.array(memory.imgs)).to(device)
        old_vecs = torch.FloatTensor(np.array(memory.vecs)).to(device)
        old_actions = torch.FloatTensor(np.array(memory.actions)).to(device)
        old_logprobs = torch.FloatTensor(np.array(memory.logprobs)).to(device)
        old_hiddens = torch.cat(memory.hidden_states, dim=0).to(device).detach()
        
        # 计算折扣奖励
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.dones)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + GAMMA * discounted_reward
            rewards.insert(0, discounted_reward)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        
        # 归一化
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        
        # PPO更新
        for _ in range(K_EPOCHS):
            # 随机打乱
            indices = torch.randperm(len(old_imgs))
            
            for start in range(0, len(old_imgs), BATCH_SIZE):
                end = min(start + BATCH_SIZE, len(old_imgs))
                batch_indices = indices[start:end]
                
                batch_imgs = old_imgs[batch_indices]
                batch_vecs = old_vecs[batch_indices]
                batch_actions = old_actions[batch_indices]
                batch_old_logprobs = old_logprobs[batch_indices]
                batch_rewards = rewards[batch_indices]
                batch_hiddens = old_hiddens[batch_indices]
                
                # 前向传播
                mu, std, values, _ = self.policy(
                    {'image': batch_imgs, 'vector': batch_vecs},
                    batch_hiddens
                )
                
                dist = Normal(mu, std)
                new_logprobs = dist.log_prob(batch_actions).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1)
                
                # 重要性采样比率
                ratios = torch.exp(new_logprobs - batch_old_logprobs)
                
                # 优势函数
                advantages = batch_rewards - values.squeeze().detach()
                
                # PPO损失
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - EPS_CLIP, 1 + EPS_CLIP) * advantages
                
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = self.mse_loss(values.squeeze(), batch_rewards)
                entropy_loss = -entropy.mean()
                
                loss = actor_loss + VALUE_LOSS_COEF * critic_loss + ENTROPY_COEF * entropy_loss
                
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), GRAD_CLIP_NORM)
                self.optimizer.step()
        
        # 更新旧策略
        self.policy_old.load_state_dict(self.policy.state_dict())


def train(model_type='full', use_astar=True, max_episodes=MAX_EPISODES, seed=None):
    """
    训练函数
    
    Args:
        model_type: 'basic' | 'attention' | 'full'
        use_astar: 是否使用A*路径引导
        max_episodes: 最大训练回合数
        seed: 随机种子（用于可复现性）
    """
    # Set random seed for reproducibility
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
    print("=" * 60)
    print(f"消融实验训练")
    print(f"  模型类型: {model_type}")
    print(f"  A*路径引导: {use_astar}")
    print(f"  设备: {device}")
    print("=" * 60)
    
    # 创建环境
    env = AutonomousNavEnv(map_type='simple')
    map_types = ['simple', 'complex', 'concave', 'narrow']
    
    # 创建训练器
    trainer = PPOTrainer(model_type=model_type, use_astar=use_astar)
    memory = Memory()
    
    # 训练记录
    episode_rewards = []
    success_rates = []
    
    timestep = 0
    
    for episode in range(max_episodes):
        # 随机选择地图类型
        map_type = map_types[episode % len(map_types)]
        env.set_map_type(map_type)
        obs, _ = env.reset()
        
        # A*路径规划（如果启用）
        if use_astar:
            planner = SmartAStarPlanner(env.static_map)
            global_path = planner.plan(env.start_pos, env.goal_pos)
            env.set_global_path(global_path)
        else:
            env.set_global_path([])
        
        episode_reward = 0
        hidden_state = torch.zeros(1, 256).to(device)
        
        for step in range(500):
            timestep += 1
            
            # 选择动作（注意：memory 里存的是“当前步”的 hidden_state，而不是 next_hidden）
            current_hidden = hidden_state
            action, logprob, next_hidden = trainer.select_action(obs, current_hidden)
            
            # 执行动作
            next_obs, reward, done, _, info = env.step(action)
            
            # 存储经验
            memory.imgs.append(obs['image'])
            memory.vecs.append(obs['vector'])
            memory.actions.append(action)
            memory.logprobs.append(logprob[0])
            memory.rewards.append(reward)
            memory.dones.append(done)
            memory.hidden_states.append(current_hidden.detach())

            hidden_state = next_hidden
            
            episode_reward += reward
            obs = next_obs
            
            # 更新策略
            if timestep % UPDATE_TIMESTEP == 0:
                trainer.update(memory)
                memory.clear()
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        
        # 计算成功率（最近100回合）
        if len(episode_rewards) >= 100:
            recent_success = sum([1 for r in episode_rewards[-100:] if r > 50]) / 100
            success_rates.append(recent_success)
        
        # 打印进度
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Episode {episode+1}/{max_episodes}, Avg Reward: {avg_reward:.2f}")
    
    # 保存模型到确定性位置
    # Deterministic checkpoint locations based on model type
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Save with deterministic names for evaluation pipeline
    if model_type == 'basic':
        model_path = os.path.join(checkpoint_dir, "model_basic_6k.pth")
    elif model_type == 'attention':
        model_path = os.path.join(checkpoint_dir, "model_attention_6k.pth")
    else:  # full
        model_path = os.path.join(checkpoint_dir, "model_full_6k.pth")
    
    torch.save(trainer.policy.state_dict(), model_path)
    
    # Also save to old location for backward compatibility
    save_dir = f"models_{model_type}_{'astar' if use_astar else 'noastar'}"
    os.makedirs(save_dir, exist_ok=True)
    torch.save(trainer.policy.state_dict(), f"{save_dir}/model.pth")
    
    # 保存训练曲线
    np.save(f"{save_dir}/rewards.npy", np.array(episode_rewards))
    
    print(f"\n训练完成！")
    print(f"  主要模型保存至: {model_path}")
    print(f"  备份保存至: {save_dir}/model.pth")
    
    return episode_rewards


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='消融实验训练')
    parser.add_argument('--model', type=str, default='full',
                       choices=['basic', 'attention', 'full'],
                       help='模型类型: basic(基础PPO), attention(双重注意力PPO), full(完整模型)')
    parser.add_argument('--no-astar', action='store_true',
                       help='不使用A*路径引导')
    parser.add_argument('--episodes', type=int, default=6000,
                       help='训练回合数 (默认6000)')
    parser.add_argument('--seed', type=int, default=None,
                       help='随机种子（用于可复现性）')
    
    args = parser.parse_args()
    
    train(
        model_type=args.model,
        use_astar=not args.no_astar,
        max_episodes=args.episodes,
        seed=args.seed
    )
