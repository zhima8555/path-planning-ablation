"""
消融实验 - 基础PPO模型（无注意力机制）
使用简单CNN替代双重注意力模块
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SimpleCNN(nn.Module):
    """简单CNN特征提取器（无注意力机制）"""
    
    def __init__(self):
        super().__init__()
        # 标准CNN特征提取
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4), nn.ReLU(),  # [B, 32, 9, 9]
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),  # [B, 64, 3, 3]
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),  # [B, 64, 1, 1]
        )
        self.feature_dim = 64
        
    def forward(self, x):
        # x: [B, 3, 40, 40]
        features = self.cnn(x)  # [B, 64, 1, 1]
        return features.flatten(1)  # [B, 64]


class BasicPPOActorCritic(nn.Module):
    """基础PPO Actor-Critic模型（无注意力机制）"""
    
    def __init__(self, action_dim=2):
        super().__init__()
        
        # 简单CNN特征提取
        self.cnn = SimpleCNN()
        self.cnn_feature_dim = 64
        
        # 速度向量编码
        self.vec_encoder = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU()
        )
        
        # 特征融合维度
        self.fusion_dim = self.cnn_feature_dim + 32  # 64 + 32 = 96
        
        # GRU时序建模
        self.hidden_dim = 256
        self.gru = nn.GRUCell(self.fusion_dim, self.hidden_dim)
        
        # Actor网络
        self.actor_mu = nn.Sequential(
            nn.Linear(self.hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh()
        )
        self.log_std = nn.Parameter(torch.zeros(1, action_dim))
        
        # Critic网络
        self.critic = nn.Sequential(
            nn.Linear(self.hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # 权重初始化
        self._initialize_weights()
        
    def _initialize_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
    def forward(self, obs_dict, hidden_state=None):
        """
        前向传播
        
        Args:
            obs_dict: {'image': [B,3,40,40], 'vector': [B,2]}
            hidden_state: [B, hidden_dim]
            
        Returns:
            mu, std, value, next_hidden
        """
        x_img = obs_dict['image']
        x_vec = obs_dict['vector']
        batch_size = x_img.size(0)
        
        # CNN特征提取（无注意力）
        img_features = self.cnn(x_img)  # [B, 64]
        
        # 速度向量编码
        vec_features = self.vec_encoder(x_vec)  # [B, 32]
        
        # 特征融合
        fusion_features = torch.cat([img_features, vec_features], dim=1)  # [B, 96]
        
        # 初始化隐藏状态
        if hidden_state is None:
            hidden_state = torch.zeros(batch_size, self.hidden_dim).to(x_img.device)
        
        # GRU时序建模
        next_hidden = self.gru(fusion_features, hidden_state)  # [B, 256]
        
        # Actor输出
        mu = self.actor_mu(next_hidden)
        std = self.log_std.exp().expand_as(mu)
        
        # Critic输出
        value = self.critic(next_hidden)
        
        return mu, std, value, next_hidden


if __name__ == "__main__":
    # 测试
    model = BasicPPOActorCritic(action_dim=2)
    
    batch_size = 4
    obs_dict = {
        'image': torch.randn(batch_size, 3, 40, 40),
        'vector': torch.randn(batch_size, 2)
    }
    
    mu, std, value, hidden = model(obs_dict)
    
    print(f"mu shape: {mu.shape}")
    print(f"std shape: {std.shape}")
    print(f"value shape: {value.shape}")
    print(f"hidden shape: {hidden.shape}")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n基础PPO参数量: {total_params:,}")
