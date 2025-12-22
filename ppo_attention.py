"""
消融实验 - 双重注意力PPO（无A*全局路径引导）
有注意力机制但不使用A*路径作为输入
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding2D(nn.Module):
    """2D正弦位置编码"""
    def __init__(self, d_model, height, width):
        super().__init__()
        self.d_model = d_model
        
        pe = torch.zeros(height, width, d_model)
        for i in range(height):
            for j in range(width):
                for k in range(0, d_model, 2):
                    div_term = 10000.0 ** (k / d_model)
                    pe[i, j, k] = math.sin(i / div_term) + math.sin(j / div_term)
                    if k + 1 < d_model:
                        pe[i, j, k + 1] = math.cos(i / div_term) + math.cos(j / div_term)
        
        self.register_buffer('pe', pe.permute(2, 0, 1))
        
    def forward(self, x):
        return x + self.pe.unsqueeze(0)


class MultiHeadSelfAttention(nn.Module):
    """多头自注意力模块"""
    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B, N, C = x.shape
        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        
        return out


class SpatialAttentionCNN(nn.Module):
    """带空间注意力的CNN（无路径交叉注意力）"""
    
    def __init__(self):
        super().__init__()
        
        # CNN特征提取 - 保持较大的特征图以便注意力
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1), nn.ReLU(),  # [B, 32, 20, 20]
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), nn.ReLU(),  # [B, 64, 10, 10]
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), nn.ReLU(),  # [B, 128, 5, 5]
        )
        
        self.feature_dim = 128
        self.spatial_size = 5
        
        # 位置编码
        self.pos_encoding = PositionalEncoding2D(128, 5, 5)
        
        # 多头自注意力（用于空间关系建模）
        self.self_attention = MultiHeadSelfAttention(
            embed_dim=128,
            num_heads=8
        )
        
        # 输出投影
        self.output_proj = nn.Sequential(
            nn.Linear(128 * 25, 256),
            nn.ReLU(),
            nn.Linear(256, 64)
        )
        
    def forward(self, x):
        # x: [B, 3, 40, 40]
        batch_size = x.size(0)
        
        # CNN特征提取
        features = self.cnn(x)  # [B, 128, 5, 5]
        
        # 添加位置编码
        features = self.pos_encoding(features)
        
        # 展平为序列
        features_flat = features.flatten(2).transpose(1, 2)  # [B, 25, 128]
        
        # 自注意力
        att_out = self.self_attention(features_flat)  # [B, 25, 128]
        
        # 残差连接
        features_flat = features_flat + att_out
        
        # 展平并投影
        out = features_flat.flatten(1)  # [B, 25*128]
        out = self.output_proj(out)  # [B, 64]
        
        return out


class DualAttentionPPOActorCritic(nn.Module):
    """双重注意力PPO（无A*路径引导）
    
    注：这里使用空间自注意力，但不使用与A*路径的交叉注意力
    目的是验证A*路径引导的重要性
    """
    
    def __init__(self, action_dim=2):
        super().__init__()
        
        # 带注意力的CNN
        self.attention_cnn = SpatialAttentionCNN()
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
        
        # 注意力CNN特征提取（只有自注意力，无交叉注意力）
        img_features = self.attention_cnn(x_img)  # [B, 64]
        
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
    model = DualAttentionPPOActorCritic(action_dim=2)
    
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
    print(f"\n双重注意力PPO（无A*）参数量: {total_params:,}")
