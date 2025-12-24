"""
Complete model with cascaded dual attention for ablation experiments.
This is the "Ours" (full) model with both spatial attention and path cross-attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding2D(nn.Module):
    """2D positional encoding for spatial features."""
    def __init__(self, d_model, height, width):
        super().__init__()
        self.d_model = d_model
        
        pe = torch.zeros(height, width, d_model)
        
        # Precompute division terms for efficiency
        div_terms = torch.tensor([10000.0 ** (k / d_model) for k in range(0, d_model, 2)])
        
        for i in range(height):
            for j in range(width):
                for idx, k in enumerate(range(0, d_model, 2)):
                    div_term = div_terms[idx].item()
                    pe[i, j, k] = math.sin(i / div_term) + math.sin(j / div_term)
                    if k + 1 < d_model:
                        pe[i, j, k + 1] = math.cos(i / div_term) + math.cos(j / div_term)
        
        self.register_buffer('pe', pe.permute(2, 0, 1))
        
    def forward(self, x):
        return x + self.pe.unsqueeze(0)


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention module."""
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


class CrossAttention(nn.Module):
    """Cross-attention module for attending to global path."""
    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.kv_proj = nn.Linear(embed_dim, embed_dim * 2)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key_value):
        """
        Args:
            query: [B, N_q, C] - local features
            key_value: [B, N_kv, C] - global path features
        """
        B, N_q, C = query.shape
        N_kv = key_value.shape[1]
        
        q = self.q_proj(query).reshape(B, N_q, self.num_heads, self.head_dim).transpose(1, 2)
        kv = self.kv_proj(key_value).reshape(B, N_kv, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        
        attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        out = (attn @ v).transpose(1, 2).reshape(B, N_q, C)
        out = self.out_proj(out)
        
        return out


class CascadedDualAttentionCNN(nn.Module):
    """Cascaded dual attention CNN with both spatial and path attention."""
    
    def __init__(self):
        super().__init__()
        
        # CNN feature extraction
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1), nn.ReLU(),  # [B, 32, 20, 20]
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), nn.ReLU(),  # [B, 64, 10, 10]
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), nn.ReLU(),  # [B, 128, 5, 5]
        )
        
        self.feature_dim = 128
        self.spatial_size = 5
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding2D(128, 5, 5)
        
        # Spatial self-attention
        self.self_attention = MultiHeadSelfAttention(embed_dim=128, num_heads=8)
        
        # Path cross-attention (attends to global path)
        self.cross_attention = CrossAttention(embed_dim=128, num_heads=8)
        
        # Path encoder (for encoding global path waypoints)
        self.path_encoder = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 128)
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(128 * 25, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        
    def forward(self, x, global_path=None):
        """
        Args:
            x: [B, 3, 40, 40] - local observation
            global_path: Optional [B, N_path, 2] - global path waypoints
        """
        batch_size = x.size(0)
        
        # CNN feature extraction
        features = self.cnn(x)  # [B, 128, 5, 5]
        
        # Add positional encoding
        features = self.pos_encoding(features)
        
        # Flatten to sequence
        features_flat = features.flatten(2).transpose(1, 2)  # [B, 25, 128]
        
        # Spatial self-attention
        att_out = self.self_attention(features_flat)  # [B, 25, 128]
        features_flat = features_flat + att_out  # residual
        
        # Path cross-attention (if global path is provided)
        if global_path is not None and global_path.shape[1] > 0:
            path_features = self.path_encoder(global_path)  # [B, N_path, 128]
            cross_att_out = self.cross_attention(features_flat, path_features)
            features_flat = features_flat + cross_att_out  # residual
        
        # Flatten and project
        out = features_flat.flatten(1)  # [B, 25*128]
        out = self.output_proj(out)  # [B, 128]
        
        return out


class CascadedDualAttentionActorCritic(nn.Module):
    """Complete Actor-Critic model with cascaded dual attention.
    
    This is the "Ours" (full) model that includes:
    1. Spatial self-attention for local obstacle awareness
    2. Cross-attention with global A* path for guidance
    3. Temporal modeling with GRU
    """
    
    def __init__(self, action_dim=2):
        super().__init__()
        
        # Cascaded dual attention CNN
        self.attention_cnn = CascadedDualAttentionCNN()
        self.cnn_feature_dim = 128
        
        # Vector encoder (velocity and other state info)
        self.vec_encoder = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU()
        )
        
        # Feature fusion
        self.fusion_dim = self.cnn_feature_dim + 32  # 128 + 32 = 160
        
        # GRU for temporal modeling
        self.hidden_dim = 256
        self.gru = nn.GRUCell(self.fusion_dim, self.hidden_dim)
        
        # Actor network
        self.actor_mu = nn.Sequential(
            nn.Linear(self.hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh()
        )
        self.log_std = nn.Parameter(torch.zeros(1, action_dim))
        
        # Critic network
        self.critic = nn.Sequential(
            nn.Linear(self.hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
    def forward(self, obs_dict, hidden_state=None, global_path=None):
        """
        Forward pass.
        
        Args:
            obs_dict: dict with keys:
                - 'image': [B, 3, 40, 40]
                - 'vector': [B, 2]
            hidden_state: [B, hidden_dim]
            global_path: Optional [B, N_path, 2] - global path waypoints
            
        Returns:
            mu, std, value, next_hidden
        """
        x_img = obs_dict['image']
        x_vec = obs_dict['vector']
        batch_size = x_img.size(0)
        
        # Cascaded dual attention CNN
        img_features = self.attention_cnn(x_img, global_path)  # [B, 128]
        
        # Vector encoding
        vec_features = self.vec_encoder(x_vec)  # [B, 32]
        
        # Feature fusion
        fusion_features = torch.cat([img_features, vec_features], dim=1)  # [B, 160]
        
        # Initialize hidden state
        if hidden_state is None:
            hidden_state = torch.zeros(batch_size, self.hidden_dim).to(x_img.device)
        
        # GRU temporal modeling
        next_hidden = self.gru(fusion_features, hidden_state)  # [B, 256]
        
        # Actor output
        mu = self.actor_mu(next_hidden)
        std = self.log_std.exp().expand_as(mu)
        
        # Critic output
        value = self.critic(next_hidden)
        
        return mu, std, value, next_hidden


if __name__ == "__main__":
    # Test
    model = CascadedDualAttentionActorCritic(action_dim=2)
    
    batch_size = 4
    obs_dict = {
        'image': torch.randn(batch_size, 3, 40, 40),
        'vector': torch.randn(batch_size, 2)
    }
    
    # Without global path
    mu, std, value, hidden = model(obs_dict)
    print("Without global path:")
    print(f"  mu shape: {mu.shape}")
    print(f"  std shape: {std.shape}")
    print(f"  value shape: {value.shape}")
    print(f"  hidden shape: {hidden.shape}")
    
    # With global path
    global_path = torch.randn(batch_size, 10, 2)  # 10 waypoints
    mu, std, value, hidden = model(obs_dict, hidden, global_path)
    print("\nWith global path:")
    print(f"  mu shape: {mu.shape}")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")
