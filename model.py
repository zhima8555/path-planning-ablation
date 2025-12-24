import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding2D(nn.Module):
    """2D正弦位置编码，为注意力机制提供位置信息"""
    def __init__(self, d_model, height, width):
        super().__init__()
        self.d_model = d_model
        self.height = height
        self.width = width

        # 创建2D位置编码
        pe = torch.zeros(height, width, d_model)
        for i in range(height):
            for j in range(width):
                for k in range(0, d_model, 2):
                    div_term = 10000.0 ** (k / d_model)
                    pe[i, j, k] = math.sin(i / div_term) + math.sin(j / div_term)
                    if k + 1 < d_model:
                        pe[i, j, k + 1] = math.cos(i / div_term) + math.cos(j / div_term)

        # 注册为buffer
        self.register_buffer('pe', pe.permute(2, 0, 1))  # [C, H, W]

    def forward(self, x):
        # x: [B, C, H, W]
        return x + self.pe.unsqueeze(0)

class MultiHeadSelfAttention(nn.Module):
    """多头自注意力模块 - 评估局部区域综合风险"""
    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == embed_dim

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, N, C], N=H*W
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, num_heads, N, head_dim]

        # 注意力计算
        attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        # 输出
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)

        return out

class MultiHeadCrossAttention(nn.Module):
    """多头交叉注意力模块 - 结合全局路径引导"""
    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.query = nn.Linear(embed_dim, embed_dim)  # 路径特征作为Query
        self.key = nn.Linear(embed_dim, embed_dim)    # 环境特征作为Key
        self.value = nn.Linear(embed_dim, embed_dim)  # 环境特征作为Value
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, env_features, path_features):
        # env_features: [B, N, C] 来自自注意力
        # path_features: [B, C] 路径特征
        B, N, C = env_features.shape

        # 准备Query, Key, Value
        Q = self.query(path_features).unsqueeze(1)  # [B, 1, C]
        K = self.key(env_features)  # [B, N, C]
        V = self.value(env_features)  # [B, N, C]

        # reshape for multi-head
        Q = Q.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # 注意力计算
        attn = (Q @ K.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        # 输出并融合
        out = (attn @ V).transpose(1, 2).reshape(B, C)
        out = self.proj(out)

        return out

class PathFeatureExtractor(nn.Module):
    """路径特征提取器 - 从全局路径通道提取特征"""
    def __init__(self, in_channels=1, feature_dim=256):
        super().__init__()
        self.path_cnn = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),  # 全局平均池化
            nn.Flatten(),
            nn.Linear(64, feature_dim)
        )

    def forward(self, path_channel):
        # path_channel: [B, 1, H, W] 或 [B, H, W]
        if path_channel.dim() == 3:
            path_channel = path_channel.unsqueeze(1)
        return self.path_cnn(path_channel)

class DualAttentionCNN(nn.Module):
    """带双重注意力的CNN特征提取器"""
    def __init__(self):
        super().__init__()
        # CNN特征提取
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4), nn.ReLU(),  # [B, 32, 9, 9]
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),  # [B, 64, 3, 3]
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),  # [B, 64, 1, 1]
        )

        # 计算展平后的维度
        self.feature_dim = 64 * 1 * 1  # 64

        # 位置编码（虽然在1x1特征图上作用不大，但保留架构）
        self.pos_encoding = PositionalEncoding2D(64, 1, 1)

        # 多头自注意力
        self.self_attention = MultiHeadSelfAttention(
            embed_dim=self.feature_dim,
            num_heads=8
        )

        # 路径特征提取器
        self.path_extractor = PathFeatureExtractor(feature_dim=self.feature_dim)

        # 多头交叉注意力
        self.cross_attention = MultiHeadCrossAttention(
            embed_dim=self.feature_dim,
            num_heads=8
        )

    def forward(self, x):
        # x: [B, 3, 40, 40]
        batch_size = x.size(0)

        # CNN特征提取
        cnn_features = self.cnn(x)  # [B, 64, 1, 1]

        # 添加位置编码
        cnn_features = self.pos_encoding(cnn_features)

        # 展平为序列
        cnn_flat = cnn_features.flatten(2).transpose(1, 2)  # [B, 1, 64]

        # 自注意力：评估风险
        self_att_out = self.self_attention(cnn_flat)  # [B, 1, 64]

        # 从输入中提取路径通道
        path_channel = x[:, 1:2, :, :]  # [B, 1, 40, 40]

        # 提取路径特征
        path_features = self.path_extractor(path_channel)  # [B, 64]

        # 交叉注意力：结合路径引导
        cross_att_out = self.cross_attention(self_att_out, path_features)  # [B, 64]

        return cross_att_out

class CascadedDualAttentionActorCritic(nn.Module):
    """级联双重注意力Actor-Critic模型"""
    def __init__(self, action_dim=2):
        super().__init__()

        # 双重注意力CNN
        self.dual_attention_cnn = DualAttentionCNN()
        self.cnn_feature_dim = 64  # 来自DualAttentionCNN

        # 速度向量编码
        self.vec_encoder = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU()
        )

        # 特征融合
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
        # 可学习的标准差参数
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
                # 使用更稳定的初始化
                nn.init.normal_(m.weight, 0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # 对于输出层使用较小的初始化
                if 'mu_head' in m.__class__.__name__ or 'std_head' in m.__class__.__name__ or 'value_head' in m.__class__.__name__:
                    nn.init.normal_(m.weight, 0, 0.01)
                else:
                    nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GRU):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        param.data.fill_(0)

    def forward(self, obs_dict, hidden_state=None):
        """
        前向传播

        Args:
            obs_dict: 包含'image'和'vector'的字典
                - 'image': [B, 3, 40, 40]
                - 'vector': [B, 2]
            hidden_state: [B, hidden_dim] GRU隐藏状态

        Returns:
            mu: 动作均值 [B, action_dim]
            std: 动作标准差 [B, action_dim]
            value: 状态价值 [B, 1]
            next_hidden: 下一时刻隐藏状态 [B, hidden_dim]
        """
        x_img = obs_dict['image']  # [B, 3, 40, 40]
        x_vec = obs_dict['vector']  # [B, 2]
        batch_size = x_img.size(0)

        # 双重注意力处理图像特征
        # CNN -> 自注意力 -> 交叉注意力
        img_features = self.dual_attention_cnn(x_img)  # [B, 64]

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
        mu = self.actor_mu(next_hidden)  # [B, action_dim]
        std = self.log_std.exp().expand_as(mu)  # [B, action_dim]

        # Critic输出
        value = self.critic(next_hidden)  # [B, 1]

        return mu, std, value, next_hidden

# 测试代码
if __name__ == "__main__":
    # 创建模型
    model = CascadedDualAttentionActorCritic(action_dim=2)

    # 创建测试输入
    batch_size = 4
    obs_dict = {
        'image': torch.randn(batch_size, 3, 40, 40),
        'vector': torch.randn(batch_size, 2)
    }
    hidden_state = torch.zeros(batch_size, 256)

    # 前向传播
    mu, std, value, next_hidden = model(obs_dict, hidden_state)

    print(f"mu shape: {mu.shape}")
    print(f"std shape: {std.shape}")
    print(f"value shape: {value.shape}")
    print(f"next_hidden shape: {next_hidden.shape}")

    # 打印模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")