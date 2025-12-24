"""
Stub for the full model (with A* path guidance).
This is imported by train_ablation.py as the 'full' model.
"""

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import math
    
    class CascadedDualAttentionActorCritic(nn.Module):
        """Full model with A* path cross-attention (stub - identical to attention model)."""
        
        def __init__(self, action_dim=2):
            super().__init__()
            
            # For simplicity, this stub is identical to the attention model
            # In a real implementation, this would include path cross-attention
            from ppo_attention import DualAttentionPPOActorCritic
            self.base_model = DualAttentionPPOActorCritic(action_dim=action_dim)
            
        def forward(self, obs_dict, hidden_state=None):
            return self.base_model(obs_dict, hidden_state)
            
        def load_state_dict(self, state_dict, strict=True):
            return self.base_model.load_state_dict(state_dict, strict=strict)
            
        def state_dict(self):
            return self.base_model.state_dict()
            
        def parameters(self):
            return self.base_model.parameters()
            
except ImportError:
    # PyTorch not available - create a minimal stub
    class CascadedDualAttentionActorCritic:
        """Stub when PyTorch is not available."""
        def __init__(self, action_dim=2):
            self.action_dim = action_dim
