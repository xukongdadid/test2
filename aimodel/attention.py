"""加性注意力（Bahdanau 风格）"""

import torch
import torch.nn as nn


class AdditiveAttention(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.W = nn.Linear(hidden, hidden)
        self.v = nn.Linear(hidden, 1, bias=False)

    def forward(self, h: torch.Tensor):
        # h: (B,T,H)
        u = torch.tanh(self.W(h))
        scores = self.v(u).squeeze(-1)      # (B,T)
        alpha = torch.softmax(scores, dim=1)
        ctx = torch.sum(h * alpha.unsqueeze(-1), dim=1)  # (B,H)
        return ctx, alpha
