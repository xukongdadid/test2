"""PINN-DACLA（Hall2015 Route B）

结构：
- CNN -> LSTM -> Attention 得到特征
- 回归头输出 residual Δy_nn（21维）
- 最终输出：y_pred = y_phys + Δy_nn
- 域判别器 + GRL 做 domain-invariant 特征学习

注意：
- 这里的“PINN”是 Route B：物理信息引导（teacher baseline），不是直接对微分方程残差做强约束。
"""

import torch
import torch.nn as nn

from .grl import GRL
from .attention import AdditiveAttention


class ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int, k: int, dropout: float):
        super().__init__()
        self.conv = nn.Conv1d(cin, cout, kernel_size=k, padding=k//2)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.act(self.conv(x)))


class PINNDACLAHallB(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_dim: int = 21,
        conv_channels=(64,64,128),
        conv_kernel=5,
        lstm_hidden=128,
        lstm_layers=2,
        dropout=0.1,
        disc_hidden=(128,64),
    ):
        super().__init__()
        # CNN 特征提取（对时间序列做 1D 卷积）
        blocks = []
        c = in_features
        for cc in conv_channels:
            blocks.append(ConvBlock(c, cc, conv_kernel, dropout))
            c = cc
        self.cnn = nn.Sequential(*blocks)

        # LSTM
        self.lstm = nn.LSTM(
            input_size=c,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0
        )
        self.attn = AdditiveAttention(lstm_hidden)

        # residual 回归头
        self.reg = nn.Sequential(
            nn.Linear(lstm_hidden, lstm_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, out_dim),
        )

        # GRL + 域判别器
        self.grl = GRL(1.0)
        disc = []
        d = lstm_hidden
        for h in disc_hidden:
            disc += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        disc += [nn.Linear(d, 2)]
        self.disc = nn.Sequential(*disc)

    def forward(self, x_seq: torch.Tensor, y_phys: torch.Tensor, grl_lam: float = 1.0):
        # x_seq: (B,T,F)
        x = x_seq.permute(0,2,1)   # (B,F,T)
        x = self.cnn(x)            # (B,C,T)
        x = x.permute(0,2,1)       # (B,T,C)

        h, _ = self.lstm(x)        # (B,T,H)
        feat, alpha = self.attn(h) # (B,H)

        delta = self.reg(feat)     # (B,21)
        y_pred = y_phys + delta

        # domain logits
        self.grl.lam = float(grl_lam)
        feat_g = self.grl(feat)
        dom_logits = self.disc(feat_g)

        return {
            "y_pred": y_pred,
            "delta": delta,
            "dom_logits": dom_logits,
            "attn": alpha,
        }
