"""GRL（Gradient Reversal Layer）用于域对抗训练"""

import torch


class _GRLFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lam: float):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_out):
        return -ctx.lam * grad_out, None


class GRL(torch.nn.Module):
    def __init__(self, lam: float = 1.0):
        super().__init__()
        self.lam = float(lam)

    def forward(self, x: torch.Tensor):
        return _GRLFn.apply(x, self.lam)
