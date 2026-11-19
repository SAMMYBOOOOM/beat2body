"""Adan optimizer (Xie et al., 2022), as used by EDGE (model/adan.py there, from lucidrains/Adan-pytorch)."""
import torch
from torch.optim import Optimizer


class Adan(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.02, 0.08, 0.01), eps=1e-8, weight_decay=0.0):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr, (b1, b2, b3), eps, wd = group["lr"], group["betas"], group["eps"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                st = self.state[p]
                if not st:
                    st["step"] = 0
                    st["m"], st["v"], st["n"] = torch.zeros_like(g), torch.zeros_like(g), torch.zeros_like(g)
                    st["prev_grad"] = torch.zeros_like(g)
                if st["step"] > 0:      # as in the reference: the first call only records the gradient
                    diff = g - st["prev_grad"]
                    st["m"].mul_(1 - b1).add_(g, alpha=b1)
                    st["v"].mul_(1 - b2).add_(diff, alpha=b2)
                    st["n"].mul_(1 - b3).add_((g + (1 - b2) * diff) ** 2, alpha=b3)
                st["step"] += 1
                k = st["step"]
                cm, cv, cn = 1 / (1 - (1 - b1) ** k), 1 / (1 - (1 - b2) ** k), 1 / (1 - (1 - b3) ** k)
                step = lr / ((st["n"] * cn).sqrt() + eps)
                p.addcmul_(step, st["m"] * cm + (1 - b2) * st["v"] * cv, value=-1.0)
                p.div_(1 + wd * lr)
                st["prev_grad"].copy_(g)
        return loss
