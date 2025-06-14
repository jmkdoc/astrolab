# norm.py
import torch
from torch import nn

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, weight_scaling: bool = True):
        super().__init__()
        self.eps = eps
        self.weight_scaling = weight_scaling
        if self.weight_scaling:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_buffer('weight', torch.tensor(1.0)) # A scalar 1.0 if no learnable weight

    def _norm(self, x):
        # RMS is root mean square
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x) # Ensure computation is in float then cast back
        return output * self.weight
