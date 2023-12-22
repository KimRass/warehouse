# References
    # https://github.com/bzhangGo/rmsnorm/blob/master/rmsnorm_torch.py

import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    """
        Root Mean Square Layer Normalization
    :param d: model size
    :param p: partial RMSNorm, valid value [0, 1], default -1.0 (disabled)
    :param eps:  epsilon value, default 1e-8
    :param bias: whether use bias term for RMSNorm, disabled by
        default because RMSNorm doesn't enforce re-centering invariance.
    """
    def __init__(self, normalized_shape, p=-1., eps=1e-10, bias=False):
        super().__init__()

        self.eps = eps
        self.normalized_shape = normalized_shape
        self.p = p
        self.bias = bias

        self.gamma = nn.Parameter(torch.ones(normalized_shape))
        if self.bias:
            self.beta = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        if self.p < 0. or self.p > 1.:
            norm_x = (x ** 2).sum(dim=-1, keepdim=True) ** 0.5
            d_x = self.normalized_shape
        else:
            partial_size = int(self.normalized_shape * self.p)
            partial_x, _ = torch.split(x, [partial_size, self.normalized_shape - partial_size], dim=-1)

            norm_x = (partial_x ** 2).sum(dim=-1, keepdim=True) ** 0.5
            d_x = partial_size

        x = x / (norm_x * (d_x ** -0.5) + self.eps)

        if self.bias:
            return self.gamma * x + self.beta
        return self.gamma * x


x = torch.randn((2, 3, 4))
normalized_shape = (4,)
axes = [-(i + 1) for i in range(len(normalized_shape))]
x.norm(p=2, dim=-1, keepdim=True)
(x ** 2).sum(dim=-1, keepdim=True) ** 0.5
(x ** 2).mean(dim=axes, keepdim=True)
# torch.norm(x, p=2, dim=-1, keepdim=True)