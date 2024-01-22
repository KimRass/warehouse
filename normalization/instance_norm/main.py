# References
    # https://nn.labml.ai/normalization/instance_norm/index.html

import torch
import torch.nn as nn


class InstanceNorm(nn.Module):
    def __init__(self, n_features, eps=1e-5, affine=True):
        """
        `n_features`: Same as `num_features`.
        `affine`: Same as `affine`.
        """
        super().__init__()
    
        self.n_features = n_features
        self.eps = eps
        self.affine = affine

        if affine:
            self.gamma = nn.Parameter(torch.ones(n_features))
            self.beta = nn.Parameter(torch.zeros(n_features))

    def forward(self, x): # `(batch_size, n_features, ...)`
        ori_shape = x.shape
        b, c = ori_shape[: 2]
        assert self.n_features == c,\
            "The argument `n_features` must be same as the number of features of the input!"

        x = x.view(b, c, -1)

        # "$\mu_{ti} = \frac{1}{HW}\sum^{W}_{l=1}\sum^{H}_{m=1}x_{tilm}$"
        mean = x.mean(dim=-1, keepdim=True)
        # "$\sigma_{ti}^{2} = \frac{1}{HW}\sum^{W}_{l=1}\sum^{H}_{m=1}(x_{tilm} - \mu_{ti})$"
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        # "$y_{tijk} = \frac{x_{tijk} - \mu_{ti}}{\sqrt{\sigma_{ti}^{2} + \epsilon}}$"
        x = (x - mean) / torch.sqrt(var + self.eps)

        if self.affine:
            x = self.gamma.view(1, -1, 1) * x + self.beta.view(1, -1, 1)

        x = x.view(ori_shape)
        return x


if __name__ == "__main__":
    x = torch.randn((16, 3, 224, 256))
    inst_norm = InstanceNorm(n_features=3)
    inst_norm(x).shape
