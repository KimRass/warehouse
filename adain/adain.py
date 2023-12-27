# References
    # https://github.com/aadhithya/AdaIN-pytorch/blob/master/model.py

import torch
import torch.nn as nn


class AdaIN(nn.Module):
    # "Unlike BN, IN or CIN, AdaIN has no learnable affine parameters. Instead, it adaptively
    # computes the affine parameters from the style input."
    def __init__(self, eps=1e-5):
        super().__init__()

        self.eps = eps

    def _get_mean_and_std(self, x): # `(batch_size, n_channels, height, width)`
        b, c, _, _ = x.shape

        x = x.view(b, c, -1)
        # "Similar to IN, these statistics are computed across spatial locations."
        mean = x.mean(dim=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        std = torch.sqrt(var + self.eps)
        return mean.view(b, c, 1, 1), std.view(b, c, 1, 1)

    def forward(self, x, y):
        # "$\text{AdaIN}(x, y) = \sigma(y)\bigg(\frac{x - \mu(x)}{\sigma(x)}\bigg) + \mu(y)$"
        # "AdaIN receives a content input $x$ and a style input $y$, and simply aligns the channel-wise
        # mean and variance of $x$ to match those of $y$."
        x_mean, x_std = self._get_mean_and_std(x)
        y_mean, y_std = self._get_mean_and_std(y)
        # "The output produced by AdaIN will preserve the spatial structure of the content image."
        return y_std * ((x - x_mean) / x_std) + y_mean


if __name__ == "__main__":
    adain = AdaIN()
    x = torch.randn(16, 3, 224, 256)
    y = torch.randn(16, 3, 64, 128)
    out = adain(x, y)
    print(out.shape)
