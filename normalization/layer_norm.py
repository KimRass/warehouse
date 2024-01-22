# Refrences:
    # https://nn.labml.ai/normalization/layer_norm/index.html
    # chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/https://arxiv.org/pdf/1607.06450.pdf

import torch
import torch.nn as nn

# Unlike batch normalization, layer normalization does not impose any constraint
# on the size of the mini-batch and it can be used in the pure online regime with batch size 1.
class LayerNorm(nn.Module):
    """
    `shape`: The shape of the elements (except the batch). Same as `normalized_shape`.
    `eps`: Epsilon for numerical stability.
    `affine_transform`: Whether to scale and shift the normalized value.
        Same as `elementwise_affine`.
    """
    def __init__(self, shape, eps=1e-5, affine_transform=False):
        super().__init__()

        if isinstance(shape, int):
            self.shape = torch.Size((shape,))
        if isinstance(shape, list) or isinstance(shape, tuple):
            self.shape = torch.Size(shape)
        self.eps = eps
        self.affine_transform = affine_transform

        if self.affine_transform:
            self.gamma = nn.Parameter(torch.ones(shape))
            self.beta = nn.Parameter(torch.zeros(shape))

    def forward(self, x): # The shape of `x` can be anything with batch size.
        assert self.shape == x.shape[-len(self.shape):],\
            "The argument `shape` should be the subset of the shape of the argument `x` from behind."

        axes = [-(i + 1) for i in range(len(self.shape))]
        mean = x.mean(dim=axes, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=axes, keepdim=True)
        # Same as `var = (x ** 2).mean(dim=axes, keepdim=True) - mean ** 2`
        x = (x - mean) / torch.sqrt(var + self.eps)
        if self.affine_transform:
            x = self.gamma * x + self.beta
        return x


if __name__ == "__main__":
    norm = LayerNorm(shape=(12, 18))
    x = torch.randn((4, 3, 12, 18))
    norm(x).shape
