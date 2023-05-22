# Refrences:
    # https://nn.labml.ai/normalization/layer_norm/index.html
    # chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/https://arxiv.org/pdf/1607.06450.pdf

import torch
import torch.nn as nn


# Unlike batch normalization, layer normalization does not impose any constraint
# on the size of the mini-batchand it can be used in the pure online regime with batch size 1.
class LayerNormalization(nn.Module):
    # `normalized_shape``: The shape of the elements (except the batch).
    # `eps``: Epsilon for numerical stability.
    # `elementwise_affine`: Whether to scale and shift the normalized value
    def __init__(self, normalized_shape, eps=1e-10, elementwise_affine=True):
        super().__init__()
        if isinstance(normalized_shape, int):
            self.normalized_shape = torch.Size((normalized_shape,))
        else:
            self.normalized_shape = torch.Size((normalized_shape[-1],))
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            self.gamma = nn.Parameter(torch.ones(normalized_shape))
            self.beta = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        axes = [-(i + 1) for i in range(len(self.normalized_shape))]
        mean = x.mean(dim=axes, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=axes, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        if self.elementwise_affine:
            x = self.gamma * x + self.beta
        return x


if __name__ == "__main__":
    ln = LayerNormalization((12, 18))
    x = torch.randn((4, 12, 18))
    ln(x).shape
