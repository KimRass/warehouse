import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.ops import drop_block2d
from scipy.stats import bernoulli
import numpy as np


class DropBlock(nn.Module):
    def __init__(self, block_size, keep_prob=0.9) -> None:
        super().__init__()
    
        # keep_prob=0.9
        # block_size=11
        self.keep_prob = keep_prob
        self.block_size = block_size
    
    def forward(self, x):
        # x = torch.randn((8, 1, 10, 10))
        # block_size=5
        b, c, h, w = x.shape
        assert self.block_size <= min(h, w),\
            "The `block_size` parameter must be smaller than or equal to"
        gamma = (self.keep_prob * w * h) /\
            ((self.block_size ** 2) * (w - self.block_size + 1) * (h - self.block_size + 1))
        init_mask = 1 - torch.bernoulli(
            torch.full(
                size=(b, c, h - self.block_size + 1, w - self.block_size + 1), fill_value=gamma, dtype=x.dtype, device=x.device
            )
        )
        # (init_mask.sum() / (b * c * (h - block_size + 1) * (w - block_size + 1))) / gamma
        expanded_mask = F.pad(
            init_mask,
            # pad=(block_size // 2, block_size // 2, block_size // 2, block_size // 2),
            pad=(self.block_size // 2 * 2, self.block_size // 2 * 2, self.block_size // 2 * 2, self.block_size // 2 * 2),
            mode="constant",
            value=1
        )
        # init_mask[0, 0]
        # expanded_mask[0, 0]
        # expanded_mask.shape
        # mask = F.max_pool2d(1 - expanded_mask, kernel_size=block_size, stride=1, padding=block_size // 2)
        mask = 1 - F.max_pool2d(1 - expanded_mask, kernel_size=self.block_size, stride=1)
        # mask[0, 0]
        # mask.shape
        x = x * mask
        return x


dropblock = DropBlock(keep_prob=0.9, block_size=5)
x = torch.randn((8, 1, 10, 10))
y = dropblock(x)
