import torch
import torch.nn as nn
import torch.nn.functional as F


class DropBlock(nn.Module):
    def __init__(self, block_size, keep_prob=0.9) -> None:
        super().__init__()
    
        self.keep_prob = keep_prob
        self.block_size = block_size
    
    def forward(self, x):
        b, c, h, w = x.shape
        assert self.block_size <= min(h, w),\
            "The `block_size` parameter must be smaller than or equal to both input width and height."

        self.gamma = ((1 - self.keep_prob) * w * h) /\
            ((self.block_size ** 2) * (w - self.block_size + 1) * (h - self.block_size + 1))
        init_mask = 1 - torch.bernoulli(
            torch.full(
                size=(b, c, h - self.block_size + 1, w - self.block_size + 1),
                fill_value=self.gamma,
                dtype=x.dtype,
                device=x.device
            )
        )
        expanded_mask = F.pad(
            init_mask,
            # pad=(block_size // 2, block_size // 2, block_size // 2, block_size // 2),
            pad=(self.block_size // 2 * 2, self.block_size // 2 * 2, self.block_size // 2 * 2, self.block_size // 2 * 2),
            mode="constant",
            value=1
        )
        mask = 1 - F.max_pool2d(1 - expanded_mask, kernel_size=self.block_size, stride=1)
        x = x * mask
        x = x * mask.numel() / x.sum()
        return x
