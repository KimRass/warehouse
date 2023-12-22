# References:
    # https://github.com/detkov/Convolution-From-Scratch/blob/main/convolution.py
    # https://towardsdatascience.com/tensorflow-for-computer-vision-how-to-implement-convolutions-from-scratch-in-python-609158c24f82
    # https://d2l.ai/chapter_computer-vision/transposed-conv.html

import numpy as np
from typing import List, Tuple, Union
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.stride = (stride, stride) if isinstance(stride, int) else stride
        self.padding = (padding, padding) if isinstance(padding, int) else padding

        self.weight = self._get_kernel()

    def _get_output_shape(self, input):
        b, _, h, w = input.shape
        return (
            b,
            self.out_channels,
            math.floor((h + self.padding[0] * 2 - (self.kernel_size[0] - 1) - 1) // self.stride[0] + 1),
            math.floor((w + self.padding[1] * 2 - (self.kernel_size[1] - 1) - 1) // self.stride[1] + 1),
        )
    
    def _pad(self, input):
        b, c, h, w = input.shape
        padded = torch.zeros(
            size=(b, c, h + self.padding[0] * 2, w + self.padding[1] * 2),
            dtype=input.dtype,
            device=input.device,
        )
        padded[:, :, self.padding[0]: self.padding[0] + h, self.padding[1]: self.padding[1] + w] = input
        return padded

    def _get_kernel(self):
        kernel = torch.randn(
            size=(self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1]),
            requires_grad=True,
        )
        return kernel

    def forward(self, input):
        out_shape = self._get_output_shape(input)
        input = self._pad(input)

        out = torch.zeros(size=out_shape, dtype=input.dtype, device=input.device)
        # Initialize
        for k in range(out_shape[1]):
            for i in range(0, out_shape[2], self.stride[0]):
                for j in range(0, out_shape[3], self.stride[0]):
                    out[:, k, i, j] = torch.sum(
                        input[
                            :,
                            :,
                            i: i + self.kernel_size[0],
                            j: j + kernel_size[1]
                        ] * self.weight[k: k + 1, ...].repeat(out_shape[0], 1, 1, 1)
                    )
        return out


class ConvTransposed2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()


BATCH_SIZE = 8
in_channels = 4
input = torch.randn(size=(BATCH_SIZE, in_channels, 14, 18), dtype=torch.float32)
input.shape
input.unfold(dimension=2, size=3, step=2).shape
out_channels = 32
kernel_size = (3, 5)
# stride = (2, 3)
stride = 3
padding = (2, 4)

conv1 = nn.Conv2d(
    in_channels=in_channels,
    out_channels=out_channels,
    kernel_size=kernel_size,
    stride=stride,
    padding=padding,
)
conv2 = Conv2d(
    in_channels=in_channels,
    out_channels=out_channels,
    kernel_size=kernel_size,
    stride=stride,
    padding=padding,
)
conv1.weight.shape, conv2.weight.shape
conv1(input).shape, conv2(input).shape



# x = torch.tensor([[0.0, 1.0], [2.0, 3.0]])[None, None, ...]
# k = torch.tensor([[0.0, 1.0], [2.0, 3.0]])[None, None, ...]
# F.conv_transpose2d(input=x, weight=k)