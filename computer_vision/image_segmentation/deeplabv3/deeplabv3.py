import torch
import torch.nn as nn


class Block(nn.Module):
    def __init__(self, in_ch, out_ch, rate, last_stride, multi_grid=(1, 2, 4)):
        super().__init__()
        rate = 8
        multi_grid=(1, 2, 4)

        self.last_stride = last_stride
        self.multi_grid = multi_grid

        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, dilation=rate * multi_grid[0])
        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, dilation=rate * multi_grid[1])
        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=last_stride, dilation=rate * multi_grid[2])

    def forward(self, x):
        


class DeepLabv3(nn.Module):
    def __init__(self):
        super().__init__()
        
        block4 = Block(rate=2, last_stride=2)
        block5 = Block(rate=4, last_stride=2)
        block6 = Block(rate=8, last_stride=2)
        block7 = Block(rate=16, last_stride=1)

    
    def forward(self, x):
        x = torch.randn((4, 32, 720, 480))