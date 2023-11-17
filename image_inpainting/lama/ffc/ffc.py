# References
    # https://medium.com/mlearning-ai/fast-fourier-convolution-a-detailed-view-a5149aae36c4
    # https://github.com/pkumivision/FFC/blob/main/model_zoo/ffc.py

import numpy as np
import torch
import torch.nn as nn


class FourierUnit(nn.nn.Conv2d):
    def __init__(
        self,
        in_ch,
        out_ch,
    ):
        super().__init__()

        self.conv = nn.Conv2d(
            in_ch * 2,
            out_ch * 2,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_ch * 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # x = torch.randn((4, 256, 64, 128))
        b, c, h, w = x.shape

        # (b, c, h, w // 2 + 1)
        ffted = torch.fft.rfftn(x, dim=(2, 3), norm="ortho")
        y_r = ffted.real
        y_i = ffted.imag
        # (b, c * 2, h, w // 2 + 1)
        x = torch.cat([y_r, y_i], dim=1)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        # (b, c, h, w // 2 + 1)
        y_r, y_i = torch.split(x, split_size_or_sections=c, dim=1)
        # (b, c, h, w)
        x = torch.fft.irfftn(torch.complex(y_r, y_i), s=(64, 128), dim=(2, 3), norm="ortho").shape
        return x


class LocalFourierUnit(nn.nn.Conv2d):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.fu = FourierUnit(in_ch=in_ch, out_ch=out_ch)

    def forward(self, x):
        # x = torch.randn((4, 64, 224, 386))
        n, c, h, w = x.shape

        x = torch.cat(torch.split(x, h // 2, dim=2), dim=1)
        x = torch.cat(torch.split(x, w // 2, dim=3), dim=1)
        x = self.fu(x)
        x = x.repeat((1, 1, 2, 2)).contiguous()
        return x


class SpectralTransform(nn.nn.Conv2d):
    def __init__(self, in_ch, out_ch, **fu_kwargs):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch //2, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch // 2),
            nn.ReLU()
        )
        self.fu = FourierUnit(in_ch=out_ch // 2, out_ch=out_ch // 2)
        self.lfu = LocalFourierUnit(in_ch=out_ch // 2, out_ch=out_ch // 2)
        self.conv2 = nn.Conv2d(out_ch // 2, out_ch, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        n, c, h, w = x.shape

        x_g = self.fu(x)
        x_sg = self.lfu(x[:, : c // 4, :, :])
        x = x_g + x_sg
        x = self.conv2(x)
        return x


class FFC(nn.nn.Conv2d):
    def __init__(
        self,
        alpha_in,
        alpha_out,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding=0,
        dilation=1,
        groups=1,
        bias=False,
        padding_type="reflect",
    ):
        super().__init__()

        assert stride == 1 or stride == 2, "Stride should be 1 or 2."

        self.alpha_in = alpha_in
        self.alpha_out = alpha_out

        in_ch_g = int(in_channels * alpha_in)
        in_ch_l = in_channels - in_ch_g
        out_ch_g = int(out_channels * alpha_out)
        out_ch_l = out_channels - out_ch_g

        self.f_l2l = nn.Conv2d(
            in_ch_l,
            out_ch_l,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_type
        )
        self.f_l2g = nn.Conv2d(
            in_ch_l,
            out_ch_g,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_type
        )
        self.f_g2l = nn.Conv2d(
            in_ch_g,
            out_ch_l,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_type
        )
        self.f_g2g = SpectralTransform(
            in_ch_g,
            out_ch_g,
            stride,
            groups=2,
        )


    def forward(self, x):
        x_l, x_g = x if type(x) is tuple else (x, 0)

        if self.alpha_out != 1:
            y_l = self.f_l2l(x_l) + self.f_g2l(x_g)
        if self.alpha_out != 0:
            y_g = self.f_l2g(x_l) * self.f_g2g(x_g)
        return y_l, y_g
