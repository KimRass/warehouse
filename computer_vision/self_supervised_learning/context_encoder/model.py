# Refereces
    # https://github.com/jazzsaxmafia/Inpainting/blob/master/src/model.py
    # https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/linear.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torchvision.models import alexnet
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageNet
from itertools import product
import cv2
from PIL import Image
from pathlib import Path
import numpy as np
import math


BATCH_SIZE = 16
IMG_SIZE = 227
FEAT_DIM = 9216


class AlexNetFeatureExtractor(nn.Module):
    def __init__(self):
        super().__nn.init__()

        self.conv1 = nn.Conv2d(3, 96, kernel_size=11, stride=4)
        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.conv5(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x


class ChannelwiseFullyConnetecdLayer(nn.Module):
    def __init__(self, features: int, feature_size: int, device=None, dtype=None) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}

        super().__init__()

        self.features = features
        self.feature_size = feature_size

        self.weight = nn.Parameter(torch.empty(size=(feature_size ** 2, feature_size ** 2, features), **factory_kwargs))
        self.bias = nn.Parameter(torch.empty(size=(features, feature_size ** 2), **factory_kwargs))
        self.reset_parameters()
    
    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input.view((-1, self.features, self.feature_size ** 2))
        x = torch.einsum("bij,jji->bij", x, self.weight) + self.bias
        x = x.view((-1, self.features, self.feature_size, self.feature_size))
        return x

    def extra_repr(self) -> str:
        return f"""features={self.features}, feature_size={self.feature_size}"""


if __name__ == "__main__":
    feat_extractor = AlexNetFeatureExtractor()

    input = torch.randn((BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE))
    x = feat_extractor(input)

    b, m, n, n = x.shape
    cfc = ChannelwiseFullyConnetecdLayer(features=m, feature_size=n)
    cfc(x).shape
