# Refereces
    # https://github.com/jazzsaxmafia/Inpainting/blob/master/src/model.py

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


BATCH_SIZE = 16
IMG_SIZE = 227
FEAT_DIM = 9216


class AlexNetFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()

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

feat_extractor = AlexNetFeatureExtractor()
input = torch.randn((BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE))


x = feat_extractor(input)

b, m, n, n = x.shape
x = x.view((b, m, n ** 2))

weight = nn.Parameter(torch.empty(size=(n ** 2, n ** 2, m)))
x = torch.einsum("bij,jji->bij", x, weight)

x.shape