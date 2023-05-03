import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torchvision.models import alexnet, AlexNet_Weights
from torchviz import make_dot
import cv2
from PIL import Image
import random


class ContextFreeNetwork(nn.Module):
    def __init__(self, n_permutations=1000):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 96, kernel_size=11, stride=2, padding=2)
        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)

        self.flatten1 = nn.Flatten(start_dim=1, end_dim=3)
        # 차원이 안 맞는데?? 논문에 따르면 (256 * 4 * 4, 512)여야 함.
        self.fc6 = nn.Linear(256 * 2 * 2, 512)

        self.flatten2 = nn.Flatten(start_dim=0, end_dim=1)
        self.fc7 = nn.Linear(4608, 4096)
        self.fc8 = nn.Linear(4096, n_permutations)

    def forward(self, tiles):
        x = self.conv1(tiles)
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

        x = self.flatten1(x)
        x = self.fc6(x)

        x = torch.cat(
            [self.flatten2(i).unsqueeze(0) for i in torch.split(x, 9, dim=0)]
        )

        x = self.fc7(x)
        x = self.fc8(x)

        x = F.softmax(x, dim=1)
        return x
