# References
    # https://github.com/clvrai/Representation-Learning-by-Learning-to-Count/blob/master/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torchvision.models import alexnet


class Network(nn.Module):
    def __init__(self, n_elems=1000):
        super().__init__()

        self.alexnet_conv = alexnet().features
        self.flatten = nn.Flatten(start_dim=1, end_dim=3)
        self.relu = nn.ReLU()
        self.fc6 = nn.Linear(256 * 2 * 2, 4096)
        # self.dropout = nn.Dropout(0.5)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, n_elems)

    def forward(self, x):
        # b, _, _, _ = x.shape
        x = self.alexnet_conv(x)
        x = self.flatten(x)
        x = self.fc6(x)
        x = self.relu(x)
        # x = self.dropout(x)
        x = self.fc7(x)
        x = self.relu(x)
        # x = self.dropout(x)
        x = self.fc8(x)
        x = self.relu(x)
        # x = self.dropout(x)
        return x

model = Network()
x = torch.randn((8, 3, 114, 114))
out = model(x)
out.shape



img_size = 228
x = torch.randn((1, 3, img_size, img_size))
y = torch.randn((1, 3, img_size, img_size))

T1x = x[:, :, : img_size // 2, : img_size // 2]
T2x = x[:, :, img_size // 2:, : img_size // 2]
T3x = x[:, :, : img_size // 2, img_size // 2:]
T4x = x[:, :, img_size // 2:, img_size // 2:]
Dx = T.Resize((img_size // 2, img_size // 2))(x)
Dy = T.Resize((img_size // 2, img_size // 2))(y)

phi_T1x = model(T1x)
phi_T2x = model(T2x)
phi_T3x = model(T3x)
phi_T4x = model(T4x)
phi_Dx = model(Dx)
phi_Dy = model(Dy)
