# References
    # https://github.com/clvrai/Representation-Learning-by-Learning-to-Count/blob/master/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torchvision.models import alexnet
from itertools import combinations


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


class ContrastiveLoss(nn.Module):
    def __init__(self, model, img_size=228, M=10):
        super().__init__()

        self.img_size = img_size
        self.M = M
        cell_size = self.img_size // 2
        model = model

    def forward(self, x, y):
        model = Network()
        data = torch.randn((batch_size, 3, img_size, img_size))
        # data = torch.clamp(data, 0, 10)
        # data *= 2
        combs = list(combinations(range(batch_size), 2))
        x = data[[i[0] for i in combs]]
        y = data[[i[1] for i in combs]]

        cell1 = x[:, :, : cell_size, : cell_size]
        cell2 = x[:, :, cell_size:, : cell_size]
        cell3 = x[:, :, : cell_size, cell_size:]
        cell4 = x[:, :, cell_size:, cell_size:]
        resized_x = T.Resize((cell_size, cell_size))(x)
        resized_y = T.Resize((cell_size, cell_size))(y)

        cell1_feat = model(cell1)
        cell2_feat = model(cell2)
        cell3_feat = model(cell3)
        cell4_feat = model(cell4)
        resized_x_feat = model(resized_x)
        resized_y_feat = model(resized_y)

        summed_feat = (cell1_feat + cell2_feat + cell3_feat + cell4_feat)
        loss1 = F.mse_loss(resized_x_feat, summed_feat)
        # loss1[:, 0]
        # loss1[:, 1]
        # loss1[0]
        loss2 = max(0, self.M - F.mse_loss(resized_y_feat, summed_feat))
        loss1, loss2
        # loss = F.mse_loss(resized_x_feat, summed_feat) + max(0, self.M - F.mse_loss(resized_y_feat, summed_feat))
        # return loss
        return loss1, loss2

model = Network()
criterion = ContrastiveLoss(model=model)
img_size=228
batch_size=8
# x = torch.randn((1, 3, img_size, img_size))
# y = torch.randn((1, 3, img_size, img_size))
# contrastive_loss(x, y)

data = torch.randn((batch_size, 3, img_size, img_size))
combs = list(combinations(range(batch_size), 2))
x = data[[i[0] for i in combs]]
y = data[[i[1] for i in combs]]
# x[:, :, : cell_size, : cell_size].shape
criterion(x, y)