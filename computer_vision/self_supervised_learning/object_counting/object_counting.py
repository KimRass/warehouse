# References
    # https://github.com/clvrai/Representation-Learning-by-Learning-to-Count/blob/master/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torchvision.models import alexnet
from itertools import combinations

# alexnet().features(torch.randn(8, 3, 224, 224)).shape
alexnet().features(torch.randn(8, 3, 128, 128)).shape
alexnet().features


class Network(nn.Module):
    def __init__(self, n_elems=1000):
        super().__init__()

        conv1 = nn.Conv2d(3, 96, kernel_size=11, stride=4)
        conv2 = nn.Conv2d(96, 256, kernel_size=5, padding=2)
        conv3 = nn.Conv2d(256, 384, kernel_size=3, padding=1)
        conv4 = nn.Conv2d(384, 384, kernel_size=3, padding=1)
        conv5 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        relu = nn.ReLU()
        maxpool = nn.MaxPool2d(kernel_size=3, stride=2)

        self.flatten = nn.Flatten(start_dim=1, end_dim=3)
        self.fc6 = nn.Linear(256 * 2 * 2, 4096)
        # self.dropout = nn.Dropout(0.5)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, n_elems)

    def forward(self, x):
        b, _, _, _ = x.shape
        
        x = torch.randn(8, 3, 112, 112)
        x = conv1(x)
        x = relu(x)
        x = maxpool(x)
        x = conv2(x)
        x = relu(x)
        x = maxpool(x)
        x = conv3(x)
        x = relu(x)
        x = conv4(x)
        x = relu(x)
        x = conv5(x)
        x = relu(x)
        x = maxpool(x)
        x.shape
        
        
        
        
        
        
        
        
        

        x = self.alexnet_conv(x)
        print(x.shape)
        x = self.flatten(x)
        # x = self.fc6(x)
        x = nn.Linear(x.numel() // b, 4096)(x)
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
    def __init__(self, img_size=228, M=10):
        super().__init__()

        self.img_size = img_size
        self.M = M
        self.cell_size = self.img_size // 2
        # self.model = model

    def forward(self, x):
        # model = Network()
        # data = torch.randn((batch_size, 3, img_size, img_size))
        # out = model(data)

        ids = torch.as_tensor(list(combinations(range(batch_size), 2)))
        x, y = x[ids[:, 0]], x[ids[:, 1]]

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
        loss2 = max(0, self.M - F.mse_loss(resized_y_feat, summed_feat))
        loss = loss1 + loss2
        return loss

img_size=256
cell_size=img_size // 2
batch_size=8
model = Network()
data = torch.randn((batch_size, 3, cell_size, cell_size))
model(data).shape

criterion = ContrastiveLoss(model=model)
# x = torch.randn((1, 3, img_size, img_size))
# y = torch.randn((1, 3, img_size, img_size))
# contrastive_loss(x, y)

combs = list(combinations(range(batch_size), 2))
x = data[[i[0] for i in combs]]
y = data[[i[1] for i in combs]]
# x[:, :, : cell_size, : cell_size].shape
criterion(x, y)