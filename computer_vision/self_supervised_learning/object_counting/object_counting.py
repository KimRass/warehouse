# References
    # https://github.com/clvrai/Representation-Learning-by-Learning-to-Count/blob/master/model.py

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
IMG_SIZE = 256
CROP_SIZE = 224
TILE_SIZE = CROP_SIZE // 2
GRAY_PROB = 0.67


def load_image(img_path):
    img_path = str(img_path)
    img = cv2.imread(img_path, flags=cv2.IMREAD_COLOR)
    img = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2RGB)
    return img


def _to_pil(img):
    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)
    return img


def show_image(img):
    copied_img = img.copy()
    copied_img = _to_pil(copied_img)
    copied_img.show()


def batched_image_to_grid(image, n_cols, normalize=False, mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)):
    b, _, h, w = image.shape
    assert b % n_cols == 0,\
        "The batch size should be a multiple of `n_cols` argument"
    pad = max(2, int(max(h, w) * 0.04))
    grid = torchvision.utils.make_grid(tensor=image, nrow=n_cols, normalize=False, padding=pad)
    grid = grid.clone().permute((1, 2, 0)).detach().cpu().numpy()

    if normalize:
        grid *= variance
        grid += mean
    grid *= 255.0
    grid = np.clip(a=grid, a_min=0, a_max=255).astype("uint8")

    for k in range(n_cols + 1):
        grid[:, (pad + h) * k: (pad + h) * k + pad, :] = 255
    for k in range(b // n_cols + 1):
        grid[(pad + h) * k: (pad + h) * k + pad, :, :] = 255
    return grid


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


class CountingFeatureExtractor(nn.Module):
    def __init__(self, feat_extractor, n_elems=1000):
        super().__init__()

        self.feat_extractor = feat_extractor

        self.flatten = nn.Flatten(start_dim=1, end_dim=3)
        # self.fc6 = nn.Linear(256 * 6 * 6, 4096)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, n_elems)

    def forward(self, x):
        x = self.feat_extractor(x)

        x = self.flatten(x)
        x = self.dropout(x)

        x = nn.Linear(x.shape[1], 4096)(x) # fc6
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc7(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc8(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


class ContrastiveLoss(nn.Module):
    def __init__(self, counting_feat_extractor, batch_size, crop_size=CROP_SIZE, M=10):
        super().__init__()

        self.counting_feat_extractor = counting_feat_extractor
        self.crop_size = crop_size
        self.M = M

        self.tile_size = self.crop_size // 2
        self.resize = T.Resize((self.tile_size, self.tile_size), antialias=True)
        ids = torch.as_tensor([(i, j) for i, j in product(range(batch_size), range(batch_size)) if i != j])
        self.x_ids, self.y_ids = ids[:, 0], ids[:, 1]

    def forward(self, image):
        tile1 = image[:, :, : self.tile_size, : self.tile_size]
        tile2 = image[:, :, self.tile_size:, : self.tile_size]
        tile3 = image[:, :, : self.tile_size, self.tile_size:]
        tile4 = image[:, :, self.tile_size:, self.tile_size:]
        resized = self.resize(image)

        tile1_feat = self.counting_feat_extractor(tile1)
        tile2_feat = self.counting_feat_extractor(tile2)
        tile3_feat = self.counting_feat_extractor(tile3)
        tile4_feat = self.counting_feat_extractor(tile4)
        resized_feat = self.counting_feat_extractor(resized)

        summed_feat = (tile1_feat + tile2_feat + tile3_feat + tile4_feat)
        loss1 = F.mse_loss(resized_feat[self.x_ids], summed_feat[self.x_ids], reduction="sum")
        loss2 = max(0, self.M - F.mse_loss(resized_feat[self.y_ids], summed_feat[self.y_ids], reduction="sum"))
        loss = loss1 + loss2
        return loss


class CustomDataset(Dataset):
    def __init__(self, root, transform=None):
        super().__init__()
        self.root = root
        self.transform = transform
        self.img_paths = list(map(str, Path(self.root).glob("*.jpg")))

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = _to_pil(load_image(img_path))

        if self.transform is not None:
            image = self.transform(image)
        return image


if __name__ == "__main__":
    transform1 = T.Compose(
        [
            T.ToTensor(),
            T.CenterCrop(IMG_SIZE),
            T.RandomCrop(CROP_SIZE),
            # T.Normalize(
            #     mean=[0.485, 0.456, 0.406],
            #     std=[0.229, 0.224, 0.225]
            # )
        ]
    )
    transform2 = T.RandomGrayscale(GRAY_PROB)
    ds = CustomDataset(root="/Users/jongbeomkim/Documents/datasets/VOCdevkit/VOC2012/JPEGImages", transform=transform1)
    dl = DataLoader(dataset=ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    for batch, image in enumerate(dl, start=1):
        image = transform2(image)

        # grid = batched_image_to_grid(image=image, n_cols=int(BATCH_SIZE ** 0.5))
        # show_image(grid)

        sample_feat_extractor = AlexNetFeatureExtractor()
        counting_feat_extractor = CountingFeatureExtractor(feat_extractor=sample_feat_extractor)
        criterion = ContrastiveLoss(counting_feat_extractor=counting_feat_extractor, batch_size=BATCH_SIZE)

        print(criterion(image))
        criterion.x_ids
        criterion.y_ids