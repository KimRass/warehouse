# References
    # https://github.com/sthalles/SimCLR/blob/master/simclr.py

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

BATCH_SIZE = 2
IMG_SIZE = 256
CROP_SIZE = 224
TILE_SIZE = CROP_SIZE // 2
GRAY_PROB = 0.67
TEMPERATURE = 3


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

    
features = torch.randn((2 * BATCH_SIZE, 4096))

# [(0, 1), (1, 0)] [(2, 3), (3, 2)]
pos_ids = sum([[2 * i + 1, 2 * i] for i in range(BATCH_SIZE)], [])
numer = torch.exp(F.cosine_similarity(features, features[pos_ids], dim=1) / TEMPERATURE)
numer

ids = torch.as_tensor([(i, j) for i, j in product(range(2 * BATCH_SIZE), range(2 * BATCH_SIZE)) if i != j])
ids

denom = torch.exp(F.cosine_similarity(features[ids[:, 0]], features[ids[:, 1]], dim=1) / TEMPERATURE)
denom
neg_ids = [list(range(i * (2 * BATCH_SIZE - 1), (i + 1) * (2 * BATCH_SIZE - 1))) for i in range(2 * BATCH_SIZE)]
neg_ids

numer

[numerdenom[i].sum() for i in neg_ids]
[denom[i] for i in neg_ids]


ids[:, 0]
[list(range(i * BATCH_SIZE, (i + 1) * BATCH_SIZE)) for i in range(2 * BATCH_SIZE)]
len([i for i in range(2 * BATCH_SIZE)])
