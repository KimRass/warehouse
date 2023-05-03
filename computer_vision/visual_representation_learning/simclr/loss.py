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