# References
    # https://github.com/sthalles/SimCLR/blob/master/simclr.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageNet
from itertools import product
import cv2
from PIL import Image
from pathlib import Path
import numpy as np
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

BATCH_SIZE = 4
IMG_SIZE = 256
CROP_SIZE = 224
TILE_SIZE = CROP_SIZE // 2
GRAY_PROB = 0.67
TEMPERATURE = 3


class ResNet50FeatureMapExtractor():
    def __init__(self):
        self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

        for name, module in self.model.named_modules():
            if name == "avgpool":
                module.register_forward_hook(self.forward_hook_fn)
    
    def forward_hook_fn(self, module, input, output):
        self.feat_map = output.squeeze()

    def get_feature_map(self, x):
        self.model(x)
        return self.feat_map


class ProjectionHead(nn.Module):
    def __init__(self, output_dim=2048, latent_dim=128):
        super().__init__()

        self.output_dim = output_dim
        self.latent_dim = latent_dim
        
        self.fc1 = nn.Linear(output_dim, latent_dim)
        self.bn1 = nn.BatchNorm1d(latent_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(latent_dim, output_dim)
        self.bn2 = nn.BatchNorm1d(output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        return x


class NTXentLoss(nn.Module):
    def __init__(self, counting_feat_extractor, batch_size, crop_size=CROP_SIZE, M=10):
        super().__init__()

        self.counting_feat_extractor = counting_feat_extractor
        self.crop_size = crop_size
        self.M = M

        self.tile_size = self.crop_size // 2
        self.resize = T.Resize((self.tile_size, self.tile_size), antialias=True)
        ids = torch.as_tensor([(i, j) for i, j in product(range(batch_size), range(batch_size)) if i != j])
        self.x_ids, self.y_ids = ids[:, 0], ids[:, 1]


if __name__ == "__main__":
    transform = get_image_transformer()
    ds = CustomDataset(root="/Users/jongbeomkim/Documents/datasets/VOCdevkit/VOC2012/JPEGImages", transform=transform)
    dl = DataLoader(dataset=ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    for batch, (view1, view2) in enumerate(dl, start=1):
        if batch >= 2:
            break
        view = torch.cat([view1, view2], dim=0)
        # grid = batched_image_to_grid(image=view, n_cols=2, normalize=True)
        # show_image(grid)
        
        feat_extractor = ResNet50FeatureMapExtractor()
        prj_head = ProjectionHead()
        feat_map = feat_extractor.get_feature_map(view)
        feat_map = prj_head(feat_map)
        feat_map.shape

        cos_sim_mat = F.cosine_similarity(feat_map.unsqueeze(1), feat_map.unsqueeze(0), dim=2)
        mat = torch.exp(cos_sim_mat / TEMPERATURE)
        mat.fill_diagonal_(0)
        
        numer = torch.cat([torch.diag(mat, -BATCH_SIZE), torch.diag(mat, BATCH_SIZE)], dim=0)
        denom = mat.sum(dim=0)
        
        loss = - torch.log(numer / denom).sum() / (2 * BATCH_SIZE)

        



        norm_feat_map = F.normalize(feat_map, p=2, dim=1)

        feat_map.unsqueeze(0).shape
        feat_map.unsqueeze(1).shape





        # pos_ids = sum([[2 * i + 1, 2 * i] for i in range(BATCH_SIZE)], [])
        # numer = torch.exp(F.cosine_similarity(feat_map, feat_map[pos_ids], dim=1) / TEMPERATURE)
        # numer

        # ids = torch.as_tensor([(i, j) for i, j in product(range(2 * BATCH_SIZE), range(2 * BATCH_SIZE)) if i != j])
        # ids

        # denom = torch.exp(F.cosine_similarity(feat_map[ids[:, 0]], feat_map[ids[:, 1]], dim=1) / TEMPERATURE)
        # denom
        # neg_ids = [list(range(i * (2 * BATCH_SIZE - 1), (i + 1) * (2 * BATCH_SIZE - 1))) for i in range(2 * BATCH_SIZE)]
        # neg_ids

        # numer

        # [numerdenom[i].sum() for i in neg_ids]
        # [denom[i] for i in neg_ids]


        # ids[:, 0]
        # [list(range(i * BATCH_SIZE, (i + 1) * BATCH_SIZE)) for i in range(2 * BATCH_SIZE)]
        # len([i for i in range(2 * BATCH_SIZE)])

1, 3, 5, 7, 2, 4, 6, 8