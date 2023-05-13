# References
    # https://github.com/sthalles/SimCLR/blob/master/simclr.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import DataLoader
import ssl

from .data_augmentation import (
    get_image_transformer,
    CustomDataset
)

ssl._create_default_https_context = ssl._create_unverified_context

BATCH_SIZE = 4096
IMG_SIZE = 256
CROP_SIZE = 224
TILE_SIZE = CROP_SIZE // 2
GRAY_PROB = 0.67
TEMPERATURE = 0.1 # "Table 5" in the paper


class ResNet50FeatureMapExtractor():
    def __init__(self, pretrained=False):
        if not pretrained:
            self.model = resnet50()
        else:
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
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # Order: image1_view1, image2_view1, ..., image2_view2, image2_view2, ...
        cos_sim_mat = F.cosine_similarity(x.unsqueeze(1), x.unsqueeze(0), dim=2)
        mat = torch.exp(cos_sim_mat / TEMPERATURE)
        mat.fill_diagonal_(0)
        
        numer = torch.cat([torch.diag(mat, -BATCH_SIZE), torch.diag(mat, BATCH_SIZE)], dim=0)
        denom = mat.sum(dim=0)
        
        loss = - torch.log(numer / denom).sum() / (2 * BATCH_SIZE)
        return loss


if __name__ == "__main__":
    criterion = NTXentLoss()

    transform = get_image_transformer()
    ds = CustomDataset(root="/Users/jongbeomkim/Documents/datasets/VOCdevkit/VOC2012/JPEGImages", transform=transform)
    dl = DataLoader(dataset=ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    for batch, (view1, view2) in enumerate(dl, start=1):
        if batch >= 11:
            break
        view = torch.cat([view1, view2], dim=0)
        # grid = batched_image_to_grid(image=view, n_cols=BATCH_SIZE, normalize=True)
        # save_image(
        #     img=grid,
        #     path=f"""/Users/jongbeomkim/Desktop/workspace/machine_learning/computer_vision/visual_representation_learning/simclr/voc2012_samples/{batch}.jpg"""
        # )
        
        feat_extractor = ResNet50FeatureMapExtractor()
        prj_head = ProjectionHead()
        feat_map = feat_extractor.get_feature_map(view)
        feat_map = prj_head(feat_map)
        
        loss = criterion(feat_map)
        loss