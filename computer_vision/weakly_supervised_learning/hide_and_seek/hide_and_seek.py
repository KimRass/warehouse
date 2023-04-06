import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision
import torchvision.transforms as T
from PIL import Image
import random
import numpy as np
import random
from pathlib import Path


def apply_hide_and_seek(image, patch_size=56, hide_prob=0.5, mean=(0.485, 0.456, 0.406)):
    b, _, h, w = image.shape
    repl_val = torch.Tensor(mean).unsqueeze(0).unsqueeze(2).unsqueeze(3).repeat(b, 1, h, w)

    copied_image = image.clone()
    for i in range(h // patch_size + 1):
        for j in range(w // patch_size + 1):
            if random.random() >= hide_prob:
                copied_image[
                    ..., i * patch_size: (i + 1) * patch_size, j * patch_size: (j + 1) * patch_size
                ] = repl_val[
                    ..., i * patch_size: (i + 1) * patch_size, j * patch_size: (j + 1) * patch_size
                ]
    return copied_image


class HideAndSeek(nn.Module):
    def __init__(self, patch_size=56, hide_prob=0.5, mean=(0.485, 0.456, 0.406)):
        super().__init__()

        self.patch_size = patch_size
        self.hide_prob = hide_prob
        self.mean = mean

    def forward(self, x):
        b, _, h, w = x.shape
        repl_val = torch.Tensor(self.mean).unsqueeze(0).unsqueeze(2).unsqueeze(3).repeat(b, 1, h, w)

        copied = x.clone()
        for i in range(h // self.patch_size + 1):
            for j in range(w // self.patch_size + 1):
                if random.random() < self.hide_prob:
                    continue
                copied[
                    ..., i * self.patch_size: (i + 1) * self.patch_size, j * self.patch_size: (j + 1) * self.patch_size
                ] = repl_val[
                    ..., i * self.patch_size: (i + 1) * self.patch_size, j * self.patch_size: (j + 1) * self.patch_size
                ]
        return copied


def batched_image_to_grid(image, normalize=False, mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)):
    b, _, h, w = image.shape
    pad = max(2, int(max(h, w) * 0.04))
    n_rows = int(b ** 0.5)
    grid = torchvision.utils.make_grid(tensor=image, nrow=n_rows, normalize=False, padding=pad)
    grid = grid.clone().permute((1, 2, 0)).detach().cpu().numpy()

    if normalize:
        grid *= variance
        grid += mean
    grid *= 255.0
    grid = np.clip(a=grid, a_min=0, a_max=255).astype("uint8")

    for k in range(n_rows + 1):
        grid[(pad + h) * k: (pad + h) * k + pad, :, :] = 255
        grid[:, (pad + h) * k: (pad + h) * k + pad, :] = 255
    return grid


def _convert_to_pil(img):
    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)
    return img


def show_image(img):
    copied_img = img.copy()
    copied_img = _convert_to_pil(copied_img)
    copied_img.show()


def save_image(img, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    _convert_to_pil(img.copy()).save(path)


if __name__ == "__main__":
    transform = T.Compose(
        [
            T.ToTensor(),
            T.Resize((224, 224)),
            T.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            ),
            # HideAndSeek()
        ]
    )
    ds = ImageFolder("/Users/jongbeomkim/Downloads/imagenet-mini/val", transform=transform)
    dl = DataLoader(dataset=ds, batch_size=16, shuffle=True, num_workers=4, drop_last=True)
    for batch, (image, label) in enumerate(dl, start=1):
        image = apply_hide_and_seek(image, patch_size=56)
        grid = batched_image_to_grid(image, normalize=True)
