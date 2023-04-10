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


def apply_hide_and_seek(image, patch_size=56, hide_prob=0.5, mean=(0, 0, 0)):
    b, _, h, w = image.shape
    assert h % patch_size == 0 and w % patch_size == 0,\
        "`patch_size` argument should be a multiple of both the width and height of the input image"

    mean_tensor = torch.Tensor(mean)[None, :, None, None].repeat(b, 1, patch_size, patch_size)

    copied_image = image.clone()
    for i in range(h // patch_size):
        for j in range(w // patch_size):
            if random.random() < hide_prob:
                    continue
            copied_image[
                ..., i * patch_size: (i + 1) * patch_size, j * patch_size: (j + 1) * patch_size
            ] = mean_tensor
    return copied_image


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
        ]
    )
    ds = ImageFolder("/Users/jongbeomkim/Downloads/imagenet-mini/val", transform=transform)
    dl = DataLoader(dataset=ds, batch_size=16, shuffle=True, num_workers=4, drop_last=True)
    for batch, (image, label) in enumerate(dl, start=1):
        image = apply_hide_and_seek(image, patch_size=56)
        grid = batched_image_to_grid(image, normalize=True)
        
        # show_image(grid)
        save_image(grid, f"""/Users/jongbeomkim/Desktop/workspace/machine_learning/computer_vision/weakly_supervised_learning/hide_and_seek/samples/{batch}.jpg""")