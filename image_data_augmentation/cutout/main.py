import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision
import torchvision.transforms as T
from PIL import Image
import random
import numpy as np
from pathlib import Path


def apply_cutout(image, cutout_size=16, mean=(0.485, 0.456, 0.406)):
    b, _, h, w = image.shape

    x = random.randint(0, w)
    y = random.randint(0, h)
    xmin = max(0, x - cutout_size // 2)
    ymin = max(0, y - cutout_size // 2)
    xmax = max(0, x + cutout_size // 2)
    ymax = max(0, y + cutout_size // 2)

    image[:, 0, ymin: ymax, xmin: xmax] = mean[0]
    image[:, 1, ymin: ymax, xmin: xmax] = mean[1]
    image[:, 2, ymin: ymax, xmin: xmax] = mean[2]
    return image


def denormalize_array(img, mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)):
    copied_img = img.copy()
    copied_img *= variance
    copied_img += mean
    copied_img *= 255.0
    copied_img = np.clip(a=copied_img, a_min=0, a_max=255).astype("uint8")
    return copied_img


def _convert_to_pil(img):
    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)
    return img


def show_image(img):
    copied_img = img.copy()
    copied_img = _convert_to_pil(copied_img)
    copied_img.show()


def get_image_grid(image):
    grid = torchvision.utils.make_grid(
        tensor=image, nrow=int(image.shape[0] ** 0.5), normalize=False, padding=6
    )
    grid = grid.clone().permute((1, 2, 0)).detach().cpu().numpy()
    grid = denormalize_array(grid)
    return grid


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
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ]
    )
    ds = ImageFolder("/Users/jongbeomkim/Downloads/imagenet-mini/val", transform=transform)
    dl = DataLoader(dataset=ds, batch_size=16, shuffle=True, num_workers=4, drop_last=True)
    for batch, (image, label) in enumerate(dl, start=1):
        cutouted_image = apply_cutout(image, cutout_size=112)
        grid = get_image_grid(cutouted_image)
        show_image(grid)

        save_image(img=grid, path=f"""/Users/jongbeomkim/Desktop/workspace/machine_learning/computer_vision/image_data_augmentation/cutout/samples/imagenet_mini{batch}.jpg""")