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


def apply_cutmix(image, label):
    b, _, h, w = image.shape

    order = torch.randperm(b)
    shuffled_image = image[order]
    shuffled_label = label[order]

    lamb = random.random()
    region_x = random.randint(0, w)
    region_y = random.randint(0, h)
    region_w = region_h = (1 - lamb) ** 0.5

    xmin = max(0, int(region_x - region_w / 2))
    ymin = max(0, int(region_y - region_h / 2))
    xmax = max(w, int(region_x + region_w / 2))
    ymax = max(h, int(region_y + region_h / 2))

    image[:, :, ymin: ymax, xmin: xmax] = shuffled_image[:, :, ymin: ymax, xmin: xmax]
    lamb = 1 - (xmax - xmin) * (ymax - ymin) / (w * h)
    label = lamb * label + (1 - lamb) * shuffled_label
    return image, label


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
    n_classes = len(ds.classes)
    for batch, (image, label) in enumerate(dl, start=1):
        label = F.one_hot(label, num_classes=n_classes)
        cutmixed_image, cutmixed_label = apply_cutmix(image=image, label=label)
        grid = get_image_grid(cutmixed_image)

        save_image(img=grid, path=f"""/Users/jongbeomkim/Desktop/workspace/machine_learning/computer_vision/image_data_augmentation/cutmix/samples/imagenet_mini{batch}.jpg""")