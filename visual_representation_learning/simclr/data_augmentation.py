import torchvision
import torchvision.transforms as T
from torchvision.models import alexnet
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np

from process_images import (
    load_image,
    _to_pil,
    show_image
)

IMG_SIZE = 224
BATCH_SIZE = 2


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


class CustomDataset(Dataset):
    def __init__(self, root, transform=None):
        super().__init__()
        self.root = root
        self.transform = transform
        self.img_paths = list(map(str, Path(self.root).glob("**/*.jpg")))

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = _to_pil(load_image(img_path))

        if self.transform is not None:
            view1 = self.transform(image)
            view2 = self.transform(image)
        # print(view1.shape)
        # view = torch.stack([view1, view2])
        return view1, view2
        # return view


def get_image_transformer(img_size=IMG_SIZE, s=1):
    kernel_size = round(img_size / 10) // 2 * 2 + 1
    transform = T.Compose(
        [
            T.ToTensor(),
            T.RandomResizedCrop(size=img_size, scale=(0.08, 1), ratio=(3 / 4, 4 / 3), antialias=True),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply(
                [T.ColorJitter(brightness=0.8 * s, contrast=0.8 * s, saturation=0.8 * s, hue=0.2 * s)],
                p=0.8
            ),
            T.RandomGrayscale(p=0.2),
            T.RandomApply(
                [T.GaussianBlur(kernel_size=(kernel_size, kernel_size), sigma=(0.1, 2))],
                p=0.5
            ),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ]
    )
    return transform


if __name__ == "__main__":
    transform = get_image_transformer()
    ds = CustomDataset(root="/Users/jongbeomkim/Documents/datasets/VOCdevkit/VOC2012/JPEGImages", transform=transform)
    dl = DataLoader(dataset=ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    for batch, (view1, view2) in enumerate(dl, start=1):
        if batch >= 2:
            break
        grid = batched_image_to_grid(image=view1, n_cols=int(BATCH_SIZE ** 0.5), normalize=True)
        show_image(grid)
        grid = batched_image_to_grid(image=view2, n_cols=int(BATCH_SIZE ** 0.5), normalize=True)
        show_image(grid)
