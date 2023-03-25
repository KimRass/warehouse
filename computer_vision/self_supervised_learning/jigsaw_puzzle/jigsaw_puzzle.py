import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torchvision.models import alexnet, AlexNet_Weights
from torchviz import make_dot
import cv2
from PIL import Image
import random


def load_image(img_path):
    img_path = str(img_path)
    img = cv2.imread(img_path, flags=cv2.IMREAD_COLOR)
    img = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2RGB)
    return img


def _denormalize_array(img, mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)):
    copied_img = img.copy()
    # copied_img *= variance
    # copied_img += mean
    copied_img *= 255.0
    copied_img = np.clip(a=copied_img, a_min=0, a_max=255).astype("uint8")
    return copied_img


def convert_tensor_to_array(tensor):
    copied_tensor = tensor.clone().squeeze().permute((1, 2, 0)).detach().cpu().numpy()
    copied_tensor = _denormalize_array(copied_tensor)
    return copied_tensor


def _convert_to_pil(img):
    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)
    return img


def show_image(img):
    copied_img = img.copy()
    copied_img = _convert_to_pil(copied_img)
    copied_img.show()


def _reorder_cells(cells):
    return torch.cat([cells[i:: 4] for i in range(4)], axis=0)


def _reorder_cells2(cells):
    return torch.cat([cells[i:: 12] for i in range(0, 12, 4)], axis=0)


# 1 2 3 4 -> 1234 1234 1234
# 1234 1234 1234 -> 111 222 333 444
# 111 222 333 444 -> 111222333444 111222333444 111222333444
# 111222333444 111222333444 111222333444 -> 111111111 222222222 333333333 444444444
def get_permutated_patches(images, perm_set, crop_size=225, patch_size=64):
    b, _, _, _ = images.shape

    col_cells = torch.cat(torch.split(images, crop_size // 3, dim=2), axis=0)
    cells = torch.cat(torch.split(col_cells, crop_size // 3, dim=3), axis=0)
    cells = _reorder_cells(cells)
    # cells = torch.cat(cells)
        
    # patches = [T.RandomCrop(patch_size)(cell) for cell in cells]
    patches = T.RandomCrop(patch_size)(cells)
    patches.shape

    perms = random.choices(perm_set, k=b)
    batched_perm = torch.cat(
        [(torch.tensor(perm) - 1) + idx * 9 for idx, perm in enumerate(perms)]
    )
    perm_patches = patches[batched_perm]
    return perm_patches


class ContextFreeNetwork(nn.Module):
    def __init__(self, n_permutations=1000):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 96, kernel_size=11, stride=2, padding=2)
        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)

        self.flatten1 = nn.Flatten(start_dim=1, end_dim=3)
        # 차원이 안 맞는데?? 논문에 따르면 (256 * 4 * 4, 512)여야 함.
        self.fc6 = nn.Linear(256 * 2 * 2, 512)

        self.flatten2 = nn.Flatten(start_dim=0, end_dim=1)
        self.fc7 = nn.Linear(4608, 4096)
        self.fc8 = nn.Linear(4096, n_permutations)

    def forward(self, patches):
        x = self.conv1(patches)
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

        x = self.flatten1(x)
        x = self.fc6(x)

        x = torch.cat(
            [self.flatten2(i).unsqueeze(0) for i in torch.split(x, 9, dim=0)]
        )

        x = self.fc7(x)
        x = self.fc8(x)

        x = F.softmax(x, dim=1)
        return x


if __name__ == "__main__":
    # model = alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)
    # model.features

    fcn = ContextFreeNetwork()

    img = load_image("/Users/jongbeomkim/Downloads/horses.jpeg")
    resize_size=256
    crop_size=225
    transform = T.Compose([T.ToTensor(), T.Resize(resize_size), T.RandomCrop(crop_size)])
    image = transform(img).unsqueeze(0)
    images = image.repeat(4, 1, 1, 1)

    perm_set = [(9, 4, 6, 8, 3, 2, 5, 1, 7)]
    # perm_set = [(1, 2, 3, 4, 5, 6, 7, 8, 9)]
    perm_patches = get_permutated_patches(images=images, perm_set=perm_set)
    perm_patches.shape
    for i in range(36):
        show_image(convert_tensor_to_array(perm_patches[i]))

    output = fcn(perm_patches)
    output.shape


