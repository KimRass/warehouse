import numpy as np
import torch
import torchvision
import torchvision.transforms as T
from torchvision.models import resnet50, ResNet50_Weights
from torchviz import make_dot
import cv2
from PIL import Image


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


def get_permutated_patches(img, perm, resize_size=256, crop_size=225, patch_size=64):
    transform = T.Compose([T.ToTensor(), T.Resize(resize_size), T.RandomCrop(crop_size)])
    image = transform(img)

    cells = list()
    for col_cell in torch.split(image, crop_size // 3, dim=2):
        cells.extend(torch.split(col_cell, crop_size // 3, dim=1))
        
    patches = [T.RandomCrop(patch_size)(cell) for cell in cells]
    permuted_patches = [patches[idx - 1] for idx in perm]
    return permuted_patches

img = load_image("/Users/jongbeomkim/Downloads/horses.jpeg")
perm_set = [(9, 4, 6, 8, 3, 2, 5, 1, 7)]
perm = perm_set[0]
perm_patches = get_permutated_patches(img=img, perm=perm)
