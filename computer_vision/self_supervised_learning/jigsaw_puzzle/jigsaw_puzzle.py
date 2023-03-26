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

from computer_vision.self_supervised_learning.jigsaw_puzzle.context_free_network import (
    ContextFreeNetwork
)
from computer_vision.self_supervised_learning.jigsaw_puzzle.permutation_set import (
    get_permutation_set,
    get_permutated_patches
)


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

    perm_set = get_permutation_set(n_perms=10, n_patches=9)
    perm_patches = get_permutated_patches(images=images, perm_set=perm_set)
    for i in range(36):
        show_image(convert_tensor_to_array(perm_patches[i]))

    output = fcn(perm_patches)
    output.shape


