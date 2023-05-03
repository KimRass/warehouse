import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torchvision.models import alexnet, AlexNet_Weights
from torchviz import make_dot
from kornia import image_to_tensor, tensor_to_image
from kornia.augmentation import RandomCrop, Resize, RandomGrayscale, IntensityAugmentationBase2D
import cv2
from PIL import Image
import random
from typing import Optional, Dict, Any

from computer_vision.self_supervised_learning.jigsaw_puzzle.context_free_network import (
    ContextFreeNetwork
)
from computer_vision.self_supervised_learning.jigsaw_puzzle.permutation_set import (
    get_permutation_set,
    get_permutated_tiles
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


class SpatiallyJitterColorChannels(nn.Module):
    def __init__(self, shift=1):
        super().__init__()

        self.shift = shift

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        for batch in range(b):
            for ch in range(c):
                x[batch: batch + 1, ch: ch + 1, ...] = torch.roll(
                    x[batch: batch + 1, ch: ch + 1, ...],
                    shifts=(random.randint(-self.shift, self.shift), random.randint(-self.shift, self.shift)),
                    dims=(2, 3)
                )
        return x


# def spatially_jitter_color_channels(tensor, shift):
#     # tensor = images
#     # shift=10

#     output = torch.zeros_like(tensor)
#     for ch in range(3):
#         output[:, ch, :, :] = torch.roll(
#             tensor[:, ch, :, :],
#             shifts=(random.randint(-shift, shift), random.randint(-shift, shift)),
#             dims=(1, 2)
#         )
#     return output


# class SpatiallyJitterColorChannels(IntensityAugmentationBase2D):
#     def __init__(self, shift=0, p: float=0, same_on_batch: bool=False) -> None:
#         super().__init__(p=p, same_on_batch=same_on_batch)

#         self.shift = shift

#     def apply_transform(
#             self,
#             input: torch.Tensor,
#             params: Dict[str, torch.Tensor],
#             flags: Dict[str, Any],
#             transform: Optional[torch.Tensor]=None
#         ) -> torch.Tensor:
#         output = spatially_jitter_color_channels(tensor=input, shift=self.shift)
#         return output


if __name__ == "__main__":
    # model = alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)
    # model.features

    fcn = ContextFreeNetwork()

    img = load_image("/Users/jongbeomkim/Downloads/horses.jpeg")
    resize_size=256
    crop_size=225
    image = T.ToTensor()(img).unsqueeze(0)

    images = image.repeat(8, 1, 1, 1)
    transform = T.Compose(
        [
            Resize(size=resize_size),
            RandomCrop(size=(crop_size, crop_size)),
            SpatiallyJitterColorChannels(shift=2),
            RandomGrayscale(p=0.3)
        ]
    )
    images = transform(images)
    for i in range(8):
        show_image(convert_tensor_to_array(images[i]))

    perm_set = get_permutation_set(n_perms=10, n_tiles=9)
    perm_tiles = get_permutated_tiles(images=images, perm_set=perm_set)
    for i in range(36):
        show_image(convert_tensor_to_array(perm_tiles[i]))

    output = fcn(perm_tiles)
    output.shape


