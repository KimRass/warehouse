import numpy as np
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models import alexnet, AlexNet_Weights
from typing import Literal

from utils import (
    load_image,
    show_image,
    save_image,
    denormalize_array,
    resize_image,
    _get_width_and_height,
    _blend_two_images,
    _apply_jet_colormap,
    _convert_rgba_to_rgb,
)


def tensor_to_array(tensor):
    copied_tensor = tensor.clone().squeeze()
    if copied_tensor.ndim == 3:
        copied_tensor = copied_tensor.permute((1, 2, 0)).detach().cpu().numpy()
    elif copied_tensor.ndim == 2:
        copied_tensor = copied_tensor.detach().cpu().numpy()
    copied_tensor = denormalize_array(copied_tensor)
    return copied_tensor


def print_all_layers(model):
    print(f"""|         NAME         |                            MODULE                            |""")
    print(f"""|----------------------|--------------------------------------------------------------|""")
    for name, module in model.named_modules():
        if not isinstance(module, nn.Sequential) and name:
            print(f"""| {name:20s} | {str(type(module)):60s} |""")


def _get_target_layer(layer):
    return eval(
        "model" + "".join(
            [f"""[{i}]""" if i.isdigit() else f""".{i}""" for i in layer.split(".")]
        )
    )


class FeatureMapExtractor():
    def __init__(self, model):
        self.model = model

        self.feat_map = None

    def get_feature_map(self, image, layer):
        def forward_hook_fn(module, input, output):
            self.feat_map = output

        trg_layer = _get_target_layer(layer)
        trg_layer.register_forward_hook(forward_hook_fn)

        self.model(image)
        return self.feat_map


def feature_map_to_attention_map(feat_map, img, mode=Literal["bw", "jet"], p=1):
    # "We first compute the feature maps of this layer, then we raise each feature activation
    # on the power $p$, and finally we sum the activations at each location of the feature map."
    feat_map = feat_map ** p
    feat_map = feat_map.sum(axis=1)

    feat_map = tensor_to_array(feat_map)
    w, h = _get_width_and_height(img)
    feat_map = resize_image(img=feat_map, w=w, h=h)
    if mode == "bw":
        output = np.concatenate([img, feat_map[..., None]], axis=2)
    elif mode == "jet":
        feat_map = _apply_jet_colormap(feat_map)
        output = _blend_two_images(img1=img, img2=feat_map, alpha=0.6)
    output = _convert_rgba_to_rgb(output)
    return output


if __name__ == "__main__":
    model = alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)
    print_all_layers(model)

    img = load_image(
        "https://hips.hearstapps.com/ghk.h-cdn.co/assets/16/08/gettyimages-147786673.jpg?crop=0.4444444444444445xw:1xh;center,top&resize=980:*"
    )
    transform = T.Compose(
        [
            T.ToTensor(),
            T.Resize((227, 227)),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ]
    )
    image = transform(img).unsqueeze(0)

    feat_map_extr = FeatureMapExtractor(model)
    feat_map1 = feat_map_extr.get_feature_map(image=image, layer="features.2") # "Conv1 $27 \times 27$"
    feat_map2 = feat_map_extr.get_feature_map(image=image, layer="features.5") # "Conv3 $13 \times 13$"
    feat_map3 = feat_map_extr.get_feature_map(image=image, layer="features.12") # "Conv5 $6 \times 6$$"

    # "For the conv. layers 1, 2, and 3 we used the powers $p = 1$, $p = 2$, and $p = 4$ respectively."
    attn_map1 = feature_map_to_attention_map(feat_map=feat_map1, img=img, mode="bw", p=1)
    attn_map2 = feature_map_to_attention_map(feat_map=feat_map2, img=img, mode="bw", p=2)
    attn_map3 = feature_map_to_attention_map(feat_map=feat_map3, img=img, mode="bw", p=4)
    
    save_image(img=attn_map1, path="attention_map_examples/golden_retriever_conv1_27.jpg")
    save_image(img=attn_map2, path="attention_map_examples/golden_retriever_conv3_13.jpg")
    save_image(img=attn_map3, path="attention_map_examples/golden_retriever_conv5_6.jpg")
