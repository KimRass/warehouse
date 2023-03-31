import numpy as np
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models import alexnet, AlexNet_Weights
import cv2
from typing import Literal
from PIL import Image
import requests


def load_image(url_or_path=""):
    url_or_path = str(url_or_path)

    if "http" in url_or_path:
        img_arr = np.asarray(
            bytearray(requests.get(url_or_path).content), dtype="uint8"
        )
        img = cv2.imdecode(img_arr, flags=cv2.IMREAD_COLOR)
        img = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2RGB)
    else:
        img = cv2.imread(url_or_path, flags=cv2.IMREAD_COLOR)
        img = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2RGB)
    return img


def show_image(img):
    _convert_to_pil(img).show()


def save_image(img, path):
    _convert_to_pil(img).save(str(path))


def denormalize_array(img):
    copied_img = img.copy()
    copied_img -= copied_img.min()
    copied_img /= copied_img.max()
    copied_img *= 255
    copied_img = np.clip(a=copied_img, a_min=0, a_max=255).astype("uint8")
    return copied_img


def convert_tensor_to_array(tensor):
    copied_tensor = tensor.clone().squeeze()
    if copied_tensor.ndim == 3:
        copied_tensor = copied_tensor.permute((1, 2, 0)).detach().cpu().numpy()
    elif copied_tensor.ndim == 2:
        copied_tensor = copied_tensor.detach().cpu().numpy()
    copied_tensor = denormalize_array(copied_tensor)
    return copied_tensor


def resize_image(img, w, h):
    resized_img = cv2.resize(src=img, dsize=(w, h))
    return resized_img


def get_width_and_height(img):
    if img.ndim == 2:
        h, w = img.shape
    else:
        h, w, _ = img.shape
    return w, h


def _convert_to_pil(img):
    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)
    return img


def _convert_to_array(img):
    img = np.array(img)
    return img


def _blend_two_images(img1, img2, alpha=0.5):
    img1 = _convert_to_pil(img1)
    img2 = _convert_to_pil(img2)
    img_blended = Image.blend(im1=img1, im2=img2, alpha=alpha)
    return _convert_to_array(img_blended)


def _apply_jet_colormap(img):
    img_jet = cv2.applyColorMap(src=(255 - img), colormap=cv2.COLORMAP_JET)
    return img_jet


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


def _convert_rgba_to_rgb(img):
    copied_img = img.copy().astype("float")
    copied_img[..., 0] *= copied_img[..., 3] / 255
    copied_img[..., 1] *= copied_img[..., 3] / 255
    copied_img[..., 2] *= copied_img[..., 3] / 255
    copied_img = copied_img.astype("uint8")
    copied_img = copied_img[..., : 3]
    return copied_img


def convert_feature_map_to_attention_map(feat_map, img, mode=Literal["bw", "jet"], p=1):
    feat_map = feat_map.sum(axis=1)
    feat_map = feat_map ** p

    feat_map = convert_tensor_to_array(feat_map)
    w, h = get_width_and_height(img)
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

    img = load_image("https://hips.hearstapps.com/ghk.h-cdn.co/assets/16/08/gettyimages-147786673.jpg?crop=0.4444444444444445xw:1xh;center,top&resize=980:*")
    transform = T.Compose(
        [
            T.ToTensor(),
            T.Resize((227, 227)),
            T.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            )
        ]
    )
    image = transform(img).unsqueeze(0)

    extractor = FeatureMapExtractor(model)
    feat_map1 = extractor.get_feature_map(image=image, layer="features.2")
    feat_map2 = extractor.get_feature_map(image=image, layer="features.5")
    feat_map3 = extractor.get_feature_map(image=image, layer="features.12")

    # Below are, from the paper, 'Conv1 27 × 27', 'Conv3 13 × 13' and 'Conv5 6 × 6' respectively.
    attn_map1 = convert_feature_map_to_attention_map(feat_map=feat_map1, img=img, mode="bw", p=1)
    attn_map2 = convert_feature_map_to_attention_map(feat_map=feat_map2, img=img, mode="bw", p=2)
    attn_map3 = convert_feature_map_to_attention_map(feat_map=feat_map3, img=img, mode="bw", p=4)
    
    save_image(img=attn_map1, path="/Users/jongbeomkim/Desktop/workspace/machine_learning/computer_vision/self_supervised_learning/image_rotation/attention_map_samples/golden_retriever_conv1_27.jpg")
    save_image(img=attn_map2, path="/Users/jongbeomkim/Desktop/workspace/machine_learning/computer_vision/self_supervised_learning/image_rotation/attention_map_samples/golden_retriever_conv3_13.jpg")
    save_image(img=attn_map3, path="/Users/jongbeomkim/Desktop/workspace/machine_learning/computer_vision/self_supervised_learning/image_rotation/attention_map_samples/golden_retriever_conv5_6.jpg")
