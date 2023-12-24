import cv2
from PIL import Image
from pathlib import Path
import numpy as np
from moviepy.video.io.bindings import mplfig_to_npimage


def load_image(img_path):
    img_path = str(img_path)
    img = cv2.imread(img_path, flags=cv2.IMREAD_COLOR)
    img = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2RGB)
    return img


def _to_pil(img):
    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)
    return img


def _to_array(img):
    img = np.array(img)
    return img


def show_image(img):
    copied_img = img.copy()
    copied_img = _to_pil(copied_img)
    copied_img.show()


def _get_width_and_height(img):
    if img.ndim == 2:
        height, width = img.shape
    else:
        height, width, _ = img.shape
    return width, height


def save_image(img, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    _to_pil(img).save(str(path))


def _figure_to_array(fig):
    arr = mplfig_to_npimage(fig)
    return arr


def denormalize_array(img):
    copied_img = img.copy()
    copied_img -= copied_img.min()
    copied_img /= copied_img.max()
    copied_img *= 255
    copied_img = np.clip(a=copied_img, a_min=0, a_max=255).astype("uint8")
    return copied_img


def resize_image(img, w, h):
    resized_img = cv2.resize(src=img, dsize=(w, h))
    return resized_img


def _blend_two_images(img1, img2, alpha=0.5):
    img1 = _to_pil(img1)
    img2 = _to_pil(img2)
    img_blended = Image.blend(im1=img1, im2=img2, alpha=alpha)
    return _to_array(img_blended)


def _apply_jet_colormap(img):
    img_jet = cv2.applyColorMap(src=(255 - img), colormap=cv2.COLORMAP_JET)
    return img_jet


def _convert_rgba_to_rgb(img):
    copied_img = img.copy().astype("float")
    copied_img[..., 0] *= copied_img[..., 3] / 255
    copied_img[..., 1] *= copied_img[..., 3] / 255
    copied_img[..., 2] *= copied_img[..., 3] / 255
    copied_img = copied_img.astype("uint8")
    copied_img = copied_img[..., : 3]
    return copied_img
