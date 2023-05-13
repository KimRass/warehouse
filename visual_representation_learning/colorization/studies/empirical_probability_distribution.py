# References
    # https://github.com/Time0o/colorful-colorization/blob/9cbbc9fb7518bd92c441e36e45466cfd663fa9db/colorization/cielab.py

import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from tqdm.auto import tqdm
from skimage.color import rgb2lab
import cv2
import numpy as np


def load_image(img_path):
    img_path = str(img_path)
    img = cv2.imread(img_path, flags=cv2.IMREAD_COLOR)
    img = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2RGB)
    return img


def _figure_to_array(fig):
    fig.tight_layout()
    fig.canvas.draw()
    heatmap = np.array(fig.canvas.renderer._renderer)
    heatmap = cv2.cvtColor(src=heatmap, code=cv2.COLOR_BGRA2BGR)
    return heatmap


def _to_pil(img):
    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)
    return img


def show_image(img):
    copied_img = img.copy()
    copied_img = _to_pil(copied_img)
    copied_img.show()


def save_image(img, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    _to_pil(img).save(str(path))


def ab_color_space_histogram(data_dir):
    data_dir = Path(data_dir)

    hist = np.zeros(shape=(2 ** 8, 2 ** 8), dtype="uint32")
    for img_path in tqdm(list(data_dir.glob("**/*.jpg"))):
        img = load_image(img_path)
        lab_img = rgb2lab(img).round().astype("int8")

        ab_vals = lab_img[..., 1:].reshape(-1, 2)
        indices = ab_vals + 2 ** 7
        np.add.at(hist, (indices[:, 0], indices[:, 1]), 1)
    return hist


def empirical_probability_distribution(hist):
    copied = hist.copy()
    copied[copied == 0] = 1
    log_scaled = np.log10(copied)
    return log_scaled


def empirical_probability_distribution_plot(hist):
    copied = hist.copy()
    copied[copied == 0] = 1
    log_scaled = np.log10(copied)

    fig, axes = plt.subplots(figsize=(8, 8))
    axes.pcolormesh(np.arange(-128, 128), np.arange(-128, 128), log_scaled, cmap="jet")
    axes.set(xticks=range(-125, 125 + 1, 10), yticks=range(-125, 125 + 1, 10))
    axes.tick_params(axis="x", labelrotation=90, labelsize=8)
    axes.tick_params(axis="y", labelsize=8)
    axes.invert_yaxis()
    axes.grid(axis="x", color="White", alpha=1, linestyle="--", linewidth=0.5)
    axes.grid(axis="y", color="White", alpha=1, linestyle="--", linewidth=0.5)

    heatmap = _figure_to_array(fig)
    heatmap = cv2.pyrDown(heatmap)
    return heatmap


if __name__ == "__main__":
    hist = ab_color_space_histogram("/Users/jongbeomkim/Documents/datasets/VOCdevkit/VOC2012/JPEGImages")
    prob_dist_plot = empirical_probability_distribution_plot(hist)
    save_image(
        img=prob_dist_plot,
        path="/Users/jongbeomkim/Desktop/workspace/machine_learning/computer_vision/visual_representation_learning/colorization/studies/voc2012_empirical_probability_distribution.jpg"
    )
