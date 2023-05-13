# References
    # https://zahid-parvez.medium.com/image-histograms-in-opencv-python-9fe3a7e0ae4f

import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from moviepy.video.io.bindings import mplfig_to_npimage

from process_images import (
    load_image
)


def get_channel_wise_pixel_intensity_histogram(img):
    fig, axes = plt.subplots(figsize=(8, 6))
    colors = ["r","g","b"]
    for c in [0, 1, 2]:
        hist = cv2.calcHist(images=[img], channels=[c], mask=None, histSize=[256], ranges=[0, 256])
        axes.plot(hist, color=colors[c], linewidth=0.5)

    axes.set_xticks(np.arange(0, 255, 25))
    axes.tick_params(axis="x", labelrotation=90)
    axes.tick_params(axis="both", which="major", labelsize=7)
    fig.tight_layout()

    arr = mplfig_to_npimage(fig)
    return arr


def get_pixel_intensity_histogram(img):
    fig, axes = plt.subplots(figsize=(8, 6))
    pseudo_img = np.concatenate([img[..., 0], img[..., 1], img[..., 2]], axis=0)
    hist = cv2.calcHist(images=[pseudo_img], channels=[0], mask=None, histSize=[256], ranges=[0, 256])
    axes.plot(hist, linewidth=0.5, color="black")

    axes.set_xticks(np.arange(0, 255, 25))
    axes.tick_params(axis="x", labelrotation=90)
    axes.tick_params(axis="both", which="major", labelsize=7)
    fig.tight_layout()    

    arr = mplfig_to_npimage(fig)
    return arr


img_path = "/Users/jongbeomkim/Documents/datasets/imagenet-mini/train/n01440764/n01440764_10845.JPEG"
img = load_image(img_path)

