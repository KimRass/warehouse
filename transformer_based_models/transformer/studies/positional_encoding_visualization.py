import cv2
import cmapy
import matplotlib.pyplot as plt

from transformer.model import PositionalEncoding
from image_utils import _figure_to_array, save_image


def visualize_positional_encoding_matrix(pos_enc, cmap="bwr"):
    vis = pos_enc.numpy()
    vis -= vis.min()
    vis /= vis.max()
    vis *= 255
    vis = vis.astype("uint8")
    vis = cv2.applyColorMap(vis, cmapy.cmap(cmap))

    fig, axes = plt.subplots(figsize=(10, 6))
    axes.imshow(vis)
    arr = _figure_to_array(fig)
    return arr


if __name__ == "__main__":
    pos_enc = PositionalEncoding(dim=128, max_len=50)
    vis = visualize_positional_encoding_matrix(pos_enc.pe)
    save_image(img=vis, path="positional_encoding_visualization.jpg")
