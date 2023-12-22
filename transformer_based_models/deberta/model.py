import numpy as np
import matplotlib.pyplot as plt
from moviepy.video.io.bindings import mplfig_to_npimage
from PIL import Image


MAX_LEN = 512
def _get_relative_distance(i, j, max_rel_dist = MAX_LEN // 4):
    if i - j <= -max_rel_dist:
        return 0
    elif i - j >= max_rel_dist:
        return 2 * max_rel_dist - 1
    else:
        return i - j + max_rel_dist


def _figure_to_array(fig):
    arr = mplfig_to_npimage(fig)
    return arr


def _to_pil(img):
    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)
    return img


def save_image(img, path):
    _to_pil(img).save(str(path))



if __name__ == "__main__":
    rel_dist_mat = np.vectorize(_get_relative_distance)(np.arange(MAX_LEN)[:, None], np.arange(MAX_LEN)[None, :])
    fig, axes = plt.subplots(1, 1, figsize=(7, 7))
    axes.pcolormesh(rel_dist_mat)
    img = _figure_to_array(fig)
    save_image(img=img, path="/Users/jongbeomkim/Desktop/workspace/transformer_based_models/deberta/relative_distance.jpg")
