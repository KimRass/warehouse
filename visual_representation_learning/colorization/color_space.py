# References
    # https://github.com/Time0o/colorful-colorization/blob/master/colorization/modules/soft_encode_ab.py

import numpy as np
from skimage.color import rgb2lab, lab2rgb
from pathlib import Path
import math
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import cv2
from PIL import Image

ab_idx2bin_idx = {
    (0, 16): 0, (0, 17): 1, (0, 18): 2, (0, 19): 3, (0, 20): 4,
    (1, 13): 5, (1, 14): 6, (1, 15): 7, (1, 16): 8, (1, 17): 9, (1, 18): 10, (1, 19): 11, (1, 20): 12,
    (2, 11): 13, (2, 12): 14, (2, 13): 15, (2, 14): 16, (2, 15): 17, (2, 16): 18, (2, 17): 19, (2, 18): 20, (2, 19): 21, (2, 20): 22,
    (3, 9): 23, (3, 10): 24, (3, 11): 25, (3, 12): 26, (3, 13): 27, (3, 14): 28, (3, 15): 29, (3, 16): 30, (3, 17): 31, (3, 18): 32, (3, 19): 33, (3, 20): 34,
    (4, 8): 35, (4, 9): 36, (4, 10): 37, (4, 11): 38, (4, 12): 39, (4, 13): 40, (4, 14): 41, (4, 15): 42, (4, 16): 43, (4, 17): 44, (4, 18): 45, (4, 19): 46, (4, 20): 47, (4, 21): 48,
    (5, 7): 49, (5, 8): 50, (5, 9): 51, (5, 10): 52, (5, 11): 53, (5, 12): 54, (5, 13): 55, (5, 14): 56, (5, 15): 57, (5, 16): 58, (5, 17): 59, (5, 18): 60, (5, 19): 61, (5, 20): 62, (5, 21): 63,
    (6, 6): 64, (6, 7): 65, (6, 8): 66, (6, 9): 67, (6, 10): 68, (6, 11): 69, (6, 12): 70, (6, 13): 71, (6, 14): 72, (6, 15): 73, (6, 16): 74, (6, 17): 75, (6, 18): 76, (6, 19): 77, (6, 20): 78, (6, 21): 79,
    (7, 6): 80, (7, 7): 81, (7, 8): 82, (7, 9): 83, (7, 10): 84, (7, 11): 85, (7, 12): 86, (7, 13): 87, (7, 14): 88, (7, 15): 89, (7, 16): 90, (7, 17): 91, (7, 18): 92, (7, 19): 93, (7, 20): 94, (7, 21): 95,
    (8, 5): 96, (8, 6): 97, (8, 7): 98, (8, 8): 99, (8, 9): 100, (8, 10): 101, (8, 11): 102, (8, 12): 103, (8, 13): 104, (8, 14): 105, (8, 15): 106, (8, 16): 107, (8, 17): 108, (8, 18): 109, (8, 19): 110, (8, 20): 111, (8, 21): 112,
    (9, 4): 113, (9, 5): 114, (9, 6): 115, (9, 7): 116, (9, 8): 117, (9, 9): 118, (9, 10): 119, (9, 11): 120, (9, 12): 121, (9, 13): 122, (9, 14): 123, (9, 15): 124, (9, 16): 125, (9, 17): 126, (9, 18): 127, (9, 19): 128, (9, 20): 129, (9, 21): 130,
    (10, 3): 131, (10, 4): 132, (10, 5): 133, (10, 6): 134, (10, 7): 135, (10, 8): 136, (10, 9): 137, (10, 10): 138, (10, 11): 139, (10, 12): 140, (10, 13): 141, (10, 14): 142, (10, 15): 143, (10, 16): 144, (10, 17): 145, (10, 18): 146, (10, 19): 147, (10, 20): 148,
    (11, 3): 149, (11, 4): 150, (11, 5): 151, (11, 6): 152, (11, 7): 153, (11, 8): 153, (11, 9): 155, (11, 10): 156, (11, 11): 157, (11, 12): 158, (11, 13): 159, (11, 14): 160, (11, 15): 161, (11, 16): 162, (11, 17): 163, (11, 18): 164, (11, 19): 165, (11, 20): 166,
    (12, 2): 167, (12, 3): 168, (12, 4): 169, (12, 5): 170, (12, 6): 171, (12, 7): 172, (12, 8): 173, (12, 9): 174, (12, 10): 175, (12, 11): 176, (12, 12): 177, (12, 13): 178, (12, 14): 179, (12, 15): 180, (12, 16): 181, (12, 17): 182, (12, 18): 183, (12, 19): 184, (12, 20): 185,
    (13, 1): 186, (13, 2): 187, (13, 3): 188, (13, 4): 189, (13, 5): 190, (13, 6): 191, (13, 7): 192, (13, 8): 193, (13, 9): 194, (13, 10): 195, (13, 11): 196, (13, 12): 197, (13, 13): 198, (13, 14): 199, (13, 15): 200, (13, 16): 201, (13, 17): 202, (13, 18): 203, (13, 19): 204, (13, 20): 205,
    (14, 1): 206, (14, 2): 207, (14, 3): 208, (14, 4): 209, (14, 5): 210, (14, 6): 211, (14, 7): 212, (14, 8): 213, (14, 9): 214, (14, 10): 215, (14, 11): 216, (14, 12): 217, (14, 13): 218, (14, 14): 219, (14, 15): 220, (14, 16): 221, (14, 17): 222, (14, 18): 223, (14, 19): 224,
    (15, 0): 225, (15, 1): 226, (15, 2): 227, (15, 3): 228, (15, 4): 229, (15, 5): 230, (15, 6): 231, (15, 7): 232, (15, 8): 233, (15, 9): 234, (15, 10): 235, (15, 11): 236, (15, 12): 237, (15, 13): 238, (15, 14): 239, (15, 15): 240, (15, 16): 241, (15, 17): 242, (15, 18): 243, (15, 19): 244,
    (16, 0): 245, (16, 1): 246, (16, 2): 247, (16, 3): 248, (16, 4): 249, (16, 5): 250, (16, 6): 251, (16, 7): 252, (16, 8): 253, (16, 9): 254, (16, 10): 255, (16, 11): 256, (16, 12): 257, (16, 13): 258, (16, 14): 259, (16, 15): 260, (16, 16): 261, (16, 17): 262, (16, 18): 263, (16, 19): 264,
    (17, 0): 265, (17, 1): 266, (17, 2): 267, (17, 3): 268, (17, 4): 269, (17, 5): 270, (17, 6): 271, (17, 7): 272, (17, 8): 273, (17, 9): 274, (17, 10): 275, (17, 11): 276, (17, 12): 277, (17, 13): 278, (17, 14): 279, (17, 15): 280, (17, 16): 281, (17, 17): 282, (17, 18): 283,
    (18, 0): 284, (18, 1): 285, (18, 2): 286, (18, 3): 287, (18, 4): 288, (18, 5): 289, (18, 6): 290, (18, 7): 291, (18, 8): 292, (18, 9): 293, (18, 10): 294, (18, 11): 295, (18, 12): 296, (18, 13): 297, (18, 14): 298, (18, 15): 299, (18, 16): 300, (18, 17): 301, (18, 18): 302,
    (19, 2): 303, (19, 3): 304, (19, 4): 305, (19, 5): 306, (19, 6): 307, (19, 7): 308, (19, 8): 309, (19, 9): 310, (19, 10): 311, (19, 11): 312
}
gamut_idx2bin_idx = {k[0] * n_cols + k[1]: v for k, v in ab_idx2bin_idx.items()}
bin_idx2ab_idx = {v: k for k, v in ab_idx2bin_idx.items()}
n_rows = 20
n_cols = 22
bin_idx_map = np.array([[ab_idx2bin_idx.get((i, j), -1) for j in range(n_cols)] for i in range(n_rows)])


def gamut_index_to_bin_index(gamut_idx):
    return gamut_idx2bin_idx.get(gamut_idx, -1)


def a_to_a_index(a):
    # a_idx = int((a + 95) / 10)
    a_idx = ((a + 95) / 10).astype("int16")
    return a_idx


def b_to_b_index(b):
    # b_idx = int((b + 115) / 10)
    b_idx = ((b + 115) / 10).astype("int16")
    return b_idx


def a_index_to_a(a_idx):
    a = -90 + 10* a_idx
    # if a_idx != -1 else -1
    return a

def b_index_to_b(b_idx):
    b = -110 + 10* b_idx
    # if b_idx != -1 else -1
    return b


def lab_to_bin_idx(lab, bin_idx_map=bin_idx_map):
    _, a, b = lab
    a_idx = int((a + 95) / 10)
    b_idx = int((b + 115) / 10)
    bin_idx = bin_idx_map[a_idx, b_idx]
    return bin_idx


def bin_idx_to_ab(bin_idx, bin_idx2ab_idx=bin_idx2ab_idx):
    a_idx, b_idx = bin_idx2ab_idx[bin_idx]
    a = - 90 + 10 * a_idx
    b = - 110 + 10 * b_idx
    return (a, b)


def get_2d_gaussian_function_output(x, y, mu_x, mu_y, sigma=5):
    fn_vals = (1 / (2 * math.pi * sigma ** 2)) * np.exp(-((x - mu_x) ** 2 + (y - mu_y) ** 2) / (2 * sigma ** 2))
    # probs = fn_vals / fn_vals.sum()
    # return probs
    return fn_vals


def load_image(img_path):
    img_path = str(img_path)
    img = cv2.imread(img_path, flags=cv2.IMREAD_COLOR)
    img = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2RGB)
    return img


def _convert_to_pil(img):
    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)
    return img


def show_image(img):
    copied_img = img.copy()
    copied_img = _convert_to_pil(copied_img)
    copied_img.show()


def lab_to_soft_encoded_vector(lab):
    # bin_idx = lab_to_bin_idx(lab=lab, bin_idx_map=bin_idx_map)
    # bin_idx
    _, a, b = lab
    a, b = -94, 101
    a_idx = a_to_a_index(a)
    b_idx = b_to_b_index(b)
    ab_indices = (a_idx, b_idx)

    t_neighbor = (max(0, a_idx - 1), b_idx)
    b_neighbor = (min(n_rows - 1, a_idx + 1), b_idx)
    l_neighbor = (a_idx, max(0, b_idx - 1))
    r_neighbor = (a_idx, min(n_cols - 1, b_idx + 1))
    neighbors = (ab_indices, t_neighbor, b_neighbor, l_neighbor, r_neighbor)
    # neighbors
    # ab_indices
    neighbors_ab = np.array([(a_index_to_a(a_idx), b_index_to_b(b_idx)) for a_idx, b_idx in neighbors])
    # neighbors_ab

    fn_vals = get_2d_gaussian_function_output(x=neighbors_ab[:, 0], y=neighbors_ab[:, 1], a=a, b=b)
    bin_indices = np.array(
        [ab_idx2bin_idx.get((a_idx, b_idx), -1) for a_idx, b_idx in  (ab_indices, t_neighbor, b_neighbor, l_neighbor, r_neighbor)]
    )
    bin_indices
    fn_vals
    fn_vals *= (bin_indices != -1)
    probs = fn_vals / fn_vals.sum()
    probs

    vec = np.zeros((313,))
    for bin_idx, prob in zip(bin_indices, probs):
        if bin_idx == -1:
            continue
        vec[bin_idx] += prob
    return vec


def _get_width_and_height(img):
    if img.ndim == 2:
        height, width = img.shape
    else:
        height, width, _ = img.shape
    return width, height


def lab_image_to_soft_encoded_labels(lab_img):
    a_indices = a_to_a_index(lab_img[..., 1])
    b_indices = b_to_b_index(lab_img[..., 2])

    a_indices_neighbors = np.stack(
        [a_indices, np.maximum(0, a_indices - 1), np.minimum(n_rows - 1, a_indices + 1), a_indices, a_indices]
    )
    b_indices_neighbors = np.stack(
        [b_indices, b_indices, b_indices, np.maximum(0, b_indices - 1), np.minimum(n_cols - 1, b_indices + 1)]
    )

    fn_vals = get_2d_gaussian_function_output(x=a_indices_neighbors, y=b_indices_neighbors, mu_x=a_indices, mu_y=b_indices)
    gamut_indices = a_indices_neighbors * n_cols + b_indices_neighbors
    bin_indices = np.vectorize(gamut_index_to_bin_index)(gamut_indices)
    fn_vals *= (bin_indices != -1)
    probs = fn_vals / fn_vals.sum(axis=0)

    w, h = _get_width_and_height(lab_img)
    labels = np.zeros((313, 10, 10))
    bin_indices[0, : 10, : 10].shape
    labels.shape
    labels[bin_indices[0, : 10, : 10]].shape
    labels == bin_indices
    
    
    
    np.put(labels, bin_indices[0, ...].reshape((h * w,)), probs[0, ...].reshape((h * w,)))
    labels.sum()
    
    # np.put(labels, bin_indices, probs)
    np.put(labels, bin_indices.reshape((5 * w * h,)), probs.reshape((5 * w * h,)))
    # labels.sum(axis=0).sum
    labels.sum()
    # bin_indices.shape
    # probs.shape
    np.where()
    return labels

img = load_image("D:/imagenet-mini/train/n02088238/n02088238_224.JPEG")
lab_img = rgb2lab(img)
# transform = T.Compose(
#     [
#         T.Resize((224, 224))
#     ]
# )
# image = torch.Tensor(img)
# image = transform(image)

lab = lab_img[100, 115]
lab_to_soft_encoded_vector(lab).sum()

np_vec = np.vectorize(lab_to_soft_encoded_vector)
np_vec(lab_img)