import numpy as np
import cv2


def get_colorfulness(img):
    r, g, b = cv2.split(img.astype("float"))
    rg = np.absolute(r - g)
    yb = np.absolute((r + g) / 2 - b)
    rg_mean, rg_std = np.mean(rg), np.std(rg)
    yb_mean, yb_std = np.mean(yb), np.std(yb)
    std_root = np.sqrt((rg_std ** 2) + (yb_std ** 2))
    mean_root = np.sqrt((rg_mean ** 2) + (yb_mean ** 2))
    colorfulness = std_root + (0.3 * mean_root)
    return colorfulness
