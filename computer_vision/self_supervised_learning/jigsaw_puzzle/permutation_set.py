import torch
import torchvision.transforms as T
import random
from itertools import permutations
import math
import numpy as np


def _reorder_cells(cells, b):
    """
    Re-order patches e.g., if batch size is 4,
    1111 2222 3333 4444 5555 6666 7777 8888 9999 -> 123456789 123456789 123456789 123456789
    Args:
        cells (torch.Tensor): Patches to re-order with the shape `(b * 9, c, h // 3, w // 3)`
        b (int): Batch size

    Returns:
        torch.Tensor: Re-ordered patches
    """
    return torch.cat([cells[i:: b] for i in range(4)], axis=0)


def get_permutated_patches(images, perm_set, crop_size=225, patch_size=64):
    # `(b, c, h, w)`
    b, _, _, _ = images.shape

    # `(b * 3, c, h // 3, w)`
    col_cells = torch.cat(torch.split(images, crop_size // 3, dim=2), axis=0)
    # `(b * 9, c, h // 3, w // 3)`
    cells = torch.cat(torch.split(col_cells, crop_size // 3, dim=3), axis=0)
    cells = _reorder_cells(cells=cells, b=b)

    patches = T.RandomCrop(patch_size)(cells)

    perms = random.choices(perm_set, k=b)
    batched_perm = torch.cat(
        [(torch.tensor(perm) - 1) + idx * 9 for idx, perm in enumerate(perms)]
    )
    perm_patches = patches[batched_perm]
    return perm_patches


def get_hamming_distance(a, b, axis=0):
    return np.count_nonzero(a != b, axis=axis)


def get_permutation_set(n_perms, n_patches=9):
    # `(9, math.factorial(n_patches))`
    P_bar = np.array(list(permutations(range(1, n_patches + 1), n_patches))).T # $\bar{P}$ # (1)
    # `(n_patches, 0)`
    P = np.empty((n_patches, 0), dtype="int64") # (2)
    j = random.randint(0, math.factorial(n_patches) - 1) # (3)
    i = 1 # (4)
    while True: # (5)
        P_bar_j = P_bar[:, j: j + 1] # $\bar{P}_{j}$
        P = np.concatenate([P, P_bar_j], axis=1) # (6)
        P_bar = np.concatenate([P_bar[:, : j], P_bar[:, j + 1:]], axis=1) # (7)
        # `(i, math.factorial(n_patches) - i)`
        D = get_hamming_distance(a=P[:, :, None], b=P_bar[:, None, :], axis=0) # (8)
        # `(1, math.factorial(n_patches) - i)`
        D_bar = D.mean(axis=0)[None, :] # (9)
        j = np.random.choice(
            np.nonzero(D_bar == D_bar.max())[1]
        ) # (10)
        i += 1 # (11)

        if i > n_perms:
            break # (12)
    perm_set = list(map(tuple, P.T))
    return perm_set
