# References:
    # https://towardsdatascience.com/getting-started-with-pytorch-image-models-timm-a-practitioners-guide-4e77b4bf9055

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.optim import AdamP
from timm.scheduler import CosineLRScheduler
import matplotlib.pyplot as plt

from image_utils import _figure_to_array, show_image


def get_lr_schedule(scheduler, n_steps):
    lrs = list()
    for step in range(1, n_steps + 1):
        lrs.append(optimizer.param_groups[0]["lr"])
        # Should be called after each optimizer update with the index of the next update.
        scheduler.step_update(num_updates=step + 1)
    return lrs


def visualize_lrs(lrs, n_steps):
    fig, axes = plt.subplots(figsize=(int(len(lrs) ** 0.2), 3))
    axes.plot(range(1, n_steps + 1), lrs)
    axes.set_ylim([0, max(lrs) * 1.1])
    axes.set_xlim([0, n_steps])
    axes.tick_params(axis="x", labelrotation=90, labelsize=5)
    axes.tick_params(axis="y", labelsize=5)
    axes.grid(axis="x", color="black", alpha=0.5, linestyle="--", linewidth=0.5)
    fig.tight_layout()

    arr = _figure_to_array(fig)
    return arr


if __name__ == "__main__":
    model = torch.nn.Linear(2, 1)

    # We used the Adam optimization scheme with a max learning rate of 2.5e-4.
    # The learning rate was increased linearly from zero over the first 2000 updates
    # and annealed to 0 using a cosine schedule.
    n_steps = 22_000
    init_lr = 0
    max_lr = 2.5e-04
    min_lr = 0
    optimizer = AdamP(model.parameters(), lr=max_lr)
    scheduler = CosineLRScheduler(
        optimizer=optimizer,
        t_initial=n_steps,
        lr_min=min_lr,
        warmup_t=2000,
        warmup_lr_init=init_lr,
        warmup_prefix=True, # Warmup 부분과 Decay 부분이 자연스럽게 이어집니다.
        t_in_epochs=False # If `True` the number of iterations is given in terms of epochs
            # rather than the number of batch updates.
    )
    lrs = get_lr_schedule(scheduler, n_steps=n_steps)
    vis = visualize_lrs(lrs=lrs, n_steps=n_steps)
    show_image(vis)
