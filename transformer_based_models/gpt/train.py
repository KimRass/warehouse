# References:
    # https://towardsdatascience.com/getting-started-with-pytorch-image-models-timm-a-practitioners-guide-4e77b4bf9055

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.optim import AdamP
from timm.scheduler import CosineLRScheduler
import matplotlib.pyplot as plt

from image_utils import _figure_to_array, show_image


def get_lr_schedule(optim, scheduler, n_epochs, data_size, batch_size):
    n_steps_per_epoch = data_size // batch_size
    lrs = list()
    for epoch in range(1, n_epochs + 1):
        for step in range(1, n_steps + 1):
            scheduler.step_update(num_updates=(epoch - 1) * n_steps_per_epoch + step + 1)
        scheduler.step(epoch=epoch + 1)

        lrs.append(optim.param_groups[0]["lr"])
        print(optim.param_groups[0]["lr"])
    return lrs


def vis_lrs(lrs, n_steps):
    fig, axes = plt.subplots(figsize=(8, 3))
    axes.plot(range(1, n_steps + 1), lrs)
    axes.set_ylim([0, max(lrs) * 1.1])
    axes.set_xlim([0, n_steps])
    axes.tick_params(axis="x", labelrotation=90, labelsize=5)
    axes.tick_params(axis="y", labelsize=5)
    axes.grid(axis="x", color="black", alpha=0.5, linestyle="--", linewidth=0.5)
    fig.tight_layout()

    arr = _figure_to_array(fig)
    return arr


def get_cosine_scheduler(optim, warmup_epochs, n_epochs, init_lr, min_lr):
    return CosineLRScheduler(
        optimizer=optim,
        t_initial=n_epochs,
        lr_min=min_lr,
        warmup_t=warmup_epochs,
        warmup_lr_init=init_lr,
        warmup_prefix=True,
        t_in_epochs=True,
    )


if __name__ == "__main__":
    model = torch.nn.Linear(2, 1)

    # We used the Adam optimization scheme with a max learning rate of 2.5e-4.
    # The learning rate was increased linearly from zero over the first 2000 updates
    # and annealed to 0 using a cosine schedule.
    n_steps = 22_000
    init_lr = 0
    max_lr = 2.5e-04
    min_lr = 0
    # optim = AdamP(model.parameters(), lr=max_lr)
    # scheduler = CosineLRScheduler(
    #     optimizer=optim,
    #     t_initial=n_steps,
    #     lr_min=min_lr,
    #     warmup_t=2000,
    #     warmup_lr_init=init_lr,
    #     warmup_prefix=True, # Warmup 부분과 Decay 부분이 자연스럽게 이어집니다.
    #     t_in_epochs=False # If `True` the number of iterations is given in terms of epochs
    #         # rather than the number of batch updates.
    # )
    # lrs = get_lr_schedule(scheduler, n_steps=n_steps)
    # vis = vis_lrs(lrs=lrs, n_steps=n_steps)
    optim = AdamP(model.parameters(), lr=0.0005)
    scheduler = get_cosine_scheduler(
        optim=optim, warmup_epochs=10, n_epochs=79, init_lr=0, min_lr=0,
    )
    lrs = get_lr_schedule(optim=optim, scheduler=scheduler, n_epochs=80, data_size=100, batch_size=5)
    vis = vis_lrs(lrs=lrs, n_steps=80)
    show_image(vis)
