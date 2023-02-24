- Reference: [Bag of Tricks for Image Classification with Convolutional Neural Networks](https://arxiv.org/pdf/1812.01187.pdf)
- Learning rate warmup.
    - **At the beginning of the training, all parameters are typically random values and therefore far away from the final solution. Using a too large learning rate may result in numerical instability. In the warmup heuristic, we use a small learning rate at the beginning and then switch back to the initial learning rate when the training process is stable.
    - Goyal et al. [7] proposes a gradual warmup strategy that increases the learning rate from 0 to the initial learning rate linearly. In other words, assume we will use the first $m$ batches (e.g. 5 data epochs) to warm up, and the initial learning rate is $\eta$, then at batch $i$ ($1 \le i \le m$) we will set the learning rate to be $\frac{\eta}{m}$.

- Reference: https://gaussian37.github.io/dl-pytorch-lr_scheduler/