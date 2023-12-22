# Reference:
    # https://nn.labml.ai/normalization/batch_norm/index.html

# "Note that when applying batch normalization after a linear transform like $Wu + b$
# the bias parameter $b$ gets cancelled due to normalization. So you can and should omit bias parameter
# in linear transforms right before the batch normalization."

# "We need to know mean and variance in order to perform the normalization.
# So during inference, you either need to go through the whole (or part of) dataset
# and find the mean and variance, or you can use an estimate calculated during training.
# The usual practice is to calculate an exponential moving average of mean and variance
# during the training phase and use that for inference."

import torch
import torch.nn as nn


class BatchNorm(nn.Module):
    def __init__(self, n_features, eps=1e-5, affine=True, track_running_stats=True):
        """
        `n_features`: Same as `num_features`.
        `affine`: Same as `affine`.
        """
        super().__init__()
    
        self.n_features = n_features
        self.eps = eps
        self.affine = affine

        if affine:
            self.gamma = nn.Parameter(torch.ones(n_features))
            self.beta = nn.Parameter(torch.zeros(n_features))

    def forward(self, x, training=False): # `(batch_size, n_features, ...)`
        ori_shape = x.shape
        b, c = ori_shape[: 2]
        assert self.n_features == c,\
            "The argument `n_features` must be same as the number of features of the input!"

        if training:
            x = x.view(b, c, -1)
            mean = x.mean(dim=(0, 2), keepdim=True)
            var = (x ** 2).mean(dim=(0, 2), keepdim=True) - mean ** 2
            x = (x - mean) / torch.sqrt(var + self.eps)
        else:
            mean = self.exp_mean
            var = self.exp_var


        if self.affine:
            x = self.gamma.view(1, -1, 1) * x + self.beta.view(1, -1, 1)

        x = x.view(ori_shape)
        return x
