# Reference:
    # https://nn.labml.ai/normalization/batch_norm/index.html

import torch
import torch.nn as nn


class BatchNorm(nn.Module):
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        ema: bool = True,
    ):
        """
        "Note that when applying batch normalization after a linear transform like $Wu + b$
        the bias parameter $b$ gets cancelled due to normalization. So you can and should omit
        bias parameter in linear transforms right before the batch normalization."
        "We need to know mean and variance in order to perform the normalization.
        So during inference, you either need to go through the whole (or part of) dataset
        and find the mean and variance, or you can use an estimate calculated during training.
        The usual practice is to calculate an exponential moving average of mean and variance
        during the training phase and use that for inference."

        Args:
            `track_running_stats`: Whether to calculate the moving averages or mean and variance
        """
        super().__init__()
    
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.ema = ema

        if ema:
            self.register_buffer("ema_mean", torch.zeros(num_features))  # EMA of the mean.
            self.register_buffer("ema_var", torch.ones(num_features))  # EMA of the variance.

        if self.affine:
            self.scale = nn.Parameter(torch.ones(num_features))
            self.shift = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        b, c, *_ = x.shape
        assert self.num_features == c

        x = x.view(b, c, -1)
        if self.training or not self.ema:
            # 학습 중이라면 항상 (`self.training`), 그리고 추론 중이라 할지라도 평균과 분산
            # 각각의 EMA를 업데이트하고 있지 않다면 (`not self.track_running_stats`),
            # mini-batch에 대해서 계산한 평균과 분산을 가지고 normalize.
            mean = x.mean(dim=(0, 2))
            var = (x ** 2).mean(dim=(0, 2)) - mean ** 2

            if self.ema:
                # 학습 중이라면 평균과 분산 각각의 EMA를 계속 업데이트.
                self.ema_mean = (1 - self.momentum) * self.ema_mean + self.momentum * mean
                self.ema_var = (1 - self.momentum) * self.ema_var + self.momentum * var
        else:
            # 추론 중이고 평균과 분산 각각의 EMA를 업데이트하고 있다면 그들을 가지고
            # normalize.
            mean = self.ema_mean
            var = self.ema_var
        x = (x - mean.view(1, -1, 1)) / (var.view(1, -1, 1) + self.eps) ** 0.5

        if self.affine:
            x = self.scale.view(1, -1, 1) * x + self.shift.view(1, -1, 1)

        x = x.view(b, c, *_)
        return x


if __name__ == "__main__":
    batch_size = 16
    num_features = 768
    h = 64
    w = 64
    device = torch.device("mps")
    x = torch.randn((batch_size, num_features, h, w), device=device)

    bn = BatchNorm(num_features=num_features, ema=True).to(device)
    bn.train()
    out = bn(x)
    bn.eval()
    out = bn(x)

    bn = BatchNorm(num_features=num_features, ema=False).to(device)
    bn.train()
    out = bn(x)
    bn.eval()
    out = bn(x)
