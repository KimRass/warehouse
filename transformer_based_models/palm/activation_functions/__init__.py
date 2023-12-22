# References
    # https://medium.com/deeplearningmadeeasy/glu-gated-linear-unit-21e71cd52081
    # https://deep-learning-study.tistory.com/556

import torch
import torch.nn as nn
import torch.nn.functional as F


class GLU(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.dim = dim

        self.proj = nn.Linear(dim, 2 * dim)

    def forward(self, x):
        x = self.proj(x)
        x1, x2 = torch.split(x, split_size_or_sections=self.dim, dim=-1)
        x = x2 * F.sigmoid(x1)
        return x

class GatedLinearUnit(GLU):
    def __init__(self, dim):
        super().__init__(dim=dim)


class Swish(nn.Module): # A.k.a "SiLU"
    def __init__(self, beta):
        super().__init__()

        self.beta = beta

    def forward(self, x):
        return x * F.sigmoid(self.beta * x)


class SwiGLU(nn.Module):
    def __init__(self, dim, beta):
        super().__init__()

        self.dim = dim
        self.beta = beta

        self.proj = nn.Linear(dim, 2 * dim)

    def forward(self, x):
        x = self.proj(x)
        x1, x2 = torch.split(x, split_size_or_sections=self.dim, dim=-1)
        x = x2 * Swish(beta=self.beta)(x1)
        return x


class ReGLU(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.dim = dim

        self.proj = nn.Linear(dim, 2 * dim)

    def forward(self, x):
        x = self.proj(x)
        x1, x2 = torch.split(x, split_size_or_sections=self.dim, dim=-1)
        x = x2 * F.relu(x1)
        return x


class GEGLU(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.dim = dim

        self.proj = nn.Linear(dim, 2 * dim)

    def forward(self, x):
        x = self.proj(x)
        x1, x2 = torch.split(x, split_size_or_sections=self.dim, dim=-1)
        x = x2 * F.gelu(x1)
        return x


if __name__ == "__main__":
    BATCH_SIZE = 32
    SEQ_LEN = 30
    DIM = 512
    x = torch.randn((BATCH_SIZE, SEQ_LEN, DIM))

    BETA = 0.1
    activ = GLU(dim=DIM)
    # activ = GatedLinearUnit(dim=DIM)
    activ = Swish(beta=BETA)
    activ = SwiGLU(dim=DIM, beta=BETA)
    activ = ReGLU(dim=DIM)
    activ = GEGLU(dim=DIM)
    out = activ(x)
    print(out.shape)
