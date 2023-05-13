# Refereces
    # https://github.com/jazzsaxmafia/Inpainting/blob/master/src/model.py
    # https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/linear.py

import torch
import torch.nn as nn
import torchvision.transforms as T
import math


BATCH_SIZE = 16
IMG_SIZE = 227


class AlexNetFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 96, kernel_size=11, stride=4)
        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.conv5(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x


class ChannelwiseFullyConnetecdLayer(nn.Module):
    def __init__(self, features: int, feature_size: int, device=None, dtype=None) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}

        super().__init__()

        self.features = features
        self.feature_size = feature_size

        self.weight = nn.Parameter(torch.empty(size=(feature_size ** 2, feature_size ** 2, features), **factory_kwargs))
        self.bias = nn.Parameter(torch.empty(size=(features, feature_size ** 2), **factory_kwargs))
        self.reset_parameters()
    
    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input.view((-1, self.features, self.feature_size ** 2))
        x = torch.einsum("bij,jji->bij", x, self.weight) + self.bias
        x = x.view((-1, self.features, self.feature_size, self.feature_size))
        return x

    def extra_repr(self) -> str:
        return f"""features={self.features}, feature_size={self.feature_size}"""


class Decoder(nn.Module):
    def __init__(self, img_size):
        super().__init__()

        self.upconv1 = nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2, padding=2)
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=2)
        self.upconv3 = nn.ConvTranspose2d(64, 64, kernel_size=5, stride=2, padding=2)
        self.upconv4 = nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2, padding=2)
        self.upconv5 = nn.ConvTranspose2d(32, 3, kernel_size=5, stride=2, padding=2)
        
        self.resize = T.Resize(size=img_size)

    def forward(self, x):
        x = self.upconv1(x)
        x = self.upconv2(x)
        x = self.upconv3(x)
        x = self.upconv4(x)
        x = self.upconv5(x)
        
        x = self.resize(x)
        return x


class ContextEncoder(nn.Module):
    def __init__(self, encoder, img_size=IMG_SIZE):
        super().__init__()

        self.encoder = encoder
        self.decoder = Decoder(img_size=img_size)

    def forward(self, x):
        x = self.encoder(input)

        _, m, n, n = x.shape
        x = ChannelwiseFullyConnetecdLayer(features=m, feature_size=n)(x)

        x = self.decoder(x)
        return x
    

if __name__ == "__main__":
    feat_extractor = AlexNetFeatureExtractor()
    
    context_enc = ContextEncoder(encoder=AlexNetFeatureExtractor())
    input = torch.randn((BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE))
    out = context_enc(input)
    out.shape