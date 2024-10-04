import torch
import torch.nn as nn


class ConvolutionBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding="same")
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x


class MaxPoolingBlock(nn.Module):
    def __init__(self):
        super().__init__()

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.maxpool(x)
        x = self.relu(x)
        return x


class VGG19(nn.Module):
    def __init__(self, n_classes):
        super().__init__()

        self.conv_3_64 = ConvolutionBlock(in_ch=3, out_ch=64)
        self.conv_64_64 = ConvolutionBlock(in_ch=64, out_ch=64)
        self.conv_64_128 = ConvolutionBlock(in_ch=64, out_ch=128)
        self.conv_128_128 = ConvolutionBlock(in_ch=128, out_ch=128)
        self.conv_128_256 = ConvolutionBlock(in_ch=128, out_ch=256)
        self.conv_256_256 = ConvolutionBlock(in_ch=256, out_ch=256)
        self.conv_256_512 = ConvolutionBlock(in_ch=256, out_ch=512)
        self.conv_512_512 = ConvolutionBlock(in_ch=512, out_ch=512)

        self.maxpool = MaxPoolingBlock()

        self.linear1 = nn.Linear(512 * 7 * 7, 4096)
        self.linear2 = nn.Linear(4096, 4096)
        self.linear3 = nn.Linear(4096, n_classes)
    
    def forward(self, x):
        x = self.conv_3_64(x) # Weight layer #1
        x = self.conv_64_64(x) # Weight layer #2
        x = self.maxpool(x)
        x = self.conv_64_128(x) # Weight layer #3
        x = self.conv_128_128(x) # Weight layer #4
        x = self.maxpool(x)
        x = self.conv_128_256(x) # Weight layer #5
        x = self.conv_256_256(x) # Weight layer #6
        x = self.conv_256_256(x) # Weight layer #7
        x = self.conv_256_256(x) # Weight layer #8
        x = self.maxpool(x)
        x = self.conv_256_512(x) # Weight layer #9
        x = self.conv_512_512(x) # Weight layer #10
        x = self.conv_512_512(x) # Weight layer #11
        x = self.conv_512_512(x) # Weight layer #12
        x = self.maxpool(x)
        x = self.conv_512_512(x) # Weight layer #13
        x = self.conv_512_512(x) # Weight layer #14
        x = self.conv_512_512(x) # Weight layer #15
        x = self.conv_512_512(x) # Weight layer #16
        x = self.maxpool(x)

        b, _, _, _ = x.shape
        x = x.view((b, -1))

        x = self.linear1(x) # Weight layer #17
        x = self.linear2(x) # Weight layer #18
        x = self.linear3(x) # Weight layer #19
        return x


if __name__ == "__main__":
    vgg19 = VGG19(n_classes=1_000)

    input = torch.randn((4, 3, 224, 224))
    vgg19(input).shape
