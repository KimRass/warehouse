# GANs
- Random noise

# cGANs (Conditional GANs)

# PatchGAN

# Pix2Pix
- Source: https://deep-learning-study.tistory.com/645
## Generator
- Image를 입력으로 받아 Image를 출력하도록 학습됩니다.
- U-Net의 구조를 사용합니다.
```python
class UNetDown(nn.Module):
    def __init__(self, in_channels, out_channels, normalize=True, dropout=0.0):
        super().__init__()

        layers = [nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1, bias=False)]

        if normalize:
            layers.append(nn.InstanceNorm2d(out_channels)),

        layers.append(nn.LeakyReLU(0.2))

        if dropout:
            layers.append(nn.Dropout(dropout))

        self.down = nn.Sequential(*layers)

    def forward(self, x):
        x = self.down(x)
        return x


class UNetUp(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()

        layers = [
            nn.ConvTranspose2d(in_channels, out_channels,4,2,1,bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU()
        ]

        if dropout:
            layers.append(nn.Dropout(dropout))

        self.up = nn.Sequential(*layers)

    def forward(self,x,skip):
        x = self.up(x)
        x = torch.cat((x,skip),1)
        return x


class GeneratorUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()

        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64,1 28)                 
        self.down3 = UNetDown(128, 256)               
        self.down4 = UNetDown(256, 512, dropout=0.5) 
        self.down5 = UNetDown(512, 512, dropout=0.5)      
        self.down6 = UNetDown(512, 512, dropout=0.5)             
        self.down7 = UNetDown(512, 512, dropout=0.5)              
        self.down8 = UNetDown(512, 512, normalize=False, dropout=0.5)

        self.up1 = UNetUp(512, 512, dropout=0.5)
        self.up2 = UNetUp(1024, 512, dropout=0.5)
        self.up3 = UNetUp(1024, 512, dropout=0.5)
        self.up4 = UNetUp(1024, 512, dropout=0.5)
        self.up5 = UNetUp(1024, 256)
        self.up6 = UNetUp(512, 128)
        self.up7 = UNetUp(256, 64)
        self.up8 = nn.Sequential(
            nn.ConvTranspose2d(128, 3, 4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)

        u1 = self.up1(d8,d7)
        u2 = self.up2(u1,d6)
        u3 = self.up3(u2,d5)
        u4 = self.up4(u3,d4)
        u5 = self.up5(u4,d3)
        u6 = self.up6(u5,d2)
        u7 = self.up7(u6,d1)
        u8 = self.up8(u7)

        return u8
```
## Discriminator
- Fake image를 0으로, Real image를 1로 예측하도록 학습됩니다.
- PatchGAN을 사용합니다. 출력은 하나의 Scalar가 아니라 Feature map입니다. 입력이 256x256x3인 경우 출력은 30x30입니다. 이때 30x30의 각 Pixel에 대해서 Real/Fake를 판별합니다.
- 또한 patch gan은 조건부 gan 이므로 조건부 데이터를 입력으로 받습니다.
- PatchGAN을 사용하면 High-frequency의 정확도가 향상됩니다. High-frequency의 정확도가 향상된다는 말은 디테일한 부분이 향상된다는 것으로 해석할 수 있습니다.
```python
class DiscriminatorBlock(nn.Module):
    def __init__(self, in_channels, out_channels, normalize=True):
        super().__init__()

        layers = [nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2))
    
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        x = self.block(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()

        self.stage_1 = DiscriminatorBlock(in_channels*2,64,normalize=False)
        self.stage_2 = DiscriminatorBlock(64,128)
        self.stage_3 = DiscriminatorBlock(128,256)
        self.stage_4 = DiscriminatorBlock(256,512)

        self.patch = nn.Conv2d(512,1,3,padding=1) # 16x16 패치 생성

    def forward(self,a,b):
        x = torch.cat((a,b),1)
        x = self.stage_1(x)
        x = self.stage_2(x)
        x = self.stage_3(x)
        x = self.stage_4(x)
        x = self.patch(x)
        x = torch.sigmoid(x)
        return x
```
## Loss function
- Fake image와 Real image 사이의 L1 loss (즉 Pixel 값의 차이)
- ![pix2pix_loss_function](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FQmtFJ%2Fbtq5bMRYiID%2FQMOOK86k411yIj3YhGOe0k%2Fimg.png)