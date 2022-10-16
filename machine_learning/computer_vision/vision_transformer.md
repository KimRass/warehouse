# Vision Transformer (ViT)
- Reference: https://www.youtube.com/watch?v=bgsYOGhpxDc&t=174
- Attention을 처음 적용한 Model: Seq2Seq
## CNN Vs. Transformer
- CNN
    - 이미지 전체의 정보를 통합하기 위해서는 몇 개의 Layers를 통과시켜야만 합니다.
    - 2차원의 Locality를 유지합니다.
    - 학습 후 Weights가 고정됩니다.
- Tansformer
    - 하나의 Attention layer만으로 전체 이미지의 정보를 통합할 수 있습니다.
    - Input에 따라 Weights가 유동적으로 변합니다.
    - Inductive bias가 더 낮고 Model의 자유도는 더 높습니다.

# Architecture
- 'ViT-Base', 'ViT-Large', 'ViT-Huge'가 있으며 이중 ViT-Base에 대한 내용입니다.
- 'ViT-Base/16'에서 '16'의 의미는, Input image를 16x16의 Image patch로 분할한다는 것입니다. 따라서 각 Image patch는 16 x 16 x 3(= 768)차원입니다.
- Linear projection: 768-D -> 768-D FCL
- Classification token: 768-D
- Positional embedding: 768-D
- Classification token + Positional embedding: ViT encoder의 Input
## Encoder
- ![encoder_architecture](https://theaisummer.com/static/aa65d942973255da238052d8cdfa4fcd/7d4ec/the-transformer-block-vit.png)
### 'Norm' Part
- Vanilla transformer과 다르게 두 번의 Layer normalization의 위치가 각각 Multi-head attention과 FCL의 앞으로 변경됐습니다.
### 'Multi-Head Attention' Part
- Input (768-D)로부터 768 x 86의 Shape을 가지는 각 Matrix를 곱해 64-D로 동일한 Query, Key, Value를 생성합니다.
- Multi-head attention을 12번 수행한 후 Concatenate하여 64-D x 12(= 768-D)의 Vectors를 생성합니다. 그리고 이 결과를 Input과 더합니다. (Skip connection)
### 'MLP' Part
- 768-D -> 3072-D -> 768-D로 변환하는 FCL를 사용합니다.
### Activation Function
- GELU

# DeiT (Data efficient image Transformer)
- Indcutive bias가 없어 많은 데이터가 필요하다는 ViT의 한계를 극복한 Model입니다.
- Teach model: RegNet (CNN-based)