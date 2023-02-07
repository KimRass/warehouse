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

# Paper Summary
## Pure Attention
- In vision, attention is either applied in conjunction with convolutional networks, or used to replace certain components of convolutional networks while keeping their overall structure in place. We show that this reliance on CNNs is not necessary and a pure transformer applied directly to sequences of image patches can perform very well on image classification tasks.
- When pre-trained on large amounts of data and transferred to multiple mid-sized or small image recognition benchmarks (ImageNet, CIFAR-100, VTAB, etc.), Vision Transformer (ViT) attains excellent results compared to state-of-the-art convolutional networks while requiring substantially fewer computational resources to train.
- In computer vision, however, convolutional architectures remain dominant. Inspired by NLP successes, multiple works try combining CNN-like architectures with self-attention, some replacing the convolutions entirely.
## Large-scale Dataset
- *When trained on mid-sized datasets such as ImageNet without strong regularization, these models yield modest accuracies of a few percentage points below ResNets of comparable size. This seemingly discouraging outcome may be expected: Transformers lack some of the inductive biases inherent to CNNs, such as translation equivariance and locality, and therefore do not generalize well when trained on insufficient amounts of data.*
- *However, the picture changes if the models are trained on larger datasets (14M-300M images). We find that large scale training trumps inductive bias. Our Vision Transformer (ViT) attains excellent results when pre-trained at sufficient scale and transferred to tasks with fewer datapoints.*
## Related Works
- Self-attention-based architectures, in particular Transformers (Vaswani et al., 2017), have become the model of choice in natural language processing (NLP). The dominant approach is to pre-train on a large text corpus and then fine-tune on a smaller task-specific dataset (Devlin et al., 2019).
- Transformers were proposed by Vaswani et al. (2017) for machine translation, and have since become the state of the art method in many NLP tasks. Large Transformer-based models are often pre-trained on large corpora and then fine-tuned for the task at hand: BERT uses a denoising self-supervised pre-training task, while the GPT line of work uses language modeling as its pre-training task.
- **Naive application of self-attention to images would require that each pixel attends to every other pixel. With quadratic cost in the number of pixels, this does not scale to realistic input sizes.** Thus, to apply Transformers in the context of image processing, several approximations have been tried in the past.
- Parmar et al. (2018) applied the self-attention only in local neighborhoods for each query pixel instead of globally. Such local multi-head dot-product self attention blocks can completely replace convolutions (Hu et al., 2019; Ramachandran et al., 2019; Zhao et al., 2020).
## Methodology
- *We split an image into patches and provide the sequence of linear embeddings of these patches as an input to a Transformer. Image patches are treated the same way as tokens (words) in an NLP application. We train the model on image classification in supervised fashion.*

# Architecture
- 'ViT-Base' (768-D), 'ViT-Large' (1024-D), 'ViT-Huge' (1280-D)가 있으며 이중 ViT-Base에 대한 내용입니다.
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