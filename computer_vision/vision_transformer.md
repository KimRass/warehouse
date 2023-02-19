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
- [AN IMAGE IS WORTH 16X16 WORDS: TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE](https://arxiv.org/pdf/2010.11929.pdf)
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
- Most related to ours is the model of Cordonnier et al. (2020), which extracts patches of size 2 × 2 from the input image and applies full self-attention on top. This model is very similar to ViT, but our work goes further to demonstrate that large scale pre-training makes vanilla transformers competitive with (or even better than) state-of-the-art CNNs. Moreover, Cordonnier et al. (2020) use a small patch size of 2 × 2 pixels, which makes the model applicable only to small-resolution images, while we handle medium-resolution images as well.
## Methodology
- The standard Transformer receives as input a 1D sequence of token embeddings. To handle 2D images, we reshape the image $x \in \mathbb{R}^{H \times W \times C}$ into a sequence of flattened 2D patches $x_{p} \in  \mathbb{R}^{N \times (P^{2} \times C)}$ , where $(H, W)$ is the resolution of the original image, $C$ is the number of channels, $(P, P)$ is the resolution of each image patch, and $N = HW/P^{2}$ is the resulting number of patches, which also serves as the effective input sequence length for the Transformer. The Transformer uses constant latent vector size $D$ through all of its layers, so we flatten the patches and map to $D$ dimensions with a trainable linear projection (Eq. 1). We refer to the output of this projection as the patch embeddings.
- Eq. 1
    - $z_{0} = [x_{class}; x^{1}_{p}E; x^{2}_{p}E; \cdots; x^{N}_{p}E] + E_{pos}$
    - $E \in \mathbb{R}^{(P^{2} \cdot C) \times D}$
    - $E_{pos} \in \mathbb{R}^{(N + 1) \times D}$
- We split an image into fixed-size patches, linearly embed each of them, add position embeddings, and feed the resulting sequence of vectors to a standard Transformer encoder. In order to perform classification, we use the standard approach of adding an extra learnable “classification token” to the sequence.
- **Similar to BERT’s [class] token, we prepend a learnable embedding to the sequence of embedded patches ($z_{0}^{0} = x_{class}$), whose state at the output of the Transformer encoder ($z^{0}_L$) serves as the image representation $y$ (Eq. 4). Both during pre-training and fine-tuning, a classification head is attached to $z^{0}_{L}$. The classification head is implemented by a MLP with one hidden layer at pre-training time and by a single linear layer at fine-tuning time.**
- Position embeddings are added to the patch embeddings to retain positional information. We use standard learnable 1D position embeddings, since we have not observed significant performance gains from using more advanced 2D-aware position embeddings (Appendix D.4). The resulting sequence of embedding vectors serves as input to the encoder.
- he Transformer encoder (Vaswani et al., 2017) consists of alternating layers of multiheaded self- attention (MSA, see Appendix A) and MLP blocks (Eq. 2, 3). Layernorm (LN) is applied before every block, and residual connections after every block (Wang et al., 2019; Baevski & Auli, 2019).
- The first layer of the Vision Transformer linearly projects the flattened patches into a lower-dimensional space
- Inductive bias. We note that Vision Transformer has much less image-specific inductive bias than CNNs. In CNNs, locality, two-dimensional neighborhood structure, and translation equivariance are baked into each layer throughout the whole model. In ViT, only MLP layers are local and transla- tionally equivariant, while the self-attention layers are global. The two-dimensional neighborhood structure is used very sparingly: in the beginning of the model by cutting the image into patches and at fine-tuning time for adjusting the position embeddings for images of different resolution (as de- scribed below). Other than that, the position embeddings at initialization time carry no information about the 2D positions of the patches and all spatial relations between the patches have to be learned from scratch.
- Hybrid Architecture. As an alternative to raw image patches, the input sequence can be formed from feature maps of a CNN (LeCun et al., 1989). In this hybrid model, the patch embedding projection E (Eq. 1) is applied to patches extracted from a CNN feature map.
- Typically, we pre-train ViT on large datasets, and fine-tune to (smaller) downstream tasks. For this, we remove the pre-trained prediction head and attach a zero-initialized D × K feedforward layer, where K is the number of downstream classes. It is often beneficial to fine-tune at higher resolution than pre-training (Touvron et al., 2019; Kolesnikov et al., 2020). When feeding images of higher resolution, we keep the patch size the same, which results in a larger effective sequence length. The Vision Transformer can handle arbitrary sequence lengths (up to memory constraints), however, the pre-trained position embeddings may no longer be meaningful. We therefore perform 2D interpolation of the pre-trained position embeddings, according to their location in the original image. Note that this resolution adjustment and patch extraction are the only points at which an inductive bias about the 2D structure of the images is manually injected into the Vision Transformer.
### Position Embedding
- Position embedding similarity
    - <img src="https://i.stack.imgur.com/zbcvp.png" width="300">
    - Similarity of position embeddings of ViT-L/32. **Tiles show the cosine similarity between the position embedding of the patch with the indicated row and column and the position embeddings of all other patches.** (Comment: e.g., 'Input patch row' 3과 'Input patch column' 4에 해당하는 그리드의 그림은 이 그리와 동일한 위치의 Patch와 다른 나머지 Patches의 Cosine similarity를 나타냅니다. Cosine similarity가 1인 영역은 Patch 자기자신과 연산을 했기 때문입니다.)
    - After the projection, a learned position embedding is added to the patch representations. **The model learns to encode distance within the image in the similarity of position embeddings, i.e. closer patches tend to have more similar position embeddings. Further, the row-column structure appears; patches in the same row/column have similar embeddings. Finally, a sinusoidal structure is sometimes apparent for larger grids (Appendix D). That the position embeddings learn to represent 2D image topology explains why hand-crafted 2D-aware embedding variants do not yield improvements**
### Self-attention
- Mean attention distance
    - <img src="https://machinelearningmastery.com/wp-content/uploads/2022/02/vit_4.png" width="300">
    - *Self-attention allows ViT to integrate information across the entire image even in the lowest layers. We investigate to what degree the network makes use of this capability. Specifically, we compute the average distance in image space across which information is integrated, based on the attention weights This “attention distance” is analogous to receptive field size in CNNs. We find that some heads attend to most of the image already in the lowest layers, showing that the ability to integrate information globally is indeed used by the model.*
    - Further, the attention distance increases with network depth. Globally, we find that the model attends to image regions that are semantically relevant for classification
- Model Variants
    - For instance, ViT-L/16 means the “Large” variant with 16×16 input patch size. Note that the Transformer’s sequence length is inversely proportional to the square of the patch size, thus models with smaller patch size are computationally more expensive.

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