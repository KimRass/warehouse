# Paper Reading
- [Context Encoders: Feature Learning by Inpainting, 2016](https://arxiv.org/pdf/1604.07379.pdf)
## Related Works
- Autoencoder
    - The context encoder is closely related to autoencoders, as it shares a similar encoder-decoder architecture. Autoencoders take an input image and try to reconstruct it after it passes through a low-dimensional "bottleneck" layer, with the aim of obtaining a compact feature representation of the scene. Unfortunately, ***this feature representation is likely to just compresses the image content without learning a semantically meaningful representation.***
- Denoising autoencoder
    - Address this issue by corrupting the input image and requiring the network to undo the damage. However, this corruption process is typically very localized and low-level, and does not require much semantic information to undo.
    - Denoising autoencoders reconstruct the image from local corruptions, to make encoding robust to such corruptions. While context encoders could be thought of as a variant of denoising autoencoders, the corruption applied to the model’s input is spatially much larger, requiring more semantic information to undo.
- Word2vec [3]
    - Learns word representation from natural language sentences by predicting a word given its context.
- [7]
    - ***One important benefit of our approach is that our supervisory signal is much richer: a context encoder needs to predict roughly 15,000 real values per training example, compared to just 1 option among 8 choices in [7]. Likely due in part to this difference, our context encoders take far less time to train than [7]. Moreover, context based prediction is also harder to "cheat" since low-level image features, such as chromatic aberration, do not provide any meaningful information, in contrast to [7] where chromatic aberration partially solves the task.*** On the other hand, it is not yet clear if requiring faithful pixel generation is necessary for learning good visual features.
## Methodology
- Region masks
    - "Central region"
        - ***While this works quite well for inpainting, the network learns low level image features that latch onto the boundary of the central mask. Those low level image features tend not to generalize well to images without masks, hence the features learned are not very general.***
    - "Random block"
        - Instead of choosing a single large mask at a fixed location, we remove a number of smaller possibly overlapping masks, covering up to $\frac{1}{4}$ of the image. However, ***the random block masking still has sharp boundaries convolutional features could latch onto.***
    - "Random region"
        - ***To completely remove those boundaries, we experimented with removing arbitrary shapes from images***, obtained from random masks in the PASCAL VOC 2012 dataset. We deform those shapes and paste in arbitrary places in the other images (not from PASCAL), again covering up to $\frac{1}{4}$ of the image. Note that ***we completely randomize the region masking process, and do not expect or want any correlation between the source segmentation mask and the image.*** We merely use those regions to prevent the network from learning low-level features corresponding to the removed mask.
    - In practice, ***we found "Random block" and "Random region" masks produce a similarly general feature, while significantly outperforming the "Central region" features. We use the random region dropout for all our feature based experiments.***
## Architecture
- Figure 9
    <img src="https://user-images.githubusercontent.com/105417680/235658801-c2b5d110-a590-40ea-8725-1a4c0bfbc3c6.png" width="900">
    - Encoder-decoder pipeline
        - The encoder takes an input image with missing regions and produces a latent feature representation of that image. The decoder takes this feature representation and produces the missing image content. We found it important to connect the encoder and the decoder through a channel-wise fully-connected layer, which allows each unit in the decoder to reason about the entire image content.
    - Encoder
        - ***Our encoder is derived from the AlexNet architecture [26]. Given an input image of size*** $227 \times 227$***, we use the first five convolutional layers and the following pooling layer (called "pool5") to compute an abstract*** $6 \times 6 \times 256$ ***dimensional feature representation.***
        - However, ***if the encoder architecture is limited only to convolutional layers, there is no way for information to directly propagate from one corner of the feature map to another. This is so because convolutional layers connect all the feature maps together, but never directly connect all locations within a specific feature map.*** In the present architectures, this information propagation is handled by fully-connected where all the activations are directly connected to each other. In our architecture, ***the latent feature dimension is*** $6 \times 6 \times 256 = 9216$ ***for both encoder and decoder. This is so because, unlike autoencoders, we do not reconstruct the original input and hence need not have a smaller bottleneck. However, fully connect-ing the encoder and decoder would result in an explosion in the number of parameters (over 100M!), to the extent that efficient training on current GPUs would be difficult. To alleviate this issue, we use a channel-wise fully-connected layer to connect the encoder features to the decoder.***
    - Channel-wise fully-connected layer
        - This layer is essentially a fully-connected layer with groups, intended to propagate information within activations of each feature map. If the input layer has $m$ feature maps of size $n \times n$, this layer will output $m$ feature maps of dimension $n \times n$. However, ***unlike a fully-connected layer, it has no parameters connecting different feature maps and only propagates information within feature maps. Thus, the number of parameters in this channel-wise fully-connected layer is*** $mn^{4}$***, compared to*** $m^{2}n^{4}$ ***parameters in a fully-connected layer (ignoring the bias term).*** This is followed by a stride 1 convolution to propagate information across channels.
    - Decoder
        - The "encoder features" are connected to the "decoder features" using a channel-wise fully- connected layer. ***The channel-wise fully-connected layer is followed by a series of five up-convolutional layers [28] [40] with learned filters, each with a rectified linear unit (ReLU) activation function.*** A up-convolutional is simply a convolution that results in a higher resolution image. It can be understood as upsampling followed by convolution (as described in [10]), or convolution with fractional stride (as described in [28]).
## Training
### Loss
- ***There are often multiple equally plausible ways to fill a missing image region which are consistent with the context. We model this behavior by having a decoupled joint loss function to handle both continuity within the context and multiple modes in the output. The reconstruction (L2) loss is responsible for capturing the overall structure of the missing region and coherence with regards to its context, but tends to average together the multiple modes in predictions. The adversarial loss, on the other hand, tries to make prediction look real, and has the effect of picking a particular mode from the distribution.***
- For each ground truth image $x$, our context encoder $$ produces an output $F(x)$. ***Let*** $\hat{M}$ ***be a binary mask corresponding to the dropped image region with a value of*** $1$ ***wherever a pixel was dropped and*** $0$ ***for input pixels. During training, those masks are automatically generated for each image and training iterations.***
## Fine-tunning
- We further validate the quality of the learned feature representation by fine-tuning the encoder for a variety of image understanding tasks, including classification, object detection, and semantic segmentation





Reconstruction Loss We use a normalized masked L2
distance as our reconstruction loss function, Lrec,
Lrec(x) = kMˆ (x − F((1 − Mˆ ) x))k
2
2
, (1)
where is the element-wise product operation. We experi-
mented with both L1 and L2 losses and found no significant
difference between them. While this simple loss encour-
ages the decoder to produce a rough outline of the predicted
object, it often fails to capture any high frequency detail
(see Fig. 1c). This stems from the fact that the L2 (or L1)
loss often prefer a blurry solution, over highly accurate tex-
tures. We believe this happens because it is much "safer"
for the L2 loss to predict the mean of the distribution, be-
cause this minimizes the mean pixel-wise error, but results
in a blurry averaged image. 

The objective for discriminator is
logistic likelihood indicating whether the input is real sample or predicted one:
min
G
max
D
Ex∈X [log(D(x))] + Ez∈Z [log(1 − D(G(z)))] 

However, conditional GANs don’t
train easily for context prediction task as the adversarial dis-
criminator D easily exploits the perceptual discontinuity in
generated regions and the original context to easily classify
predicted versus real samples. We thus use an alternate for-
mulation, by conditioning only the generator (not the dis-
criminator) on context. We also found results improved
when the generator was not conditioned on a noise vector.
Hence the adversarial loss for context encoders, Ladv, is
Ladv = max
D
Ex∈X [log(D(x))
+ log(1 − D(F((1 − Mˆ ) x)))], (2)
where, in practice, both F and D are optimized jointly us-
ing alternating SGD. Note that this objective encourages the 

entire output of the context encoder to look realistic, not just
the missing regions as in Equation (1).
Joint Loss We define the overall loss function as
L = λrecLrec + λadvLadv.


The missing region in
the masked input image is filled with constant mean value. 

Pool-free encoders We experimented with replacing all
pooling layers with convolutions of the same kernel size
and stride. The overall stride of the network remains the
same, but it results in finer inpainting. Intuitively, there is
no reason to use pooling for reconstruction based networks. 

In classification, pooling provides spatial invariance, which
may be detrimental for reconstruction-based training. To be
consistent with prior work, we still use the original AlexNet
architecture (with pooling) for all feature learning results. 

We now evaluate the encoder features for their seman-
tic quality and transferability to other image understanding
tasks. We experiment with images from two datasets: Paris
StreetView [8] and ImageNet [37] without using any of the
accompanying labels.

## References
- [7] [Unsupervised Visual Representation Learning by Context Prediction, 2015](https://arxiv.org/pdf/1505.05192.pdf)
- [26] [ImageNet Classification with Deep Convolutional Neural Networks, 2012](https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)
- [28] [Fully Convolutional Networks for Semantic Segmentation, 2015](https://arxiv.org/pdf/1411.4038.pdf)
- [30] [Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/pdf/1310.4546.pdf)
- [40] [Visualizing and Understanding Convolutional Networks, 2014](https://arxiv.org/pdf/1311.2901.pdf)

# TO DO
## Paper reading
1. Loss function 부분 내용 정리
## PyTorch Implementation
1. Loss function 구현
