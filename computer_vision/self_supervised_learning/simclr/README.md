# Paper Reading
- [A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/pdf/2002.05709.pdf)
## Introduction
- ***Composition of multiple data augmentation operations is crucial in defining the contrastive prediction tasks that yield effective representations.***
- ***Unsupervised contrastive learning benefits from larger batch sizes and longer training compared to its supervised counterpart.*** Like supervised learning, contrastive learning benefits from deeper and wider networks.
## Related Works
- Self-supervised learning (Representation learning)
    - ***Many approaches have relied on heuristics to design pretext tasks [1] [2] [3]***, which could limit the generality of the learned representations. Discriminative approaches based on contrastive learning in the latent space have recently shown great promise, achieving state-of-the-art results.

- Figure 2
    - SimCLR learns representations by maximizing agreement between differently augmented views of the same data example via a contrastive loss in the latent space.
    - Data augmentation
        - ***A stochastic data augmentation module that transforms any given data example randomly resulting in two correlated views of the same example, denoted*** $\tilde{x}_{i}$ ***and*** $\tilde{x}_{j}$***, which we consider as a positive pair. In this work, we sequentially apply three simple augmentations: random cropping followed by resize back to the original size, random color distortions, and random Gaussian blur.***
    - A neural network base encoder $f(\cdot)$ that extracts representation vectors from augmented data examples. Our framework allows various choices of the network architecture without any constraints. We adopt the commonly used ResNet to obtain $h_{i} = f(\tilde{x}_{i}) = \text{ResNet}(\tilde{x}_{i})$ where $h_{i} \in \mathbb{R}^{d}$ ***is the output after the average pooling layer.***
    - A small neural network projection head $g(\cdot)$ that maps representations to the space where contrastive loss is applied. We use a MLP with one hidden layer to obtain $z_{i} = g(h_{i}) = W_{2}\text{ReLU}(W_{1}h_{i})$. ***We find it beneficial to define the contrastive loss on*** $z_{i}$***’s rather than*** $h_{i}$’s ***.***
## Methodology
### Data Augmentation
- In our default pre-training setting (which is used to train our best models), we utilize random crop (with resize and random flip), random color distortion, and random Gaussian blur as the data augmentations.
- Random crop and resize to $224 \times 224$
    <!-- - We use standard Inception-style random cropping (Szegedy et al., 2015). -->
    - The crop of random size (uniform from 0.08 to 1.0 in area) of the original size and a random aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. This crop is finally resized to the original size. This has been implemented in Pytorch as `torchvision.transforms.RandomResizedCrop`. Additionally, the random crop (with resize) is always followed by a random horizontal/left-to-right flip with 50% probability.
- Color distortion
    - Color distortion is composed by color jittering and color dropping. We find stronger color jittering usually helps, so we set a strength parameter.
- Gaussian blur
    - We blur the image 50% of the time using a Gaussian kernel. We randomly sample $σ \in [0.1, 2]$, and the kernel size is set to be 10% of the image height/width.


While data
augmentation has been widely used in both supervised and
unsupervised representation learning (Krizhevsky et al.,
2012; Hénaff et al., 2019; Bachman et al., 2019), it has
not been considered as a systematic way to define the con-
trastive prediction task. 


We randomly sample a minibatch of N examples and define
the contrastive prediction task on pairs of augmented exam-
ples derived from the minibatch, resulting in 2N data points.
We do not sample negative examples explicitly. Instead,
given a positive pair, similar to (Chen et al., 2017), we treat
the other 2(N − 1) augmented examples within a minibatch
as negative examples. Let sim(u, v) = u
>v/kukkvk de-
note the cosine similarity between two vectors u and v.
Then the loss function for a positive pair of examples (i, j)
is defined as
`i,j = − log exp(sim(zi
, zj )/τ )
P2N
k=1 1[k6=i] exp(sim(zi
, zk)/τ )
, (1)
where 1[k6=i] ∈ {0, 1} is an indicator function evaluating to
1 iff k 6= i and τ denotes a temperature parameter. The fi-
nal loss is computed across all positive pairs, both (i, j)
and (j, i), in a mini-batch. This loss has been used in
previous work (Sohn, 2016; Wu et al., 2018; Oord et al.,
2018); for convenience, we term it NT-Xent (the normalized
temperature-scaled cross entropy loss).
Algorithm 1 summarizes the proposed method. 

Global BN. Standard ResNets use batch normaliza-
tion (Ioffe & Szegedy, 2015). In distributed training with
data parallelism, the BN mean and variance are typically
aggregated locally per device. In our contrastive learning,
as positive pairs are computed in the same device, the model
can exploit the local information leakage to improve pre-
diction accuracy without improving representations. We
address this issue by aggregating BN mean and variance
over all devices during the training. Other approaches in-
clude shuffling data examples (He et al., 2019a), or replacing
BN with layer norm (Hénaff et al., 2019).


Most of our study for unsupervised
pretraining (learning encoder network f without labels)
is done using the ImageNet ILSVRC-2012 dataset
To evalu-
ate the learned representations, we follow the widely used
linear evaluation protocol (Zhang et al., 2016; Oord et al.,
2018; Bachman et al., 2019; Kolesnikov et al., 2019), where
a linear classifier is trained on top of the frozen base net-
work, and test accuracy is used as a proxy for representation
quality. Beyond linear evaluation, we also compare against
state-of-the-art on semi-supervised and transfer learning.


Default setting. Unless otherwise specified, for data aug-
mentation we use random crop and resize (with random flip),
color distortions, and Gaussian blur (for details, see Ap-
pendix A). We use ResNet-50 as the base encoder network,
and a 2-layer MLP projection head to project the representa-
tion to a 128-dimensional latent space. As the loss, we use
NT-Xent, optimized using LARS with linear learning rate
scaling (i.e. LearningRate = 0.3 × BatchSize/256) and
weight decay of 10−6
. We train at batch size 4096 for 100
epochs.2 Furthermore, we use linear warmup for the first 10
epochs, and decay the learning rate with the cosine decay
schedule without restarts (Loshchilov & Hutter, 2016). 


After
training is completed, we through away the projection head g(·)
and use encoder f(·) and representation h for downstream tasks



## Studies
- Composition of data augmentation operations is crucial for learning good representations
    - Figure 5. Linear evaluation (ImageNet top-1 accuracy) under individual or composition of data augmentations, applied only to one branch.
        - For all columns but the last, diagonal entries corre- spond to single transformation, and off-diagonals correspond to composition of two transformations (applied sequentially). The last column reflects the average over the row.- Figure 5. Linear evaluation (ImageNet top-1 accuracy) under in- dividual or composition of data augmentations, applied only to one branch. For all columns but the last, diagonal entries corre- spond to single transformation, and off-diagonals correspond to composition of two transformations (applied sequentially). The last column reflects the average over the row.
    - we always first randomly crop im- ages and resize them to the same resolution, and we then apply the targeted transformation(s) only to one branch of the framework in Figure 2, while leaving the other branch as the identity (i.e. t(xi) = xi). Note that this asymmet- ric data augmentation hurts the performance.
    - We observe that no single transformation suffices to learn good representations, even though the model can almost perfectly identify the positive pairs in the contrastive task. When composing aug- mentations, the contrastive prediction task becomes harder, but the quality of representation improves dramatically.
    - One composition of augmentations stands out: random crop- ping and random color distortion. We conjecture that one serious issue when using only random cropping as data augmentation is that most patches from an image share a similar color distribution. Figure 6 shows that color his- tograms alone suffice to distinguish images. Neural nets may exploit this shortcut to solve the predictive task. There- fore, it is critical to compose cropping with color distortion in order to learn generalizable features.

Figure 6. Histograms of pixel intensities (over all channels) for
different crops of two different images (i.e. two rows). 

Contrastive learning needs stronger data
augmentation than supervised learning
Table 1. Top-1 accuracy of unsupervised ResNet-50 using linear
evaluation and supervised ResNet-504
, under varied color distor-
tion strength (see Appendix A) and other data transformations.
Strength 1 (+Blur) is our default data augmentation policy. 

To further demonstrate the importance of the color aug-
mentation, we adjust the strength of color augmentation as
shown in Table 1. Stronger color augmentation substan-
tially improves the linear evaluation of the learned unsu-
pervised models. In this context, AutoAugment (Cubuk
et al., 2019), a sophisticated augmentation policy found us-
ing supervised learning, does not work better than simple
cropping + (stronger) color distortion. When training supervised models with the same set of augmentations, we
observe that stronger color augmentation does not improve
or even hurts their performance. Thus, our experiments
show that unsupervised contrastive learning benefits from
stronger (color) data augmentation than supervised learning.
Although previous work has reported that data augmenta-
tion is useful for self-supervised learning (Doersch et al.,
2015; Bachman et al., 2019; Hénaff et al., 2019), we show
that data augmentation that does not yield accuracy bene-
fits for supervised learning can still help considerably with
contrastive learning. 

Unsupervised contrastive learning benefits (more)
from bigger models
Figure 7 shows, perhaps unsurprisingly, that increasing
depth and width both improve performance. While similar
findings hold for supervised learning (He et al., 2016), we
find the gap between supervised models and linear classifiers
trained on unsupervised models shrinks as the model size
increases, suggesting that unsupervised learning benefits
more from bigger models than its supervised counterpart. 

Figure 7. Linear evaluation of models with varied depth and width.
Models in blue dots are ours trained for 100 epochs, models in red
stars are ours trained for 1000 epochs, and models in green crosses
are supervised ResNets trained for 90 epochs6
(He et al., 2016).
Training longer does not improve supervised ResNets (see
Appendix B.1). 

A nonlinear projection head improves the
representation quality of the layer before it
We then study the importance of including a projection
head, i.e. g(h). Figure 8 shows linear evaluation results
using three different architecture for the head: (1) identity 

mapping; (2) linear projection, as used by several previous
approaches (Wu et al., 2018); and (3) the default nonlinear
projection with one additional hidden layer (and ReLU acti-
vation), similar to Bachman et al. (2019). We observe that a
nonlinear projection is better than a linear projection (+3%),
and much better than no projection (>10%). When a pro-
jection head is used, similar results are observed regardless
of output dimension. Furthermore, even when nonlinear
projection is used, the layer before the projection head, h,
is still much better (>10%) than the layer after, z = g(h),
which shows that the hidden layer before the projection
head is a better representation than the layer after

## References
- [1] [Unsupervised Visual Representation Learning by Context Prediction, 2015](https://arxiv.org/pdf/1505.05192.pdf)
- [2] [Unsupervised Learning of Visual Representations by Solving Jigsaw Puzzles, 2016](https://arxiv.org/pdf/1603.09246.pdf)
- [3] [Unsupervised Representation Learning by Predicting Image Rotations, 2018](https://arxiv.org/pdf/1803.07728.pdf)
- [4] [Discriminative Unsupervised Feature Learning with Exemplar Convolutional Neural Networks, 2014]