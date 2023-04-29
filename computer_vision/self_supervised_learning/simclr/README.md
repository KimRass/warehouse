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

## References
- [1] [Unsupervised Visual Representation Learning by Context Prediction, 2015](https://arxiv.org/pdf/1505.05192.pdf)
- [2] [Unsupervised Learning of Visual Representations by Solving Jigsaw Puzzles, 2016](https://arxiv.org/pdf/1603.09246.pdf)
- [3] [Unsupervised Representation Learning by Predicting Image Rotations, 2018](https://arxiv.org/pdf/1803.07728.pdf)
- [4] [Discriminative Unsupervised Feature Learning with Exemplar Convolutional Neural Networks, 2014]