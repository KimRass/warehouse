# Paper Summary
- [Fast Fourier Convolution](https://papers.nips.cc/paper/2020/file/2fd5d41ec6cfab47e32164d5624269b1-Paper.pdf)
## Related Works
- In convolutional networks, receptive field refers to the image part that is accessible by one filter. A majority of modern networks have adopted the architecture of deeply stacking many convolutions with small receptive field ($3 \times 3$ in ResNet for images or $3 \times 3 \times 3$ in C3D for videos). This still ensures that all image parts are visible to high layers, since stacking convolutional layers can increase the receptive field either linearly or exponentially (e.g., using atrous convolutions [2]). However, for context-sensitive tasks such as human pose estimation, large receptive field in convolutions is highly desired. Recent endeavor on enlarging receptive field includes deformable convolution [9] and non-local neural networks [31].
- ***The theory of effective receptive field [21] revealed that convolutions tend to contract to the central regions. This questions the necessity of large convolutional kernels. Besides, small-kernel convolutions are also favored in CNNs for mitigating the risk of over-fitting.***
- ***Recently, researchers gradually realized that linking two arbitrary distant neurons in a layer is crucial for many context-sensitive tasks. This is addressed by recent research on non-local networks.***
- In CNNs, it is widely acknowledged that features extracted from different locations in a network are highly complementary, providing low-level (edges, blobs etc), mid-level (meaningful shapes) or high-level semantic abstraction.
- Cross-scale feature fusion has widely celebrated effectiveness in numerous ways. For example, FCN [20] directly concatenated feature maps of different scales, generating more accurate image segments.
- The visual object detection task requires both accurate localization and prediction of object categories. To this end, FPN [18] propagated features in a top-down manner, seamlessly bridging the high spatial resolution in lower layers and semantic discriminative ability in higher layers.
- Recently-proposed HRNet [29] conducted cross-scale fusion among multiple network branches that maintain different spatial resolutions.
## Methodology
- To our best knowledge, ***FFC is the first work that explores an efficient ensemble of local and non-local receptive fields in a single unit. It can be used in a plug-and-play fashion for easily replacing vanilla convolutions in mainstream CNNs without any additional effort. FFC consumes comparable GFLOPs and parameters with respect to vanilla convolutions, yet conveys richer information.***
## Architecture
- Figure 1
  - <img src="https://user-images.githubusercontent.com/67457712/226341506-103f0897-c3e5-4640-a440-8a151855b7b5.png" width="800">
- ***Conceptually, FFC is comprised of two inter-connected paths:***
  - ***A spatial (or local) path that conducts ordinary convolutions on a part of input feature channels***
  - ***A spectral (or global) path that operates in the spectral domain.***
  - ***Each path can capture complementary information with different receptive field. Information exchange between these paths is performed internally.***
  - Formally, let $X \in \mathbb{R}^{H \times W \times C}$ be the input feature map of some FFC, where $H \times W$, $C$ represent the spatial resolution and the number of channels respectively. At the entry of FFC, we first split $X$ along the dimension of feature channels, namely $X = \{X^{l} , X^{g}\}$. ***The local part*** $X^{l} \in \mathbb{R}^{H \times W \times (1 − \alpha_{in})C}$ ***is expected to learn from local neighborhood and a second global part*** $X^{g} \in \mathbb{R}^{H \times W \times \alpha_{in}C}$ ***is designed to capture long-range context.*** $\alpha_{in} \in [0, 1]$ ***represents the percentage of feature channels allocated to the global part.*** To simplify the network, assume the output is same sized to the input. Use $Y \in \mathbb{R}^{H \times W \times C}$ for the output tensor. Likewise, let $Y = \{Y^{l}, Y^{g}\}$ be a local-global split and the ratio of global part for output tensor is controlled by a hyper-parameter $\alpha_{out} \in [0, 1]$. The updating procedure within FFC can be described by following formulas:
  $$Y^{l} = Y^{l \rightarrow l} + Y^{g \rightarrow l} = f_{l \rightarrow l}(X^{l}) + f_{g \rightarrow l}(X^{g})$$
  $$Y^{g} = Y^{g \rightarrow g} + Y^{l \rightarrow g} = f_{g \rightarrow g}(X^{g}) + f_{l \rightarrow g}(X^{l})$$
  - ***For the component*** $Y^{l \rightarrow l}$ ***which aims to capture small scale information, a regular convolution is adopted. Similarly, other two components (***$Y^{g \rightarrow l}$ ***and*** $Y^{l \rightarrow g}$***) obtained via inter-path transition are also implemented using regular convolutions to take full advantage of multi-scale receptive fields. Major complication stems from the calculation of*** $Y^{g \rightarrow g}$***. For statement clarity, we term*** $f_{g \rightarrow g}$ ***as spectral transformer.***
  - Spectral transformer
    - ***The goal of global path is to enlarge the receptive field of convolution to the full resolution of input feature map in an efficient way.*** We adopt discrete Fourier transform (DFT) for this purpose, using the accelerated version with Cooley-Tukey algorithm. Inspired by the bottleneck block in ResNet, in order to reduce the computational cost, a $1 \times 1$ convolution is used at the beginning for halving the channels. Another $1 \times 1$ convolution is included to restore the feature channel dimension. As seen, between these two convolutions there are one Fourier Unit (FU) with global receptive field, a Local Fourier Unit (LFU) that is designed to capture semi-global information and operates on a quarter of feature channels, and a residual connection.
- Figure 2
  - <img src="https://user-images.githubusercontent.com/67457712/226341586-3088ffcb-fe00-4a37-a681-a631bf2ff850.png" width="500">
- Table 1
  - <img src="https://user-images.githubusercontent.com/67457712/226341643-7969e67c-1599-4628-b689-022a8857e41b.png" width="600">
## References
- [2] [DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs](https://arxiv.org/pdf/1606.00915.pdf)
- [9] [Deformable Convolutional Networks](https://arxiv.org/pdf/1703.06211.pdf)
- [18] [Feature Pyramid Networks for Object Detection](https://arxiv.org/pdf/1612.03144.pdf)
- [20] [Fully Convolutional Networks for Semantic Segmentation, 2015](https://arxiv.org/pdf/1411.4038.pdf)
- [21] [Understanding the Effective Receptive Field in Deep Convolutional Neural Networks](https://arxiv.org/pdf/1701.04128.pdf)
- [29] [Deep High-Resolution Representation Learning for Visual Recognition](https://arxiv.org/pdf/1908.07919.pdf)
- [31] [Non-local Neural Networks](https://arxiv.org/pdf/1711.07971.pdf)
