# Paper Reading
- [DropBlock: A regularization method for convolutional networks](https://arxiv.org/pdf/1810.12890.pdf)
## Related Works
- [1]
    - ***We argue that the main drawback of dropout is that it drops out features randomly. While this can be effective for fully connected layers, it is less effective for convolutional layers, where features are correlated spatially. When the features are correlated, even with dropout, information about the input can still be sent to the next layer, which causes the networks to overfit.*** This intuition suggests that a more structured form of dropout is needed to better regularize convolutional networks.
    - Since its introduction, Dropout[1] has inspired a number of regularization methods for neural networks. ***The basic principle behind these methods is to inject noise into neural networks so that they do not overfit the training data.***
- [23]
    - Our method is inspired by Cutout [23], a data augmentation method where parts of the input examples are zeroed out. ***DropBlock generalizes Cutout by applying Cutout at every feature map in a convolutional networks.***
## Introduction
- In DropBlock, features in a block, i.e., a contiguous region of a feature map, are dropped together. As DropBlock discards features in a correlated area, ***the networks must look elsewhere for evidence to fit the data.***
- Figure 1
    - <img src="https://user-images.githubusercontent.com/105417680/232967687-139c6589-ba09-456e-97dc-08129923e7e4.png" width="700">
    - The green regions in (b) and (c) include the activation units which contain semantic information in the input image.
    - (b): Dropping out activations at random is not effective in removing semantic information because nearby activations contain closely related information.
    - (c): Dropping continuous regions can remove certain semantic information (e.g., head or feet) and consequently enforcing remaining units to learn features for classifying input image.
## Methodology
- DropBlock has two main parameters which are $block\_size$ and $\gamma$. $block\_size$ is the size of the block to be dropped, and $\gamma$, controls how many activation units to drop.
- Similar to dropout we do not apply DropBlock during inference.
- Setting the value of $block\_size$.
    - ***We set a constant*** $block\_size$ ***for all feature maps, regardless the resolution of feature map.*** DropBlock resembles dropout [1] when $block\_size$ = 1 and resembles SpatialDropout [20] when $block\_size$ covers the full feature map.
- Setting the value of $\gamma$.
    - In practice, we do not explicitly set $\gamma$. As stated earlier, $\gamma$ controls the number of features to drop. Suppose that we want to keep every activation unit with the probability of $keep\_prob$, in dropout [1] the binary mask will be sampled with the Bernoulli distribution with mean $1 − keep\_prob$. However, to account for the fact that every zero entry in the mask will be expanded by $block\_size$2 and the blocks will be fully contained in feature map, we need to adjust $\gamma$ accordingly when we sample the initial binary mask. In our implementation, $\gamma$ can be computed as
- Figure 2
    - <img src="https://user-images.githubusercontent.com/105417680/232974285-1816083a-3976-4c58-bb59-394c802f6c71.png" width="500">
## Experiments
- ***Having a fixed zero-out ratio for DropBlock during training is not as robust as having an increasing schedule for the ratio during training. In other words, it’s better to set the DropBlock ratio to be small initially during training, and linearly increase it over time during training.***
## References
- [1] [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](https://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf)
- [20] [Efficient Object Localization Using Convolutional Networks](https://arxiv.org/pdf/1411.4280.pdf)
- [23] [Improved Regularization of Convolutional Neural Networks with Cutout](https://arxiv.org/pdf/1708.04552.pdf)