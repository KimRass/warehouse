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
- Figure 2
    - <img src="https://user-images.githubusercontent.com/105417680/232974285-1816083a-3976-4c58-bb59-394c802f6c71.png" width="500">
    - '(a)': On every feature map, similar to dropout, we first sample a mask $M$. We only sample mask from shaded green region in which each sampled entry can expanded to a mask fully contained inside the feature map.
    - '(b)': Every zero entry on $M$ is expanded to $block\textunderscore size \times block\textunderscore size$ zero block.
- DropBlock has two main parameters which are $block\textunderscore size$ and $\gamma$. $block\textunderscore size$ is the size of the block to be dropped, and $\gamma$, controls how many activation units to drop.
- ***Similar to dropout we do not apply DropBlock during inference.***
- Setting the value of $block\textunderscore size$.
    - ***We set a constant*** $block\textunderscore size$ ***for all feature maps, regardless the resolution of feature map.*** DropBlock resembles dropout [1] when $block\textunderscore size$ = 1 and resembles SpatialDropout [20] when $block\textunderscore size$ covers the full feature map.
- Setting the value of $\gamma$.
    - In our implementation, $\gamma$ can be computed as
    $$\gamma = \frac{(1 - keep\textunderscore prob) \cdot feat\textunderscore width \cdot feat\textunderscore height}{block\textunderscore size^{2} \cdot (feat\textunderscore width - block\textunderscore size + 1) \cdot (feat\textunderscore height - block\textunderscore size + 1)}$$
    - (Comment: $\gamma$를 위 식을 통해 정하게 되면 이것을 Parameter로 하여 베르누이 분포로부터 Mask를 샘플링했을 때 그 Mask의 평균이 $keep\textunderscore prob$에 매우 가까운 값을 가지게 됩니다.)
## Training
- Data augmentation
    - We used horizontal flip, scale, and aspect ratio augmentation for training images as in [12] and [26].
- We trained models on Tensor Processing Units (TPUs) and used the official Tensorflow implementations for ResNet-501 and AmoebaNet2 . We used the default image size (224 × 224 for ResNet-50 and 331 × 331 for AmoebaNet), batch size (1024 for ResNet-50 and 2048 for AmoebaNet) and hyperparameters setting for all the models. We only increased number of training epochs from 90 to 270 for ResNet-50 architecture. The learning rate was decayed by the factor of 0.1 at 125, 200 and 250 epochs. AmoebaNet models were trained for 340 epochs and exponential decay scheme was used for scheduling learning rate. Since baselines are usually overfitted for the longer training scheme and have lower validation accuracy at the end of training, we report the highest validation accuracy over the full training course for fair comparison.
## Experiments
- Scheduled DropBlock
    - ***Having a fixed zero-out ratio for DropBlock during training is not as robust as having an increasing schedule for the ratio during training. In other words, it’s better to set the DropBlock ratio to be small initially during training, and linearly increase it over time during training.***
    - Scheduled DropBlock. We found that DropBlock with a fixed $keep\textunderscore prob$ during training does not work well. Applying small value of $keep\textunderscore prob$ hurts learning at the beginning. Instead, gradually decreasing $keep\textunderscore prob$ over time from 1 to the target value is more robust and adds improvement for the most values of $keep\textunderscore prob$.
- DropBlock in ResNet-50
    - Table 1
        - <img src="https://user-images.githubusercontent.com/105417680/233234178-95b4e940-6863-4f32-ad1b-142fb790f921.png" width="600">
        - For dropout [1], DropPath [17], and SpatialDropout [20], we trained models with different $keep\textunderscore prob$ values and reported the best result. DropBlock is applied with $block\textunderscore size = 7$. We report average over 3 runs.
        - ***DropBlock has better performance compared to strong data augmentation [27] and label smoothing [28]. The performance improves when combining DropBlock and label smoothing and train for 290 epochs, showing the regularization techniques can be complimentary when we train for longer.***
- DropBlock vs. dropout [1]
    - Figure 3
    - The original ResNet architecture does not apply any dropout in the model. For the ease of discussion, we define the dropout baseline for ResNet as applying dropout on convolution branches only.
- Comparison with Cutout
    - ***Although Cutout improves accuracy on the CIFAR-10 dataset as suggested by [23], it does not improve the accuracy on the ImageNet dataset in our experiments.***
- DropBlock in AmoebaNet [10]
    - We apply DropBlock after all batch normalization layers and also in the skip connections of the last 50% of the cells. The resolution of the feature maps in these cells are $21 \times 21$ or $11 \times 11$ for input image with the size of $331 \times 331$. Based on the experiments in the last section, we used $keep\textunderscore prob = 0.9$ and set $block\textunderscore size = 11$ which is the width of the last feature map.
- DropBlock drops more semantic information
    - Figure 5
        - <img src="https://user-images.githubusercontent.com/105417680/233234487-40b89cfc-4ded-4724-b817-87359e1eb7e1.png" width="700">
        - (a): When we apply DropBlock with $block\textunderscore size = 1$ at inference with different $keep\textunderscore prob$
        - (b): When we apply DropBlock with $block\textunderscore size = 7$ at inference with different $keep\textunderscore prob$.
        - The models are trained and evaluated with DropBlock at groups 3 and 4.
        - 'trained with block_size=1' and 'trained with block_size=7' in (a) and (b): ResNet-50 model trained with $block\textunderscore size = 7$ and $keep\textunderscore prob = 0.9$ has higher accuracy compared to the ResNet-50 model trained with $block\textunderscore size = 1$ and $keep\textunderscore prob = 0.9$.
        - (a) and (b): ***The accuracy drops more quickly with decreasing*** $keep\textunderscore prob$***, for*** $block\textunderscore size = 1$ ***in comparison with*** $block\textunderscore size = 7$ ***which suggests DropBlock is more effective to remove semantic information than dropout.*** (Comment: (a)와 (b)를 서로 같은 색깔의 선에 대해서 비교해 봤을 때, (a)보다 (b)에서 'accuracy (validation)'이 더 낮습니다.)
        - 'trained with block_size=7' in (a) and 'trained_with_block_size=1' in (b): We show that model trained with large block size, which removes more semantic information, results in stronger regularization. We demonstrate the fact by taking model trained with $block\textunderscore size = 7$ and applied $block\textunderscore size = 1$ during inference and vice versa. The performance of model trained with $block\textunderscore size = 1$ reduced more quickly with decreasing $keep\textunderscore prob$ when applying $block\textunderscore size = 7$ during inference. The results suggest that block_size = 7 is more robust and has the benefit of block_size = 1 but not vice versa.
- Figure 6
    - <img src="https://user-images.githubusercontent.com/105417680/233234589-8bf0304c-ecf9-475d-a41a-31b6df8a2239.png" width="800">
## References
- [1] [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](https://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf)
- [10] [Regularized Evolution for Image Classifier Architecture Search](https://arxiv.org/pdf/1802.01548.pdf)
- [17] [FractalNet: Ultra-Deep Neural Networks without Residuals](https://arxiv.org/pdf/1605.07648.pdf)
- [20] [Efficient Object Localization Using Convolutional Networks](https://arxiv.org/pdf/1411.4280.pdf)
- [23] [Improved Regularization of Convolutional Neural Networks with Cutout](https://arxiv.org/pdf/1708.04552.pdf)
- [27] [AutoAugment: Learning Augmentation Policies from Data](https://arxiv.org/pdf/1805.09501.pdf)
- [28] [Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/pdf/1512.00567.pdf)