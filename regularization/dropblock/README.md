# Paper Reading
- [DropBlock: A Regularization Method for Convolutional Networks, 2018](https://arxiv.org/pdf/1810.12890.pdf)
## Methodology
- Figure 2
    - <img src="https://user-images.githubusercontent.com/105417680/232974285-1816083a-3976-4c58-bb59-394c802f6c71.png" width="450">
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
        - <img src="https://user-images.githubusercontent.com/105417680/233578482-d801c6c6-98b5-4ae2-a013-04b985d9120c.png" width="650">
        - The original ResNet architecture does not apply any dropout in the model. For the ease of discussion, we define the dropout baseline for ResNet as applying dropout on convolution branches only.
        - '(a)': DropBlock outperforms dropout with 1.3% for top-1 accuracy.
        - '(b)': ***The scheduled*** $keep\textunderscore prob$ ***makes DropBlock more robust to the change of*** $keep\textunderscore prob$ ***and adds improvement for the most values of*** $keep\textunderscore prob$***.***
    - Figure 4
        - <img src="https://user-images.githubusercontent.com/105417680/233579158-15b45329-a4af-41d8-a741-09af9cb99082.png" width="850">
        - With the best keep_prob found in Figure 3, we swept over block_size from 1 to size covering full feature map. Figure 4 shows applying larger block_size is generally better than applying block_size of 1. The best DropBlock configuration is to apply block_size = 7 to both groups 3 and 4. In all configurations, DropBlock and dropout share the similar trend and DropBlock has a large gain compared to the best dropout result. This shows evidence that the DropBlock is a more effective regularizer compared to dropout.
- Comparison with Cutout
    - ***Although Cutout improves accuracy on the CIFAR-10 dataset as suggested by [23], it does not improve the accuracy on the ImageNet dataset in our experiments.***
- DropBlock in AmoebaNet [10]
    - We apply DropBlock after all batch normalization layers and also in the skip connections of the last 50% of the cells. The resolution of the feature maps in these cells are $21 \times 21$ or $11 \times 11$ for input image with the size of $331 \times 331$. Based on the experiments in the last section, we used $keep\textunderscore prob = 0.9$ and set $block\textunderscore size = 11$ which is the width of the last feature map.
- Figure 5
    - <img src="https://user-images.githubusercontent.com/105417680/233234487-40b89cfc-4ded-4724-b817-87359e1eb7e1.png" width="700">
    - '(a)': When we apply DropBlock with $block\textunderscore size = 1$ at inference with different $keep\textunderscore prob$
    - '(b)': When we apply DropBlock with $block\textunderscore size = 7$ at inference with different $keep\textunderscore prob$.
    <!-- - The models are trained and evaluated with DropBlock at groups 3 and 4. -->
    - DropBlock drops more semantic information
        - 'trained without DropBlock`: We first took the model trained without any regularization and tested it with DropBlock with $block\textunderscore size = 1$ and $block\textunderscore size = 7$. The validation accuracy reduced quickly with decreasing $keep\textunderscore prob$ during inference. ***This suggests DropBlock removes semantic information and makes classification more difficult.*** (Comment: 초록색 선이, $block\textunderscore size$가 감소함에 따라 '(a)'보다 '(b)'에서 'accuracy (validation)'가 더 급격하게 감소합니다.)
        - 'trained with block_size=1' and 'trained with block_size=7': ***The accuracy drops more quickly with decreasing*** $keep\textunderscore prob$***, for*** $block\textunderscore size = 1$ ***in comparison with*** $block\textunderscore size = 7$ ***which suggests DropBlock is more effective to remove semantic information than dropout.*** (Comment: 빨간색 선과 파란색 선이, $block\textunderscore size$가 감소함에 따라 각각 '(a)'보다 '(b)'에서 'accuracy (validation)'가 더 급격하게 감소합니다.)
    - Model trained with DropBlock is more robust
        - Next we show that model trained with large block size, which removes more semantic information, results in stronger regularization.

        - 'trained with block_size=1' and 'trained with block_size=7': ResNet-50 model trained with $block\textunderscore size = 7$ has higher accuracy compared to the ResNet-50 model trained with $block\textunderscore size = 1$.
    - 'trained with block_size=7' in (a) and 'trained_with_block_size=1' in (b): We show that model trained with large block size, which removes more semantic information, results in stronger regularization. We demonstrate the fact by taking model trained with $block\textunderscore size = 7$ and applied $block\textunderscore size = 1$ during inference and vice versa. The performance of model trained with $block\textunderscore size = 1$ reduced more quickly with decreasing $keep\textunderscore prob$ when applying $block\textunderscore size = 7$ during inference. The results suggest that $block\textunderscore size$= 7$ is more robust and has the benefit of $block\textunderscore size = 1$ but not vice versa.
## References
- [1] [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](https://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf)
- [10] [Regularized Evolution for Image Classifier Architecture Search](https://arxiv.org/pdf/1802.01548.pdf)
- [17] [FractalNet: Ultra-Deep Neural Networks without Residuals](https://arxiv.org/pdf/1605.07648.pdf)
- [20] [Efficient Object Localization Using Convolutional Networks](https://arxiv.org/pdf/1411.4280.pdf)
- [23] [Improved Regularization of Convolutional Neural Networks with Cutout](https://arxiv.org/pdf/1708.04552.pdf)
- [24] [Focal Loss for Dense Object Detection](https://arxiv.org/pdf/1708.02002.pdf)
- [27] [AutoAugment: Learning Augmentation Policies from Data](https://arxiv.org/pdf/1805.09501.pdf)
- [28] [Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/pdf/1512.00567.pdf)
- [29] [Learning Deep Features for Discriminative Localization](https://arxiv.org/pdf/1512.04150.pdf)
