# Paper Understanding
- [Unsupervised Representation Learning by Predicting Image Rotations, 2018](https://arxiv.org/pdf/1803.07728.pdf)
## Related Works
- In order to learn features, [1] and [2] train ConvNets to colorize gray scale images, [3] and [4] predict the relative position of image patches, and [5] predict the egomotion (i.e., self-motion) of a moving vehicle between two consecutive frames.
- The core intuition of our self-supervised feature learning approach is that if someone is not aware of the concepts of the objects depicted in the images, he cannot recognize the rotation that was applied to them.
- Other successful cases of unsupervised feature learning are clustering based methods ([6], [7] and [8]), reconstruction based methods (Bengio et al., 2007; Huang et al., 2007; Masci et al., 2011), and methods that involve learning generative probabilistic models Goodfellow et al. (2014); Donahue et al. (2016); Radford et al. (2015).
## Introduction
- Self-supervised learning defines an annotation free pretext task, using only the visual information present on the images or videos, in order to provide a surrogate supervision signal for feature learning.
## Methodolgy
<!-- - We propose to train a ConvNet model $F$ to estimate the geometric transformation applied to an image that is given to it as input. Specifically, we define a set of $K$ discrete geometric transformations $G = \{g(; y)\}^{K}_{y = 1}$, where $g(; y)$ is the operator that applies the geometric transformation $y$ to image $X$ that yields the transformed image $X^{y} = g(X; y)$. The ConvNet model $F$ gets an image $X^{y^*}$ (where $y^{∗}$ is unknown to model $F$) as input (Comment: $y^{*}$는 실제로 $X$에 가해진 회전 변환, 즉 ground truth를 의미합니다.) and yields as output a probability distribution over all possible geometric transformations:
$$p(X^{y^{∗}}; \theta) = \{F^{y}(X^{y^{*}}; \theta)\}^{K}_{y = 1}$$
- where $F^{y}(X^{y^{∗}}; \theta)$ is the predicted probability for the geometric transformation $k$ and $\theta$ are the learnable parameters of model $F$. Therefore, given a set of $N$ training images $D = \{X_{i}\}^{N}_{i = 1}$, the self-supervised training objective that the ConvNet model must learn to solve is:
$$\argmin_{\theta}\frac{1}{N}\sum^{N}_{i = 1}loss(X_{i}, \theta)$$
- where the loss function $loss$ is defined as:
$$loss(X_{i}, \theta) = −\frac{1}{K}\sum^{K}_{k = 1}\log\bigg(p^{k}\big(g(X_{i}; k); \theta\big)\bigg)$$
- Figure 2
    - <img src="https://user-images.githubusercontent.com/67457712/227820083-dea02047-3d9d-43d0-8936-e4172c94aeec.png" width="700">
    - In the above formulation, the geometric transformations G must define a classification task that should force the ConvNet model to learn semantic features useful for visual perception tasks (e.g., object detection or image classification). In our work we propose to define the set of geometric transformations G as all the image rotations by multiples of 90 degrees, i.e., 2d image rotations by 0, 90, 180, and 270 degrees (see Figure 2). More formally, if Rot(X, φ) is an operator that rotates image X by φ degrees, then our set of geometric transformations consists of the K = 4 image rotations G = {g(X|y)} 4 y=1, where g(X|y) = Rot(X,(y − 1)90). -->
- Comment: 원래의 논문에서는 이미지 회전 변환에 관한 수식이 등장하나, 필요 이상으로 어렵게 설명되어 있습니다. 단순하게 4가지 회전 변환에 대한 분류 문제를 풀도록 모델의 구조를 짜면 되며 loss는 cross entropy를 사용하면 됩니다.
- The core intuition behind using these image rotations as the set of geometric transformations relates to the simple fact that **it is essentially impossible for a ConvNet model to effectively perform the above rotation recognition task unless it has first learned to recognize and detect classes of objects as well as their semantic parts in images. More specifically to successfully predict the rotation of an image the ConvNet model must necessarily learn to localize salient objects in the image, recognize their orientation and object type, and then relate the object orientation with the dominant orientation that each type of object tends to be depicted within the available images.***
- Absence of low-level visual artifacts: An additional important advantage of using image rotations by multiples of 90 degrees over other geometric transformations, is that they can be implemented by flip and transpose operations ***that do not leave any easily detectable low-level visual artifacts that will lead the ConvNet to learn trivial features with no practical value for the vision perception tasks. In contrast, had we decided to use as geometric transformations, e.g., scale and aspect ratio image transformations, in order to implement them we would need to use image resizing routines that leave easily detectable image artifacts.***
- No ambiguity: Given an image rotated by 0, 90, 180, or 270 degrees, ***there is usually no ambiguity of what is the rotation transformation (with the exception of images that only depict round objects). In contrast, that is not the case for the object scale that varies significantly on human captured images.***
### Training
- ***We found that we get significant improvement when during training we train the network by feeding it all the four rotated copies of an image simultaneously instead of each time randomly sampling a single rotation transformation. Therefore, at each training batch the network sees 4 times more images than the batch size.***
## Experiments
- Figure 3
    - <img src="https://user-images.githubusercontent.com/105417680/228284692-dde6840d-8b27-4f2a-acee-27d90688d22b.png" width="750">
    - '(b)': We visualize some attention maps generated by a model trained on the rotation recognition task.
    - ***These attention maps are computed based on the magnitude of activations at each spatial cell of a convolutional layer and essentially reflect where the network puts most of its focus in order to classify an input image. We observe that in order for the model to accomplish the rotation prediction task it learns to focus on high level object parts in the image, such as eyes, nose, tails, and heads.***
    - '(a)': By comparing them with the attention maps generated by a model trained on the object recognition task in a supervised way we observe that both models seem to focus on roughly the same image regions.
    - In order to generate the attention map of a conv. layer ***we first compute the feature maps of this layer, then we raise each feature activation on the power*** $p$***, and finally we sum the activations at each location of the feature map. For the conv. layers 1, 2, and 3 we used the powers*** $p = 1$***,*** $p = 2$***,*** ***and*** $p = 4$ ***respectively.***
- Figure 6
    - <img src="https://user-images.githubusercontent.com/105417680/228836382-3630b46a-0c2e-4cc1-a3b8-89d60204fcb5.png" width="500">
    - ***We observe that the attention maps of all the rotated copies of an image are roughly the same, i.e., the attention maps are equivariant w.r.t. the image rotations. This practically means that in order to accomplish the rotation prediction task the network focuses on the same object parts regardless of the image rotation.***
- Figure 4: Visualization of the first layer filters of AlexNet
    - <img src="https://user-images.githubusercontent.com/67457712/227821123-ea732b32-49ca-4783-a849-9ec18da55a78.png" width="600">
    - '(a)': First layer filters learned by a AlexNet model trained on the supervised object recognition task
    - '(b)': Trained on the proposed rotation recognition task. As can be seen, they appear to have a big variety of edge filters on multiple orientations and multiple frequencies. Remarkably, these filters seem to have a greater amount of variety even than the filters learned by the supervised object recognition task ('(a)').
- Table 2: The quality of the self-supervised learned features w.r.t. the number of recognized rotations
    - <img src="https://user-images.githubusercontent.com/67457712/227821317-0973c388-b632-47b6-80a6-cb531009e778.png" width="550">
    - We explore how the quality of the self-supervised features depends on the number of discrete rotations used in the rotation prediction task. For that purpose we defined three extra rotation recognition tasks: 
        - '(a)': One with 8 rotations that includes all the multiples of 45 degrees,
        - '(b)': One with only the 0 and 180 degrees rotations
        - '(c)': One with only the 90 and 270 degrees rotations.
    - We observe that for 4 discrete rotations we achieve better object recognition performance than the 8 or 2 cases. We believe that this is because ***the 2 orientations case offers too few classes for recognition (i.e., less supervisory information is provided) while in the 8 orientations case the geometric transformations are not distinguishable enough and furthermore the 4 extra rotations introduced may lead to visual artifacts on the rotated images.***
    - We observe that among the RotNet models trained with 2 discrete rotations, the RotNet model trained with 90 and 270 degrees rotations achieves worse object recognition performance than the model trained with the 0 and 180 degrees rotations, which is probably due to the fact that the former model does not 'see' the 0 degree rotation during the unsupervised phase that is typically used during the object recognition training phase. (Comment: 모델이 object recognition에 대해 supervised learning 중에는 회전되지 않은 이미지(0 degree rotation)를 보게 되는데, image rotation에 대해 self-supervised learning 중에는 90도나 270도 회전된 이미지만 보지 회전되지 않은 이미지는 보지 못하지 때문입니다.)
## References
- [1] [Colorful Image Colorization](https://arxiv.org/pdf/1603.08511.pdf)
- [2] [Learning Representations for Automatic Colorization](https://arxiv.org/pdf/1603.06668.pdf)
- [3] [Unsupervised Visual Representation Learning by Context Prediction, 2015](https://arxiv.org/pdf/1505.05192.pdf)
- [4] [Unsupervised Learning of Visual Representations by Solving Jigsaw Puzzles, 2016](https://arxiv.org/pdf/1603.09246.pdf)
- [5] [Learning to See by Moving](https://arxiv.org/pdf/1505.01596.pdf)
- [6] [Discriminative Unsupervised Feature Learning with Exemplar Convolutional Neural Networks](https://arxiv.org/pdf/1406.6909.pdf)
- [7] [Learning Deep Parsimonious Representations]
- [8] [Joint Unsupervised Learning of Deep Representations and Image Clusters]
