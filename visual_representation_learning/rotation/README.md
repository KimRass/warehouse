 # Paper Understanding
- [Unsupervised Representation Learning by Predicting Image Rotations, 2018](https://arxiv.org/pdf/1803.07728.pdf)
## Related Works
- In order to learn features, [1] and [2] train ConvNets to colorize gray scale images, [3] and [4] predict the relative position of image patches, and [5] predict the egomotion (i.e., self-motion) of a moving vehicle between two consecutive frames.
- The core intuition of our self-supervised feature learning approach is that if someone is not aware of the concepts of the objects depicted in the images, he cannot recognize the rotation that was applied to them.
- Other successful cases of unsupervised feature learning are clustering based methods ([6], [7] and [8]), reconstruction based methods (Bengio et al., 2007; Huang et al., 2007; Masci et al., 2011), and methods that involve learning generative probabilistic models Goodfellow et al. (2014); Donahue et al. (2016); Radford et al. (2015).
## Introduction
- Self-supervised learning defines an annotation free pretext task, using only the visual information present on the images or videos, in order to provide a surrogate supervision signal for feature learning.
## Methodolgy
- Comment: 원래의 논문에서는 이미지 회전 변환에 관한 수식이 등장하나, 필요 이상으로 어렵게 설명되어 있습니다. 단순하게 4가지 회전 변환에 대한 분류 문제를 풀도록 모델의 구조를 짜면 되며 loss는 cross entropy를 사용하면 됩니다.
- The core intuition behind using these image rotations as the set of geometric transformations relates to the simple fact that **it is essentially impossible for a ConvNet model to effectively perform the above rotation recognition task unless it has first learned to recognize and detect classes of objects as well as their semantic parts in images. More specifically to successfully predict the rotation of an image the ConvNet model must necessarily learn to localize salient objects in the image, recognize their orientation and object type, and then relate the object orientation with the dominant orientation that each type of object tends to be depicted within the available images.***
- Absence of low-level visual artifacts: An additional important advantage of using image rotations by multiples of 90 degrees over other geometric transformations, is that they can be implemented by flip and transpose operations ***that do not leave any easily detectable low-level visual artifacts that will lead the ConvNet to learn trivial features with no practical value for the vision perception tasks. In contrast, had we decided to use as geometric transformations, e.g., scale and aspect ratio image transformations, in order to implement them we would need to use image resizing routines that leave easily detectable image artifacts.***
- No ambiguity: Given an image rotated by 0, 90, 180, or 270 degrees, ***there is usually no ambiguity of what is the rotation transformation (with the exception of images that only depict round objects). In contrast, that is not the case for the object scale that varies significantly on human captured images.***
### Training
- ***We found that we get significant improvement when during training we train the network by feeding it all the four rotated copies of an image simultaneously instead of each time randomly sampling a single rotation transformation. Therefore, at each training batch the network sees 4 times more images than the batch size.***
## Experiments
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
