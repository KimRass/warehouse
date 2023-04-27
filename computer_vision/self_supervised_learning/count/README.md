# Paper Reading
- [Representation Learning by Learning to Count](https://arxiv.org/pdf/1708.06734.pdf)
## Related Works
- Self-supervised learning
    - Some recent feature learning methods, in the so-called self-supervised learning paradigm, have managed to avoid annotation by defining a task which provides a supervision signal. For example, some methods recover color from gray scale images and vice versa [21] [22] [43] [44], recover a whole patch from the surrounding pixels [33], or recover the relative location of patches [9] [29]. ***These methods use information already present in the data as supervision signal A rationale behind self-supervised learning is that pretext tasks that relate the most to the final problems (e.g., classification and detection) will be more likely to build relevant representations.***
    - ***A technique that substitutes the labels for a task with artificial or surrogate ones.***
    - Regression
        - In recent work [33] choose as surrogate label $x_{2}$ a region of pixels in an image and use the remaining pixels in the image as $x_{1}$.
        - Other related work [21] and [43] maps images to the Lab (luminance and opponent colors) space, and then uses the opponent colors as labels $x_{2}$ and the luminance as $x_{1}$.
        - [44] combine this choice to the opposite task of predicting the grayscale image from the opponent colors and outperform prior work.
    - Classification
        - [9] and [29] define a categorical problem where the surrogate labels are the relative positions of patches.
## Introduction
- As a novel candidate pretext task, we propose counting visual primitives. To obtain a supervision signal useful to learn to count, ***we exploit the following property: If we partition an image into non-overlapping regions, the number of visual primitives in each region should sum up to the number of primitives in the original image We make the hypothesis that the model needs to disentangle the image into high-level factors of variation, such that the complex relation between the original image and its regions is translated to a simple arithmetic operation [35].***
## Methodology
- We define the counting relationship "having the same number of visual primitives" between two images. ***We use the fact that this relationship is satisfied by two identical images undergoing certain transformations, but not by two different images (although they might, with very low probability). Thus, we are able to assign a binary label (same or different number of visual primitives) to pairs of images.***
- One way to characterize a feature of interest is to describe how it should vary as a function of changes in the input data. For example, a feature that counts visual primitives should not be affected by scale, 2D translation, and 2D rotation changes of the input image. Other relationships might indicate instead that a feature should increase its values as a result of some transformation of the input. For example, the magnitude of the feature for counting visual primitives applied to half of an image should be smaller than when applied to the whole image. In general, we propose to learn a deep representation by using the known relationship between input and output transformations as a supervisory signal.
- Let us denote a color image with $\textbf{x} \in \mathbb{R}^{m \times n \times 3}$ , where $m \times n$ is the size in pixels and there are 3 color channels (RGB).
- We define a family of image transformations $\mathcal{G} := \{G_{1}, \ldots, G_{J}\}$, where $G_{j} : \mathbb{R}^{m \times n \times 3} \rightarrow \mathbb{R}^{p \times q \times 3}$ , with $j = 1, \ldots, J$ that take images $x$ and map them to images of $p \times q$ pixels.
- Let us also define a feature $\phi : \mathbb{R}^{p \times q \times 3} \rightarrow \mathbb{R}^{k}$ mapping the transformed image to some $k$-dimensional vector.
- Finally, we define a feature transformation $g : \mathbb{R}^{k} \times \cdots \times \mathbb{R}^{k} \rightarrow \mathbb{R}^{k}$ that takes $J$ features and maps them to another feature.
- Given the image transformation family $\mathcal{G}$ and feature transformation family $g$, we learn the feature $\phi$ by using the following relationship as an artificial supervisory signal;
- Equantion 1
$$g\big(\phi(G_{1} \circ \textbf{x}), \ldots, \phi(G_{J} \circ \textbf{x})\big) = \textbf{0} \quad \forall \textbf{x}$$
- In this work, ***the transformation family consists of the downsampling operator*** $D$***, with a downsampling factor of*** $2$***, and the tiling operator*** $T_{j}$***, where*** $j = 1, \ldots, 4$***, which extracts the*** $j$***−th tile from a*** $2 \times 2$ ***grid of tiles. Notice that these two transformations produce images of the same size. Thus, we can set*** $\mathcal{G} := \{D, T_{1}, \ldots, T_{4}\}$***.***
- ***We also define our desired relation between counting features on the transformed images as*** $g(d, t_{1}, \ldots, t_{4}) = d − \sum^{4}_{j = 1}t_{j}$***. This can be written explicitly as;***
- Equation 2
$$\phi(D \circ \textbf{x}) = \sum^{4}_{j=1}\phi(T_{j} \circ \textbf{x})$$
- Figure 2
    - <img src="https://user-images.githubusercontent.com/67457712/234449734-f7330820-3bf8-4252-a6fa-1c8a0cbc2eb8.png" width="550">
## Training
### Loss
- We use convolutional neural networks to obtain our representation. In principle, our network could be trained with color images $\textbf{x}$ from a large database (e.g., ImageNet or COCO) using an $l_{2}$ loss based on Equation 2, for example,
- Equation 3
    $$\ell(x) = \bigg\lvert \phi(D \circ x) − \sum^{4}_{j = 1}\phi(T_{j} \circ x)\bigg\rvert^{2}$$
- However, this loss has $\phi(\textbf{z}) = \textbf{0}, \forall \textbf{z}$, as its trivial solution. To avoid such a scenario, ***we use a contrastive loss, where we also enforce that the counting feature should be different between two randomly chosen different images.*** Therefore, for any $\textbf{x} \neq \textbf{y}$, we would like to minimize
- Equation 4
$$\ell_{con}(\textbf{x}, \textbf{y}) = \bigg\lvert \phi(D \circ \textbf{x}) − \sum^{4}_{j = 1}\phi(T_{j} \circ \textbf{x})\bigg\rvert^{2} + \max\bigg\{0, M − \bigg\lvert \phi(D \circ \textbf{y}) − \sum^{4}_{j = 1} \phi(T_{j} \circ \textbf{x}) \bigg\rvert^{2}\bigg\}$$
- where in our experiments the constant scalar $M = 10$.
- Figure 3
    - <img src="https://user-images.githubusercontent.com/67457712/234450015-2a5eefe6-fcbd-42e0-ae89-821e8dd84fc6.png" width="450">
## Architecture
- ***In principle, the choice of the architecture is arbitrary.*** For ease of comparison with state-of-the-art methods when transferring to classification and detection tasks, we adopt the AlexNet architecture [20] as commonly done in other self-supervised learning methods.
- We use the first 5 convolutional layers from AlexNet followed by three fully connected layers ($(3 \times 3 \times 256)\times 4096$, $4096 \times 4096$, and $4096 \times 1000$), and ReLU units. ***Note that*** $1000$ ***is the number of elements that we want to count. We use ReLU in the end since we want the counting vector to be all positive. Our input is*** $114 \times 114$ ***pixels to handle smaller tiles.*** Because all the features are the same, training with the loss function in equation 4 is equivalent to training a 6-way siamese network.
## Experiments

We
call the activation of the last layer of our network, on which
the loss (4) is defined, the counting vector. We evaluate
whether each unit in the counting vector is counting some
visual primitive or not. Our model is based on AlexNet [20]
in all experiments. In our tables we use boldface for the top
performer and underline the second top performer. 

We begin with a learning rate of 10−4
and drop it by a fac-
tor of 0.9 every 10K iterations. An important step is to nor-
malize the input by subtracting the mean intensity value and
dividing the zero-mean images by their standard deviation. 

Transfer Learning Evaluation
We evaluate our learned representation on the detec-
tion, classification, and segmentation tasks on the PASCAL
dataset as well as the classification task on the ImageNet
dataset. 

Fine-tuning on PASCAL
In this set of experiments, we fine-tune our network on
the PASCAL VOC 2007 and VOC 2012 datasets, which
are standard benchmarks for representation learning. 

Notice that while classification and detection are evaluated on VOC
2007, segmentation is evaluated on VOC 2012. 

Table 1: Evaluation of transfer learning on PASCAL.
Classification and detection are evaluated on PASCAL VOC
2007 in the frameworks introduced in [19] and [11] respec-
tively. Both tasks are evaluated using mean average pre-
cision (mAP) as a performance measure. Segmentation is
evaluated on PASCAL VOC 2012 in the framework of [26],
which reports mean intersection over union (mIoU). (*) de-
notes the use of the data initialization method [19].
## References
- [9] [Unsupervised Visual Representation Learning by Context Prediction](https://arxiv.org/pdf/1505.05192.pdf)
- [21] [Learning Representations for Automatic Colorization](https://arxiv.org/pdf/1603.06668.pdf)
- [22] [Colorization as a Proxy Task for Visual Understanding](https://arxiv.org/pdf/1703.04044.pdf)
- [29] [Unsupervised Learning of Visual Representations by Solving Jigsaw Puzzles](https://arxiv.org/pdf/1603.09246.pdf)
- [33] [Context Encoders: Feature Learning by Inpainting](https://arxiv.org/pdf/1604.07379.pdf)
- [35] [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/pdf/1511.06434.pdf)
- [43] [Colorful Image Colorization](https://arxiv.org/pdf/1603.08511.pdf)
- [44] [Split-Brain Autoencoders: Unsupervised Learning by Cross-Channel Prediction](https://arxiv.org/pdf/1611.09842.pdf)