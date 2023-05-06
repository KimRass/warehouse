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
- Least effort bias
    - Figure 3
        - <img src="https://user-images.githubusercontent.com/67457712/234450015-2a5eefe6-fcbd-42e0-ae89-821e8dd84fc6.png" width="550">
        - A bias of the system is that ***it can easily satisfy the constraint (3) by learning to count as few visual primitives as possible. Thus, many entries of the feature mapping may collapse to zero.*** This effect is observed in the final trained network. In figure 3, we show the average of features computed over the ImageNet validation set. ***There are only 30 and 44 non zero entries out of 1000 after training on ImageNet and on COCO respectively.*** Despite the sparsity of the features, our transfer learning experiments show that the features in the hidden layers (conv1-conv5) perform very well on several benchmarks. ***In equation 4, the contrastive term limits the effects of the least effort bias. Indeed, features that count very few visual primitives cannot differentiate much the content across different images. Therefore, the contrastive term will introduce a tradeoff that will push features towards counting as many primitives as is needed to differentiate images from each other.***
## Architecture
- ***In principle, the choice of the architecture is arbitrary.*** For ease of comparison with state-of-the-art methods when transferring to classification and detection tasks, we adopt the AlexNet architecture [20] as commonly done in other self-supervised learning methods.
- We use the first 5 convolutional layers from AlexNet followed by three fully connected layers ($(3 \times 3 \times 256)\times 4096$, $4096 \times 4096$, and $4096 \times 1000$), and ReLU units. ***Note that*** $1000$ ***is the number of elements that we want to count. We use ReLU in the end since we want the counting vector to be all positive. Our input is*** $114 \times 114$ ***pixels to handle smaller tiles.*** Because all the features are the same, training with the loss function in equation 4 is equivalent to training a 6-way siamese network.
## Ablation Studies
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
- Table 4: Performance on the detection task on PASCAL VOC 2007 under different training scenarios
    - <img src="https://user-images.githubusercontent.com/67457712/235175524-43cd7d35-55cf-444a-ac36-1e60643194e7.png" width="450">
    - Counting vector length
        - The first row and last rows: Shows the impact of the counting vector length. As discussed earlier on, the network tends to generate sparse counting features. We train the network on ImageNet with only 20 elements in the counting vector. ***This leads to a small drop in the performance, thus showing little sensitivity with respect to feature length.***
    - Dataset size
        - The first row and second rows: We also train the network with a smaller set of training images. The results show that ***our method is sensitive to the size of the training set.*** This shows that the counting task is non-trivial and ***requires a large training dataset.***
    - An important part of the design of the learning procedure is the identification of trivial solutions, i.e., ***solutions that would not result in a useful representation and that the neural network could converge to.*** By identifying such trivial learning scenarios, we can provide suitable countermeasures. We now discuss possible shortcuts that the network could use to solve the counting task and also the techniques that we use to avoid them.
    - Color histograms
        - A first potential problem is that the neural network learns trivial features such as low-level texture statistics histograms. For example, a special case is color histograms. This representation is undesirable because it would be semantically agnostic (or very weak) and therefore we would not expect it to transfer well to classification and detection. In general, these histograms would not satisfy equation 2. However, ***if the neural network could tell tiles apart from downsampled images, then it could apply a customized scaling factor to the histograms in the two cases and satisfy equation 2. In other words, ***the network might learn the following degenerate feature
        - Equation 5
            $$
            \begin{equation}
                \phi(z) =
                    \begin{cases}
                    \frac{1}{4}hist(\textbf{z}) & \text{if \textbf{z} is a tile}\\
                    hist(\textbf{z}) & \text{if \textbf{z} is downsampled}
                    \end{cases}       
                \end{equation}
            $$
        - Notice that this feature would satisfy the first term in equation 4. The second (contrastive) term would also be easily satisfied since different images have typically different low-level texture histograms. We discuss below scenarios when this might happen and present our solutions towards reducing the likelihood of trivial learning.
        - The network recognizes the downsampling style.
            - During training, we randomly crop a 224 × 224 region from a 256 × 256 image. Next, we downsample the whole im- age by a factor of 2. The downsampling style, e.g., bilinear, bicubic, and Lanczos, may leave artifacts in images that the network may learn to recognize. To make the identifica- tion of the downsampling method difficult, at each stochas- tic gradient descent iteration, we randomly pick either the bicubic, bilinear, lanczos, or the area method as defined in OpenCV [16]. As shown in Table 4, the randomization of different downsampling methods significantly improves the detection performance by at least 2.2%. In Table 5, we perform another experiment that clearly shows that network learns the downsampling style. We train our network by using only one downsampling method. Then, we test the network on the pretext task by using only one (possibly different) method. If the network has learned to detect the downsampling method, then it will perform poorly at test time when using a different one. As an error metric, we use the first term in the loss function normalized by the average of the norm of the feature vector. More pre- cisely, the error when the network is trained with the i-th downsampling style and tested on the j-th one is eij = P x P4 p=1 φ i
    - The network recognizes chromatic aberration
        - The 4th, 5th and last rows: The presence of chromatic aberration and its undesirable effects on learning have been pointed out by [9]. ***Chromatic aberration is a relative shift between the color channels that increases in the outward radial direction. Hence, our network can use this property to tell tiles apart from the dowsampled images. In fact, tiles will have a strongly diagonal chromatic aberration, while the downsampled image will have a radial aberration***
        - ***We already reduce its effect by choosing the central region in the very first cropping preprocessing. To further reduce its effect, we train the network with both color and grayscale images (obtained by replicating the average color across all 3 channels). In training, we randomly choose color images 33% of the time and grayscale images 67% of the time. This choice is consistent across all the terms in the loss function (i.e., all tiles and downsampled images are either colored or grayscale).***
        - ***While this choice does not completely solve the issue, it does improve the performance of the model. We find that completely eliminating the color from images leads to a loss in performance in transfer learning.***
- Table 5
    - <img src="https://user-images.githubusercontent.com/67457712/235180136-f273d7d9-a7a1-41db-91af-e70be14637a1.png" width="450">


We use visualization and nearest neighbor search to see
what visual primitives our trained network counts. Ideally,
these visual primitives should capture high-level concepts like objects or object parts rather than low-level concepts
like edges and corners. In fact, detecting simple corners
will not go a long way in semantic scene understanding. To
avoid dataset bias, we train our model on ImageNet (with
no labeles) and show the results on COCO dataset. 

Quantitative Analysis
We illustrate quantitatively the relation between the mag-
nitude of the counting vector and the number of objects.
Rather than counting exactly the number of specific ob-
jects, we introduce a simple method to rank images based
on how many objects they contain. The method is based on
cropping an image with larger and larger regions which are
then rescaled to the same size through downsampling (see
Fig. 5). We build two sets of 100 images each. We assign images yielding the highest and lowest feature magnitude
into two different sets. We randomly crop 10 regions with
an area between 50%−95% of each image and compute the
corresponding counting vector. The mean and the standard
deviation of the counting vector magnitude of the cropped
images for each set is shown in Fig 6. We observe that
our feature does not count low-level texture, and is instead
more sensitive to composite images. A better understanding
of this observation needs futher investigation.

- Qualitative Analysis
    - Figure 4
        - <img src="https://user-images.githubusercontent.com/67457712/235311148-99782898-190e-4eac-86e6-871e942d8674.png" width="700">
        - Activating/Ignored images. In Fig 4, we show blocks of
        16 images ranked based on the magnitude of the count-
        ing vector. We observe that images with the lowest feature
        norms are textures without any high-level visual primitives.
        In contrast, images with the highest feature response mostly
        contain multiple object instances or a large object. For this
        experiment we use the validation or the test set of the dataset
        that the network has been trained on, so the network has not
        seen these images during training.
- Nearest neighbor search
    - Figre 7
        - <img src="https://user-images.githubusercontent.com/67457712/235311068-bc42fb3b-fbf9-4126-bccd-c2858a9775f9.png" width="900">
        - To qualitatively evaluate our
        learned representation, for some validation images, we vi-
        sualize their nearest neighbors in the training set in Fig. 7.
        Given a query image, the retrieval is obtained as a rank-
        ing of the Euclidean distance between the counting vector
        of the query image and the counting vector of images in
        the dataset. Smaller values indicate higher affinity. Fig. 7
        shows that the retrieved results share a similar scene outline
        and are semantically related to the query images. Note that
        we perform retrieval in the counting space, which is the last
        layer of our network. This is different from the analogous
        experiment in [19] which performs the retrieval in the in-
        termediate layers. This result can be seen as an evidence
        that our initial hypothesis, that the counting vectors capture
        high level visual primitives, was true.
        - Figure 7: Nearest neighbor retrievals. Left: COCO retrievals. Right: ImageNet retrievals. In both datasets, the leftmost
    column (with a red border) shows the queries and the other columns show the top matching images sorted with increasing
    Euclidean distance in our counting feature space from left to right. On the bottom 3 rows, we show the failure retrieval cases.
    Note that the matches share a similar content and scene outline. 
- Neuron activations
    - Figure 8
        - <img src="https://user-images.githubusercontent.com/67457712/235311077-cbbf27d6-a762-4232-af86-e7e9d57f9689.png" width="900">
        - To visualize what each single count-
        ing neuron (i.e., feature element) has learned, we rank images not seen during training based on the magnitude of
        their neuron responses. We do this experiment on the vali-
        dation set of ImageNet and the test set of COCO. In Fig. 8,
        we show the top 8 most activating images for 4 neurons out
        of 30 active ones on ImageNet and out of 44 active ones on
        COCO. We observe that these neurons seem to cluster im-
        ages that share the same scene layout and general content.
        - Figure 8: Blocks of the 8 most activating images for 4 neurons of our network trained on ImageNet (top row) and COCO
    (bottom row). The counting neurons are sensitive to semantically similar images. Interestingly, dominant concepts in each
    dataset, e.g., dogs in ImageNet and persons playing baseball in COCO, emerge in our counting vector.

## References
- [9] [Unsupervised Visual Representation Learning by Context Prediction](https://arxiv.org/pdf/1505.05192.pdf)
- [21] [Learning Representations for Automatic Colorization](https://arxiv.org/pdf/1603.06668.pdf)
- [22] [Colorization as a Proxy Task for Visual Understanding](https://arxiv.org/pdf/1703.04044.pdf)
- [29] [Unsupervised Learning of Visual Representations by Solving Jigsaw Puzzles](https://arxiv.org/pdf/1603.09246.pdf)
- [33] [Context Encoders: Feature Learning by Inpainting, 2016](https://arxiv.org/pdf/1604.07379.pdf)
- [35] [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/pdf/1511.06434.pdf)
- [43] [Colorful Image Colorization](https://arxiv.org/pdf/1603.08511.pdf)
- [44] [Split-Brain Autoencoders: Unsupervised Learning by Cross-Channel Prediction](https://arxiv.org/pdf/1611.09842.pdf)