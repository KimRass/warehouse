# Paper Summary
- [Colorful Image Colorization](https://arxiv.org/pdf/1603.08511.pdf)
## Related Works
- Previous works have trained convolutional neural networks (CNNs) to predict color on large datasets. However, ***the results from these previous attempts tend to look desaturated.*** One explanation is that use loss functions that encourage conservative predictions. These losses are inherited from standard regression problems, where ***the goal is to minimize Euclidean error between an estimate and the ground truth.***
## Introduction
- Our goal is not necessarily to recover the actual ground truth color, ***but rather to produce a plausible colorization that could potentially fool a human observer.*** Therefore, our task becomes much more achievable: ***to model enough of the statistical dependencies between the semantics and the textures of grayscale images and their color versions in order to produce visually compelling results.***
- Given the lightness channel $L$, our system predicts the corresponding $a$ and $b$ color channels of the image in the CIELAB colorspace. ***Predicting color has the nice property that training data is practically free: any color photo can be used as a training example, simply by taking the image’s L\* channel as input and its a\*b\* channels as the supervisory signal.***
## Architecture
- Figure 2
    - <img src="https://user-images.githubusercontent.com/105417680/228839815-080cba72-b815-4b9a-95d9-8c3c9135f148.png" width="800">
- Table 4
    - <img src="https://user-images.githubusercontent.com/105417680/228840531-b60c8aa5-176b-47cc-9995-31be345d6357.png" width="400">
    - 'X': Spatial resolution of output
    - 'C': Number of channels of output
    - 'S': Computation stride, values greater than 1 indicate downsampling following convolution, values less than 1 indicate upsampling preceding convolution
    - 'D': Kernel dilation
    <!-- - 'Sa': Accumulated stride across all preceding layers (product over all strides in previous layers)
    - 'De' Effective dilation of the layer with respect to the input (layer dilation times accumulated stride) -->
    - 'BN' Whether BatchNorm layer was used after layer
    - 'L': Whether a 1 x 1 conv and cross-entropy loss layer was imposed
## Training
### Loss
- ***Color prediction is inherently multimodal – many objects can take on several plausible colorizations. For example, an apple is typically red, green, or yellow, but unlikely to be blue or orange.***
- ***To appropriately model the multimodal nature of the problem, we predict a distribution of possible colors for each pixel.***
- ***We re-weight the loss at training time to emphasize rare colors.*** This encourages our model to exploit the full diversity of the large-scale data on which it is trained.
- Given an input lightness channel $X \in \mathbb{R}^{H \times W \times 1}$, our objective is to learn a mapping $\hat{Y} = F(X)$ to the two associated color channels $Y \in \mathbb{R}^{H \times W \times 2}$, where $H$, $W$ are image dimensions. (We denote predictions with a $\hat{\cdot}$ symbol and ground truth without.)
- We perform this task in CIELAB color space. Because distances in this space model perceptual distance, a natural objective function is the Euclidean loss $L_{2}$ between predicted and ground truth colors. However, ***this loss is not robust to the inherent ambiguity and multimodal nature of the colorization problem. If an object can take on a set of distinct a\*b\* values, the optimal solution to the Euclidean loss will be the mean of the set. In color prediction, this averaging effect favors grayish, desaturated results.*** Additionally, if the set of plausible colorizations is non-convex, the solution will in fact be out of the set, giving implausible results.
- Instead, we treat the problem as multinomial classification. We quantize the a\*b\* output space into bins with grid size 10 and keep the $Q = 313$ values.
- For a given input X, we learn a mapping Zb = G(X) to a probability distribution over possible colors Zb ∈ [0, 1]H×W×Q, where Q is the number of quantized ab values. To compare predicted Zb against ground truth, we define function Z = H −1 gt (Y), which converts ground truth color Y to vector Z, using a soft-encoding scheme2 . We then use multinomial cross entropy loss Lcl(·, ·), defined as: Lcl(Zb, Z) = − X h,w v(Zh,w) X q Zh,w,q log(Zbh,w,q) (2) where v(·) is a weighting term that can be used to rebalance the loss based on color-class rarity, as defined in Section 2.2 below. Finally, we map probability distribution Zb to color values Yb with function Yb = H(Zb), which will be further discussed in Section 2.3. 2.2 Class rebalancing The distribution of ab values in natural images is strongly biased towards val- ues with low ab values, due to the appearance of backgrounds such as clouds, pavement, dirt, and walls. Figure 3(b) shows the empirical distribution of pix- els in ab space, gathered from 1.3M training images in ImageNet [28]. Observe that the number of pixels in natural images at desaturated values are orders of magnitude higher than for saturated values. Without accounting for this, the loss function is dominated by desaturated ab values. We account for the class- imbalance problem by reweighting the loss of each pixel at train time based on the pixel color rarity. This is asymptotically equivalent to the typical approach of resampling the training space [32]. Each pixel is weighed by factor w ∈ RQ, based on its closest ab bin.
## Evaluation
<!-- - We set up a "colorization Turing test" in which we show participants real and synthesized colors for an image, and ask them to identify the fake. In this quite difficult paradigm, we are able to fool participants on 32% of the instances (ground truth colorizations would achieve 50% on this metric), signif- icantly higher than prior work [2]. -->





- 2 Each ground truth value Yh,w can be encoded as a 1-hot vector Zh,w by searching for the nearest quantized ab bin. However, we found that soft-encoding worked well for training, and allowed the network to quickly learn the relationship between elements in the output space [31]. We find the 5-nearest neighbors to Yh,w in the output space and weight them proportionally to their distance from the ground truth using a Gaussian kernel with σ = 5.

- Our classification loss with re- balancing produces more accurate and vibrant results than a regression loss or a clas- sification loss without rebalancing.

- We train our network on the 1.3M images from the ImageNet training set [28], validate on the first 10k images in the ImageNet validation set, and test on a separate 10k images in the validation set,

- Ours (L2) Our network trained from scratch, with L2 regression loss, de- scribed in Equation 1, following the same training protocol. 4. Ours (L2, ft) Our network trained with L2 regression loss, fine-tuned from our full classification with rebalancing network. 5. Larsson et al. [23] A CNN method that also appears in these proceedings. 6. Dahl [2] A previous model using a Laplacian pyramid on VGG features, trained with L2 regression loss. 7. Gray Colors every pixel gray, with (a, b) = 0. 8. Random Copies the colors from a random image from the training set.

- 1. Perceptual realism (AMT): For many applications, such as those in graphics, the ultimate test of colorization is how compelling the colors look to a human observer. To test this, we ran a real vs. fake two-alternative forced choice experiment on Amazon Mechanical Turk (AMT). Participants in the experiment were shown a series of pairs of images. Each pair consisted of a color photo next to a re-colorized version, produced by either our algorithm or a baseline. Par- ticipants were asked to click on the photo they believed contained fake colors generated by a computer program. Individual images of resolution 256×256 were shown for one second each, and after each pair, participants were given unlim- ited time to respond. Each experimental session consisted of 10 practice trials

- (excluded from subsequent analysis), followed by 40 test pairs. On the practice trials, participants were given feedback as to whether or not their answer was correct. No feedback was given during the 40 test pairs. Each session tested only a single algorithm at a time, and participants were only allowed to com- plete at most one session. A total of 40 participants evaluated each algorithm. To ensure that all algorithms were tested in equivalent conditions (i.e. time of day, demographics, etc.), all experiment sessions were posted simultaneously and distributed to Turkers in an i.i.d. fashion. To check that participants were competent at this task, 10% of the trials pitted the ground truth image against the Random baseline described above. Participants successfully identified these random colorizations as fake 87% of the time, indicating that they understood the task and were paying attention.

- Fig. 6. Images sorted by how often AMT participants chose our algorithm’s colorization over the ground truth. In all pairs to the left of the dotted line, participants believed our colorizations to be more real than the ground truth on ≥ 50% of the trials. In some cases, this may be due to poor white balancing in the ground truth image, corrected by our algorithm, which predicts a more prototypical appearance. Right of the dotted line are examples where participants were never fooled.