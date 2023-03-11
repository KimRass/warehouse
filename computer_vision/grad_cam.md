# Paper Summary
- [Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization](https://arxiv.org/pdf/1610.02391.pdf)
- ***Our approach – Gradient-weighted Class Activation Mapping (Grad-CAM), uses the gradients of any target concept (say 'dog' in a classification network or a sequence of words in captioning network) flowing into the final convolutional layer to produce a coarse localization map highlighting the important regions in the image for predicting the concept. Unlike previous approaches, Grad-CAM is applicable to a wide variety of CNN model-families, all without architectural changes or re-training.***
- ***In the context of image classification models, our visualizations lend insights into failure modes of these models (showing that seemingly unreasonable predictions have reasonable explanations) and help achieve model generalization by identifying dataset bias.***
- ***For image captioning and VQA, our visualizations show that even non-attention based models learn to localize discriminative regions of input image.***
- Guided Grad-CAM explanations also help untrained users successfully discern a 'stronger' network from a 'weaker' one, even when both make identical predictions.
- ***Another relevant line of work is weakly-supervised localization in the context of CNNs, where the task is to localize objects in images using holistic image class labels only.***
- Figure 1.
    - <img src="https://i.imgur.com/7xeTdaK.png" width="800">
    - Fig. 1 shows outputs from a number of visualizations for the 'tiger cat' class (top) and 'boxer' (dog) class (bottom).
    - (a) Original image with a cat and a dog.
    - (b) Guided Backpropagation [53]: highlights all contributing features.
    - (c, f) Grad-CAM (Ours): localizes class-discriminative regions.
    - (d) Combining (b) and (c) gives Guided Grad-CAM, which gives high-resolution class-discriminative visualizations.
    - (f, l) are Grad-CAM visualizations for ResNet-18 layer. Note that in (e, k), blue corresponds to evidence for the class.
    - ***Localization approaches like CAM or our proposed method Grad-CAM, are highly class-discriminative (the 'cat' explanation exclusively highlights the 'cat' regions but not 'dog' regions in Fig. 1c, and vice versa in Fig. 1i).***
    - ***We show that it is possible to fuse existing pixel-space gradient visualizations with Grad-CAM to create Guided Grad-CAM visualizations that are both high-resolution and class-discriminative. As a result, important regions of the image which correspond to any decision of interest are visualized in high-resolution detail even if the image contains evidence for multiple possible concepts, as shown in Figures 1d and 1j.*** When visualized for 'tiger cat', Guided Grad-CAM not only highlights the cat regions, but also highlights the stripes on the cat, which is important for predicting that particular variety of cat.
## Related Works
- Guided Backpropagation
    - ***Pixel-space gradient visualizations such as Guided Backpropagation [53] and Deconvolution [57] are high-resolution and highlight fine-grained details in the image, but are not class-discriminative (Fig. 1b and Fig. 1h are very similar).***
- Class Activation Mapping (CAM)
    - Zhou et al. [59] recently proposed a technique called CAM for identifying discriminative regions used by a restricted class of image classification CNNs which do not contain any fully-connected layers. This approach modifies image classification CNN architectures replacing fully-connected layers with convolutional layers and global average pooling, thus achieving class-specific feature maps. A drawback of CAM is that it requires feature maps to directly precede softmax layers, so it is only applicable to a particular kind of CNN architectures performing global average pooling over convolutional maps immediately prior to prediction (i.e. conv feature maps → global average pooling → softmax layer).
    - For a fully-convolutional architecture, CAM is a special case of Grad-CAM.
## Methodology
- No Architecture Modification
    - ***We make existing state-of-the-art deep models interpretable without altering their architecture, thus avoiding the interpretability vs. accuracy trade-off. Our approach is a generalization of CAM [59] and is applicable to a significantly broader range of CNN model families.***
- ***A number of previous works have asserted that deeper representations in a CNN capture higher-level visual constructs. Convolutional layers naturally retain spatial information which is lost in fully-connected layers, so we can expect the last convolutional layers to have the best compromise between high-level semantics and detailed spatial information.*** The neurons in these layers look for semantic class-specific information in the image (say object parts). ***Grad-CAM uses the gradient information flowing into the last convolutional layer of the CNN to assign importance values to each neuron for a particular decision of interest. Although our technique is fairly general in that it can be used to explain activations in any layer of a deep network, in this work, we focus on explaining output layer decisions only. We find that Grad-CAM maps become progressively worse as we move to earlier convolutional layers as they have smaller receptive fields and only focus on less semantic local features.***
- The gradients are global-average-pooled over the width and height dimension (indexed by $i$ and $j$ respectively) to obtain the neuron importance weights $\alpha^{c}_{k}$.
$$\alpha^{c}_{k} = \frac{1}{Z} \sum_{i} \sum_{j} \frac{\partial{y_{c}}}{\partial{A^{k}_{ij}}}$$
- $y^{c}$: Score for class $c$ before the softmax. In general, $y^{c}$ need not be the class score produced by an image classification CNN. It could be any differentiable activation including words from a caption or answer to a question.
- $k$: Feature map.
- $A^{k}$: Feature map activations.
- We perform a weighted combination of forward activation maps, and follow it by a ReLU to obtain class-dicriminative localization map,
$$L^{c}_{Grad-CAM} = \text{ReLU}\bigg(\sum_{k}\alpha^{k}A^{k}\bigg)$$
- ***Notice that this results in a coarse heatmap of the same size as the convolutional feature maps (14 × 14 in the case of last convolutional layers of VGG and AlexNet networks). We apply a ReLU to the linear combination of maps because we are only interested in the features that have a positive influence on the class of interest, i.e. pixels whose intensity should be increased in order to increase $y^{c}$. Negative pixels are likely to belong to other categories in the image. As expected, without this ReLU, localization maps sometimes highlight more than just the desired class and perform worse at localization.***
## References
- [53] [Striving for Simplicity: The All Convolutional Net](https://arxiv.org/pdf/1412.6806.pdf)
- [57] [Visualizing and Understanding Convolutional Networks](https://arxiv.org/pdf/1311.2901.pdf)
- [59] [Learning Deep Features for Discriminative Localization](https://arxiv.org/pdf/1512.04150.pdf)
