# Paper Reading
- [Hide-and-Seek: Forcing a Network to be Meticulous for Weakly-supervised Object and Action Localization](https://arxiv.org/pdf/1704.04232.pdf)
## Related Works
- However, due to intra-category variations or relying only on a classification objective, ***these methods often fail to identify the entire extent of the object and instead localize only the most discriminative part.***
- Weakly-supervied object localization
    - Fully-supervised convolutional networks (CNNs) require expensive human annotations for training (e.g. bounding box for object localization). To alleviate expensive annotation costs, weakly-supervised approaches learn using cheaper labels, for example, image-level labels for predicting an object’s location [3] [8] [9] [32] [43] [61].
    - Most weakly-supervised object localization approaches mine discriminative features or patches in the data that frequently appear in one class and rarely in other classes [3] [7] [9] [13] [41] [42] [43] [55]. However, ***these approaches tend to focus only on the most discriminative parts, and thus fail to cover the entire spatial extent of an object.***
    - Recent work modify CNN architectures designed for image classification so that the convolutional layers learn to localize objects while performing image classification [32] [61]. Other network architectures have been designed for weakly-supervised object detection [4] [20] [24]. Although these methods have significantly improved the state-of-the-art, ***they still essentially rely on a classification objective and thus can fail to capture the full extent of an object if the less discriminative parts do not help improve classification performance.***
- Masking pixels or activations
    - ***For object localization, [1] [59] train a CNN for image classification and then localize the regions whose masking leads to a large drop in classification performance. Since these approaches mask out the image regions only during testing and not during training, the localized regions are limited to the highly-discriminative object parts.***
    -  [1] hides patches during testing but not during training. For [1], since the network has already learned to focus on the most discimirinative parts during training, it is essentially too late, and hiding patches during testing has no significant effect on localization performance.
    - Our work is closely related to the adversarial erasing method [56], which iteratively trains a sequence of models for weakly-supervised semantic segmentation. Each model identifies the relevant object parts conditioned on the previous iteration model’s output.
    - Dropout [44] and its variants [47] [49] are also related. There are two main differences: (1) these methods are designed to prevent overfitting while our work is designed to improve localization; and (2) in dropout, units in a layer are dropped randomly, while in our work, contiguous image regions or video frames are dropped.
## Methodology
- Figure 1: Main idea
    - <img src="https://user-images.githubusercontent.com/105417680/230265308-f5fc7dc8-1994-4f90-8d8e-f37516545aa6.png" width="350">
    - (Top row): A network tends to focus on the most discriminative parts of an image (e.g., face of the dog) for classification.
    - (Bottom row):
        - By hiding images patches randomly, we can force the network to focus on other relevant object parts in order to correctly classify the image as 'dog'.
        - If we randomly remove some patches from the image then there is a possibility that the dog’s face, which is the most discriminative, will not be visible to the model. In this case, ***the model must seek other relevant parts like the tail and legs in order to do well on the classification task.*** By randomly hiding different patches in each training epoch, the model sees different parts of the image and is forced to focus on multiple relevant parts of the object beyond just the most discriminative one. Importantly, ***we only apply this random hiding of patches during training and not during testing.***
- Figure 2
    - <img src="https://user-images.githubusercontent.com/105417680/230295312-ce83fdeb-bf98-4b01-bbed-4935310372ef.png" width="700">
    - (Left): ***For each training image, we divide it into a grid of*** $S \times S$ ***patches. Each patch is then randomly hidden with probability*** $p_{hide}$ ***and given as input to a CNN to learn image classification. The hidden patches change randomly across different epochs.***
    - (Right): ***We hide patches only during training. During testing, the full image—without any patches hidden—is given as input to the network.*** Since the network has learned to focus on multiple relevant parts during training, it is not necessary to hide any patches during testing.
- We make changes to the input image. ***The key idea is to hide patches from an image during training so that the model needs to seek the relevant object parts from what remains.*** We thus name our approach 'Hide-and-Seek'.
- For weakly-supervised object localization, we are given a set of images in which each image is labeled only with its category label. ***Our goal is to learn an object localizer that can predict both the category label as well as the bounding box for the object-of-interest in a new test image. In order to learn the object localizer, we train a CNN which simultaneously learns to localize the object while performing the image classification task.***
- ***The purpose of hiding patches is to show different parts of an object to the network while training it for the classification task. By hiding patches randomly, we can ensure that the most discriminative parts of an object are not always visible to the network, and thus force it to also focus on other relevant parts of the object.*** In this way, we can overcome the limitation of existing weakly-supervised methods that focus only on the most discriminative parts of an object.
- Same activation distributions
    - Since the full image is observed during testing, the data distribution will be different to that seen during training. We show that setting the hidden pixels’ value to be the data mean can allow the two distributions to match, and provide a theoretical justification.
    - Due to the discrepancy of hiding patches during training while not hiding patches during testing, the first convolutional layer activations during training versus testing will have different distributions. For a trained network to generalize well to new test data, the activation distributions should be roughly equal.
    - ***We resolve this issue by setting the RGB value of a hidden pixel to be equal to the mean RGB vector of the images over the entire dataset.***
    - ***This process is related to the scaling procedure in dropout [44], in which the outputs are scaled proportional to the drop rate during testing to match the expected output during training. In dropout, the outputs are dropped uniformly across the entire feature map, independently of spatial location.***
- Figure 4
    - <img src="https://user-images.githubusercontent.com/105417680/230535964-12c5d192-56ae-4c78-83cd-bb81799c1736.png" wdith="800">
    - (Left): The bounding box and CAM obtained by AlexNet-GAP
    - (Right): The bounding box and CAM obtained by our method
    - Our Hide-and-Seek approach localizes multiple relevant parts of an object whereas AlexNet-GAP mainly focuses only on the most discriminative parts.
- ***To obtain the binary fg/bg map, 20% and 30% of the max value of the CAM is chosen as the threshold for AlexNet-GAP and GoogLeNet GAP, respectively; the thresholds were chosen by observing a few qualitative results on training data.***
## Training
### Datasets
- We use ILSVRC 2016 to evaluate object localization accuracy.
## Evaluation
### Metrics
- We use three evaluation metrics to measure performance:
    - ('Top-1 Loc'; Top-1 localization accuracy): Fraction of images for which the predicted class with the highest probability is the same as the ground-truth class and the predicted bounding box for that class has more than 50% IoU with the ground-truth box. (Comment: Detection + Recognition)
    - ('GT-known Loc'; Localization accuracy with known ground-truth class): Fraction of images for which the predicted bounding box for the ground-truth class has more than 50% IoU with the ground-truth box. (Comment: Detection) As our approach is primarily designed to improve localization accuracy, we use this criterion to measure localization accuracy independent of classification performance.
    - ('Top-1 Clas'; We also use classification accuracy) to measure the impact of Hide-and-Seek on image classification performance. (Comment: Recognition)
## Experiments
- Table 1: Evaluation with different patch sizes for hiding
    - <img src="https://user-images.githubusercontent.com/105417680/230537429-560003cf-6116-470f-9722-077307bb9801.png" width="450">
    - ('AlexNet-GAP [61]'): ***Our baseline in which the network has seen the full image during training without any hidden patches.***
    - ('Alex-HaS-N'):
        - Our approach, in which patches of size $N \times N$ are hidden with $0.5$ probability during training.
        - We explored four different patch sizes $N = {16, 32, 44, 56}$, and each performs significantly better than AlexNet-GAP for both GT-known Loc and Top-1 Loc.
        - Our GoogLeNet-HaS-N models also outperforms GoogLeNet-GAP for all patch sizes. These results clearly show that hiding patches during training leads to better localization. Although ***our approach can lose some classification accuracy (Top-1 Clas) since it has never seen a complete image and thus may not have learned to relate certain parts,*** the huge boost in localization performance (which can be seen by comparing the GT-known Loc accuracies) makes up for any potential loss in classification.
        - 'AlexNet-HaS-Mixed': We also train a network with mixed patch sizes. ***During training, for each image in every epoch, the patch size*** $N$ ***to hide is chosen randomly from*** $16$***,*** $32$***,*** $44$ ***and*** $56$ ***as well as no hiding (full image). Since different sized patches are hidden, the network can learn complementary information about different parts of an object (e.g. small/large patches are more suitable to hide smaller/larger parts). Indeed, we achieve the best results for Top-1 Loc using AlexNet-HaS-Mixed.***
- Table 2: Comparison to state-of-the-art
    - <img src="https://user-images.githubusercontent.com/105417680/230537457-fdb552c8-59d6-47a6-8221-aac1c0936d40.png" width="400">
    - ('Ours-ensemble'): Since each patch size provides complementary information, we also create an ensemble model of different patch sizes. ***To produce the final localization for an image, we average the CAMs obtained using AlexNet-HaS-16, 32, 44, and 56, while for classification, we average the classification probabilities of all four models as well as the probability obtained using AlexNet-GAP.*** This ensemble model gives a boost of 5.24 % and 4.15% over AlexNet-GAP for GT-known Loc and Top-1 Loc, respectively. For a more fair comparison, we also combine the results of five independent AlexNet-GAPs to create an ensemble baseline. Ours-ensemble outperforms this strong baseline (AlexNet-GAP-ensemble) by 3.23% and 1.82% for GT-known Loc and Top-1 Loc, respectively.
- Table 4
    - <img src="https://user-images.githubusercontent.com/105417680/230537508-68e183f6-5aea-4d45-82f5-13309ac9c49c.png" width="350">
    - ***[61] showed that GAP is better than global max pooling (GMP) for object localization, since average pooling encourages the network to focus on all the discriminative parts. For max pooling, only the most discriminative parts need to contribute.*** But is global max pooling hopeless for localization? ***With our Hide-and-Seek, even with max pooling, the network is forced to focus on a different discriminative parts.***
    - ('AlexNet-GMP', 'AlexNet-Max-HaS' and 'AlexNet-Avg-HaS'): ***We see that max pooling is inferior to average poling (AlexNet-GAP) for the baselines. But with Hide-and-Seek, max pooling localization accuracy increases by a big margin and even slightly outperforms average pooling. The slight improvement is likely due to max pooling being more robust to noise.*** For this experiment, we use patch size 56.
- Table 5: Hide-and-Seek on convolutional feature maps
    - <img src="https://user-images.githubusercontent.com/105417680/230557749-99dfe3f6-9f90-448d-b687-ff5cb90768e4.png" width="400">
    - We apply our idea to convolutional layers. We divide the convolutional feature maps into a grid and hide each patch (and all of its corresponding channels) with $0.5$ probability. We hide patches of size $5$ ('AlexNet-HaS-conv1-5') and 11 ('AlexNet-HaS-conv1-11') in the conv1 feature map (which has size $55 \times 55 \times 96$). This leads to a big boost in performance compared to the baseline AlexNet-GAP. This shows that ***our idea of randomly hiding patches can be generalized to the convolutional layers.***
- Table 6: Probability of hiding
    - <img src="https://user-images.githubusercontent.com/105417680/230557797-23176fcc-4734-48d2-a52b-8da56e572494.png" width="380">
    - In all of the previous experiments, we hid patches with 50% probability. ***If we increase the probability then GT-known Loc remains almost the same while Top-1 Loc decreases a lot. This happens because the network sees fewer pixels when the hiding probability is high; as a result, classification accuracy reduces and Top-1 Loc drops.***
    - ***If we decrease the probability then GT-known Loc decreases but our Top-1 Loc improves. In this case, the network sees more pixels so its classification improves but since less parts are hidden, it will focus more on only the discriminative parts decreasing its localization ability.***
## References
- [1] [Self-taught Object Localization with Deep Networks](https://arxiv.org/pdf/1409.3964.pdf)
- [3] [Weakly Supervised Object Detection with Posterior Regularization](https://homepages.inf.ed.ac.uk/hbilen/assets/pdf/Bilen14b.pdf)
- [8] [Weakly Supervised Object Localization with Multi-fold Multiple Instance Learning](https://arxiv.org/pdf/1503.00949.pdf)
- [32] [Is object localization for free? - Weakly-supervised learning with convolutional neural networks]
- [43] [Weakly-supervised Discovery of Visual Pattern Configurations](https://arxiv.org/pdf/1406.6507.pdf)
- [49] [Regularization of Neural Networks using DropConnect](http://proceedings.mlr.press/v28/wan13.pdf)
- [56] [Object Region Mining with Adversarial Erasing: A Simple Classification to Semantic Segmentation Approach](https://arxiv.org/pdf/1703.08448.pdf)
- [59] [Visualizing and Understanding Convolutional Networks, 2014](https://arxiv.org/pdf/1311.2901.pdf)
- [61] [Learning Deep Features for Discriminative Localization]
