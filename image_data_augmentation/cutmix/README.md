# Paper Summary
- [CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features](https://arxiv.org/pdf/1905.04899.pdf)
<!-- ## Introduction -->
<!-- - We therefore propose the CutMix augmen- tation strategy: patches are cut and pasted among train- ing images where the ground truth labels are also mixed proportionally to the area of the patches. By making ef- ficient use of training pixels and retaining the regulariza- tion effect of regional dropout, CutMix-trained ImageNet classifier, when used as a pretrained model, re- sults in consistent performance gains in Pascal detection and MS-COCO image captioning benchmarks. -->
## Related Works
- ***To prevent a CNN from focusing too much on a small set of intermediate activations or on a small region on input images, random feature removal regularizations have been proposed. Examples include dropout [34] for randomly dropping hidden activations and regional dropout [3] [8] [33] [51] for erasing random regions on the input. Researchers have shown that the feature removal strategies improve generalization and localization by letting a model attend not only to the most discriminative parts of objects, but rather to the entire object region [8] [33].***
- While regional dropout strategies have shown improvements of classification and localization performances to a certain degree, ***deleted regions are usually zeroed-out [3] [33] or filled with random noise [51], greatly reducing the proportion of informative pixels on training images. We recognize this as a severe conceptual limitation.***
## Methodology
- ***The added patches further enhance localization ability by requiring the model to identify the object from a partial view.***
- CutMix incurs only negligible additional cost for training.

- Mixup samples suffer from the fact that they are locally ambiguous and unnatural, and therefore confuses the model, especially for localization. Recently, Mixup variants [42, 35, 10, 40] have been proposed; they perform feature-level interpola- tion and other types of transformations.
- We use vanilla ResNet-50 model1 for obtaining the CAMs to clearly see the effect of augmentation method only.
- The CAM for Mixup, as a re- sult, shows that the model is confused when choosing cues for recognition. We hypothesize that such confusion leads to its suboptimal performance in classification and localiza- tion,
- We observe, first of all, that CutMix achieves lower val- idation errors than the baseline at the end of training. At epoch 150 when the learning rates are reduced, the base- lines suffer from overfitting with increasing validation error. CutMix, on the other hand, shows a steady decrease in val- idation error; diverse training samples reduce overfitting.
- For fair comparison, we use the stan- dard augmentation setting for ImageNet dataset such as re- sizing, cropping, and flipping,
- we have trained all the models for 300 epochs with initial learning rate 0.1 decayed by factor 0.1 at epochs 75, 150, and 225. The batch size is set to 256. We report the best performances of CutMix and other baselines during training.
- We set the dropping rate of residual blocks to 0.25 for the best performance of Stochastic Depth [17]. The mask size for Cutout [3] is set to 112×112 and the location for dropping out is uniformly sampled. The performance of DropBlock [8] is from the original paper and the difference from our setting is the training epochs which is set to 270. Manifold Mixup [42] applies Mixup operation on the ran- domly chosen internal feature map. We have tried α = 0.5 and 1.0 for Mixup and Manifold Mixup and have chosen 1.0 which has shown better performances. It is also possible to extend CutMix to feature-level augmentation (Feature Cut- Mix). Feature CutMix applies CutMix at a randomly chosen layer per minibatch as Manifold Mixup does.
- On the feature level as well, we find CutMix preferable to Mixup, with top-1 errors 21.78% and 22.50%, respectively.
- We have also compared improvements due to CutMix versus architectural improvements (e.g. greater depth or additional modules). We observe that CutMix improves the perfor- mance by +2.28% while increased depth (ResNet-50 → ResNet-152) boosts +1.99% and SE [15] and GE [14] boosts +1.56% and +1.80%, respectively. Note that un- like above architectural boosts improvements due to Cut- Mix come at little or memory or computational time.
- CIFAR Classification We set mini-batch size to 64 and training epochs to 300. The learning rate was initially set to 0.25 and decayed by the factor of 0.1 at 150 and 225 epoch
- All experiments were conducted three times and the averaged best performances during training are reported.
- N, C, and K denote the size of minibatch, channel size of input image, and the number of classes. First, CutMix shuffles the order of the minibatch input and target along the first axis of the tensors. And the lambda and the cropping region (x1,x2,y1,y2) are sampled. Then, we mix the input and input s by replacing the crop- ping region of input to the region of input s. The target label is also mixed by interpolating method.

## References
- [3] [Improved Regularization of Convolutional Neural Networks with Cutout](https://arxiv.org/pdf/1708.04552.pdf)
- [8] [DropBlock: A regularization method for convolutional networks](https://arxiv.org/pdf/1810.12890.pdf)
- [33] [Hide-and-Seek: Forcing a Network to be Meticulous for Weakly-supervised Object and Action Localization](https://arxiv.org/pdf/1704.04232.pdf)
- [34] [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](https://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf)
- [51] [Random Erasing Data Augmentation](https://arxiv.org/pdf/1708.04896.pdf)