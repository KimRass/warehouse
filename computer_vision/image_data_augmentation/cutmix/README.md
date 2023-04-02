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
## References
- [3] [Improved Regularization of Convolutional Neural Networks with Cutout](https://arxiv.org/pdf/1708.04552.pdf)
- [8] [DropBlock: A regularization method for convolutional networks](https://arxiv.org/pdf/1810.12890.pdf)
- [33] [Hide-and-Seek: Forcing a Network to be Meticulous for Weakly-supervised Object and Action Localization](https://arxiv.org/pdf/1704.04232.pdf)
- [34] [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](https://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf)
- [51] [Random Erasing Data Augmentation](https://arxiv.org/pdf/1708.04896.pdf)