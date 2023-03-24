# Paper Summary
- [A Review of Generalized Zero-Shot Learning Methods](https://arxiv.org/pdf/2011.08641.pdf)
## Introduction
- DL models can only recognize samples belonging to the classes that have been seen during the training phase, and they are not able to handle samples from unseen classes.
- ***ZSL aims to train a model that can classify objects of unseen classes (target domain) via transferring knowledge obtained from other seen classes (source domain) with the help of semantic information.*** The semantic information embeds the names of both seen and unseen classes in high-dimensional vectors. Semantic information can be manually defined attribute vectors, automatically extracted word vectors, context-based embedding, or their combinations [17], [18]. ***This learning paradigm can be compared to a human when recognizing a new object by measuring the likelihoods between its descriptions and the previously learned notions [19].***
- Generalized Zero-Shot Learning (GZSL)
    - ***In conventional ZSL techniques, the test set only contains samples from the unseen classes, which is an unrealistic setting and it does not reflect the real-world recognition conditions. In practice, data samples of the seen classes are more common than those from the unseen ones, and it is important to recognize samples from both classes simultaneously rather than classifying only data samples of the unseen classes. This setting is called generalized zero-shot learning (GZSL) [20]. Indeed, GZSL is a pragmatic version of ZSL. The main motivation of GZSL is to imitate human recognition capabilities, which can recognize samples from both seen and unseen classes.***
## Related Works
- One-shot and few-shot learning techniques can learn from classes with a few learning samples. These techniques use the knowledge obtained from data samples of other classes and formulate a classification model for handling classes with few samples.
- While the open set recognition (OSR) [9] techniques can identify whether a test sample belongs to an unseen class, they are not able to predict an exact class label.
- Out-of-distribution [10] techniques attempt to identify test samples that are different from the training samples.
- ***However, none of the above-mentioned techniques can classify samples from unseen classes.*** In contrast, human do not need to learn all these categories in advance. As an example, a child can easily recognize zebra, if he/she has seen horses previously, and have the knowledge that a zebra looks like a horse with black and white strips.
- Zero-shot learning (ZSL) [12] [13] techniques offer a good solution to address such challenge.
## References
- [9] [Towards Open Set Recognition](https://vast.uccs.edu/~tboult/PAPERS/towards_openset_recognition.pdf)
- [10] [Generalized Out-of-Distribution Detection: A Survey](https://arxiv.org/pdf/2110.11334.pdf)
- [17] [Generalized Zero-shot Learning with Multi-source Semantic Embeddings for Scene Recognition](http://www.jdl.link/doc/2011/20210104_2020-Song-ACMMM.pdf)
- [18] [Pseudo Distribution on Unseen Classes for Generalized Zero Shot Learning](https://dro.dur.ac.uk/30892/1/30892.pdf?DDD4+)
- [19] [An embarrassingly simple approach to zero-shot learning](https://proceedings.mlr.press/v37/romera-paredes15.pdf)
