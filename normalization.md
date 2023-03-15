# Batch Normalization
- Reference: https://towardsdatascience.com/different-normalization-layers-in-deep-learning-1a7214ff71d6
- ***BN calculates the batch statistics(Mini-batch mean and variance) in every training iteration, therefore it requires larger batch sizes while training so that it can effectively approximate the population mean and variance from the mini-batch.*** This makes BN harder to train networks for application such as object detection, semantic segmentation, etc because they generally work with high input resolution(often as big as 1024x 2048) and training with larger batch sizes is not computationally feasible.
- ***BN does not work well with RNNs. The problem is RNNs have a recurrent connection to previous timestamps and would require a separate $\beta$ and $\gamma$ for each timestep in the BN layer which instead adds additional complexity and makes it harder to use BN with RNNs.***
- ***Different training and test calculation: During test(or inference) time, the BN layer doesn’t calculate the mean and variance from the test data mini-batch but uses the fixed mean and variance calculated from the training data.*** This requires cautious while using BN and introduces additional complexity. In pytorch `model.eval()` makes sure to set the model in evaluation model and hence the BN layer leverages this to use fixed mean and variance from pre-calculated from training data. which normalizes the activations along the feature direction instead of mini-batch direction. This overcomes the cons of BN by removing the dependency on batches and makes it easier to apply for RNNs as well.

# Layer Normalization
- ***Unlike batch normalization, layer normalization does not impose any constraint on the size of the mini-batch and it can be used in the pure online regime with batch size 1***.
- Reference: https://github.com/CyberZHG/torch-layer-normalization/blob/89f405b60f53f85da6f03fe685c190ef394ce50c/torch_layer_normalization/layer_normalization.py#L8
- [Layer Normalization](https://arxiv.org/pdf/1607.06450.pdf)
```python

```