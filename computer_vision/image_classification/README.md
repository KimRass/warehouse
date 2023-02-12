## EfficientNet
- Source: https://keras.io/examples/vision/image_classification_efficientnet_fine_tuning/
- The input data should range [0, 255]. Normalization is included as part of the model.
- When the images are much smaller than the size of EfficientNet input, we can simply upsample the input images. It has been shown in Tan and Le, 2019 that transfer learning result is better for increased resolution even if input images remain small. 
- The first step to transfer learning is to freeze all layers and train only the top layers. For this step, a relatively large learning rate (1e-2) can be used. Note that validation accuracy and loss will usually be better than training accuracy and loss. This is because the regularization is strong, which only suppresses training-time metrics. 
- The second step is to unfreeze a number of layers and fit the model using smaller learning rate. In this example we show unfreezing all layers, but depending on specific dataset it may be desireble to only unfreeze a fraction of all layers.
- When the feature extraction with pretrained model works good enough, this step would give a very limited gain on validation accuracy. In our case we only see a small improvement, as ImageNet pretraining already exposed the model to a good amount of dogs.
- On the other hand, when we use pretrained weights on a dataset that is more different from ImageNet, this fine-tuning step can be crucial as the feature extractor also needs to be adjusted by a considerable amount. Such a situation can be demonstrated if choosing CIFAR-100 dataset instead, where fine-tuning boosts validation accuracy by about 10% to pass 80% on EfficientNetB0. In such a case the convergence may take more than 50 epochs.
- A side note on freezing/unfreezing models: setting trainable of a Model will simultaneously set all layers belonging to the Model to the same trainable attribute. Each layer is trainable only if both the layer itself and the model containing it are trainable. Hence when we need to partially freeze/unfreeze a model, we need to make sure the trainable attribute of the model is set to True.
- Tips for fine tuning EfficientNet on unfreezing layers: 
	- The BathcNormalization layers need to be kept frozen (more details). If they are also turned to trainable, the first epoch after unfreezing will significantly reduce accuracy.
	- In some cases it may be beneficial to open up only a portion of layers instead of unfreezing all. This will make fine tuning much faster when going to larger models like B7.
	- Each block needs to be all turned on or off. This is because the architecture includes a shortcut from the first layer to the last layer for each block. Not respecting blocks also significantly harms the final performance.
- Smaller batch size benefit validation accuracy, possibly due to effectively providing regularization.
```python
# `include_top=False`: This option excludes the final Dense layer that turns 1280 features on the penultimate layer into prediction of the 1000 ImageNet classes. Replacing the top layer with custom layers allows using EfficientNet as a feature extractor in a transfer learning workflow.
# `input_tensor`: Optional Keras tensor (i.e. output of `Input()`) to use as image input for the model.
effi_net = EfficientNetB7(include_top=False, input_tensor=z, weights="imagenet")
```