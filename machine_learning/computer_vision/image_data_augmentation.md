# Mixup

# RandAugment

# Data Augmentation
- Reference: https://www.v7labs.com/blog/contrastive-learning-guide
- Methods
  - Colour Jittering: Here, the brightness, contrast, and saturation of an RGB image are changed randomly. This technique is helpful to ensure a model is not memorizing a given object by the scene's colors. While output image colors can appear odd to human interpretation, such augmentations help a model consider the edges and shape of objects rather than only the colors.
  - Image Rotation: An image is rotated randomly within 0-90 degrees. Since rotating an image doesn’t change the core information contained in it (i.e., a dog in an image will still be a dog), models are trained to be rotation invariant for robust prediction.
  - Image Flipping: The image is flipped (mirrored) about its center, either vertically or horizontally. This is an extension of the concept of image rotation-based augmentation.
  - Image Noising: Random noise is added to the images pixel-wise. This technique allows the model to learn how to separate the signal from the noise in the image and makes it more robust to changes in the image during test time. For example, randomly changing some pixels in the image to white or black is known as salt-and-pepper noise (an example is shown below).
  - Random Affine: Affine is a geometric transformation that preserves lines and parallelism, but not necessarily the distances and angles.
- Reference: https://www.tensorflow.org/tutorials/images/data_augmentation, https://www.tensorflow.org/api_docs/python/tf/keras/layers
## Using `Sequential()`
- ***With this option, your data augmentation will happen on device, synchronously with the rest of the model execution, meaning that it will benefit from GPU acceleration.***
- ***Data augmentation is inactive at test time, so the input samples will only be augmented during `fit()`, not when calling `evaluate()` or `predict()`.***
```python
inputs = Input(shape=(img_size, img_size, 3))

data_aug = Sequential([
	RandomRotation(factor=0.1, fill_mode="constant", fill_value=255), 
	RandomTranslation(height_factor=0.1, width_factor=0.1, fill_mode="constant", fill_value=255),  
	RandomFlip("horizontal"), 
	RandomZoom(height_factor=0.1, fill_mode="constant", fill_value=255)]
	)
z = data_aug(inputs)
```
## Using `Dataset.map()`
- With this option, your data augmentation will happen on CPU, asynchronously, and will be buffered before going into the model.
- If you're training on CPU, this is the better option, since it makes data augmentation asynchronous and non-blocking.
## Using `ImageDataGenerator()`
- Reference: https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator, https://m.blog.naver.com/PostView.nhn?blogId=isu112600&logNo=221582003889&proxyReferer=https:%2F%2Fwww.google.com%2F
- Each new batch of our data is randomly adjusting according to the parameters supplied to `ImageDataGenerator()`.
```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# `shear_range`: (float). Shear Intensity (Shear angle in counter-clockwise direction as radians)
# `zoom_range`
	# (`[lower, upper]`) Range for random zoom.
	# (float) Range of `[1 - zoom_range, 1 + zoom_range]`
# `rotation_range`
# `brightness_range`: (Tuple or List of two floats) Range for picking a brightness shift value from.
# `rescale`: rescaling factor. Defaults to None. If None or 0, no rescaling is applied, otherwise we multiply the data by the value provided (before applying any other transformation).
# `horizontal_flip`, `vertical_flip`: (bool). Randomly flip inputs horizontally.
# `width_shift_range`, `height_shift_range`
	# (float) Fraction of total width (or height).
	# (Tuple or List) Random elements from the array.
	# (int) Pixels from interval (-`width_shift_range`, `width_shift_range`) (or (-`height_shift_range`, `height_shift_range`))
# `preprocessing_function`: function that will be applied on each input. The function will run after the image is resized and augmented. The function should take one argument: one image (Numpy tensor with rank 3), and should output a Numpy tensor with the same shape.
gen = ImageDataGenerator([shear_range], [zoom_range], [ratation_range], [brightness_range], [rescale], [horizontal_flip], [vertical_flip], [width_shift_range], [height_shift_range], [validation_split])
# Fits the data generator to some sample data.
# This computes the internal data stats related to the data-dependent transformations, based on an array of sample data.
# Only required if `featurewise_center` or `featurewise_std_normalization` or `zca_whitening` are set to True.
# When `rescale` is set to a value, rescaling is applied to sample data before computing the internal data stats.
# `x`: Sample data. Should have rank 4
gen.fit(x, [seed])
```
```python
gen.apply_transform()
```
- Using `ImageDataGenerator().flow()` (for GoogLeNet, for example)
	- References: https://www.tensorflow.org/api_docs/python/tf/keras/Model, https://gist.github.com/swghosh/f728fbba5a26af93a5f58a6db979e33e
	```python
	def gen_flow(x, y, subset):
		gen = ImageDataGenerator(..., validation_split=0.2)
		gen.fit(X_tr_val)
		# `subset`: (`"training"`, `"validation"`) If `validation_split` is set in `ImageDataGenerator()`.
		# `save_to_dir`: This allows you to optionally specify a directory to which to save the augmented pictures being generated.
		return gen.flow(x=x, y=y, batch_size=batch_size, subset=subset)
	def generator(flow):
		for xi, yi in flow:
			yield xi, [yi, yi, yi]
			
	flow_tr = gen_flow(X_tr_val, y_tr_val, subset="training")
	flow_val = gen_flow(X_tr_val, y_tr_val, subset="validation")

	# `steps_per_epoch`: We can calculate the value of `steps_per_epoch` as the total number of samples in your dataset divided by the batch size.
	# `validation_steps`: Only if the `validation_data` is a generator then only this argument can be used. It specifies the total number of steps taken from the generator before it is stopped at every epoch and its value is calculated as the total number of validation data points in your dataset divided by the validation batch size.
	hist = model.fit(x=generator(flow_tr), validation_data=generator(flow_val), epochs, steps_per_epoch=len(flow_tr), validation_steps=len(flow_val), callbacks=[es, mc])
	```
- Using `ImageDataGenerator().flow_from_directory()`
	```python
	# `target_size`: the dimensions to which all images found will be resized.
	# `class_mode`: (`"binary"`, `"categorical"`, `"sparse"`, `"input"`, `None`)
		# `"binary"`: for binary classification.
		# `"categorical"`: for multi-class classification (OHE).
		# `"sparse"`: for multi-class classification (no OHE).
		# `"input"`
		# `None`: Returns no label.
	```