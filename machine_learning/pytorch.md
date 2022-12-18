# Set Seed
```python
torch.manual_seed(3)
```

# Training
- Reference: https://discuss.pytorch.org/t/how-are-optimizer-step-and-loss-backward-related/7350
```python
for epoch in range(1, n_epochs + 1):
	running_loss = 0
	for batch, (x, y) in enumerate(dl_tr, 1):
		...
        # Set each parameter's gradient zero.
        # Calling `loss.backward()` mutiple times accumulates the gradient (by addition) for each parameter. This is why you should call `optimizer.zero_grad()` after each `optimizer.step()` call.
		optimizer.zero_grad()
		...

		logits = model(inputs)
		loss = criterion(inputs, logits)

        # Backpropogate.
		loss.backward()
        # Perform a parameter update based on the current gradient (stored in `.grad` attribute of a parameter)
		optimizer.step()

		running_loss += loss.item()
		...
		if batch % ... == 0:
			...
			running_loss = 0
```

# GPU on PyTorch
- Install
  - Reference: https://pytorch.org/get-started/locally/
  ```sh
  # Example
  pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
  ```
```python
# `torch.cuda.is_available()`: Returns a bool indicating if CUDA is currently available.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
...

model = ModelName()
# In-place (Tensor에 대해서는 작동하지 않습니다.)
model = model.to(device)
# Or
model = model.to("cuda")
# Or (Not recommended)
model = model.cuda()
```
```python
# Returns the index of a currently selected device.
torch.cuda.current_device()
# Returns the number of GPUs available.
torch.cuda.device_count()
# Gets the name of the device.
torch.cuda.get_device_name(<index>)
```
```python
import platform
import torch

def check_envirionment(use_cuda: bool):
    device = torch.device(
		"cuda" if use_cuda and torch.cuda.is_available() else "cpu"
	)

    logger.info(f"Operating System : {platform.system()} {platform.release()}")
    logger.info(f"Processor : {platform.processor()}")

    if str(device) == "cuda":
        for idx in range(torch.cuda.device_count()):
            logger.info(f"device : {torch.cuda.get_device_name(idx)}")
        logger.info(f"CUDA is available : {torch.cuda.is_available()}")
        logger.info(f"CUDA version : {torch.version.cuda}")
    else:
        logger.info(f"CUDA is available : {torch.cuda.is_available()}")
    logger.info(f"PyTorch version : {torch.__version__}")
    return device
```

# Tensors
```python
# Returns a copy of this object in CPU memory. If this object is already in CPU memory and on the correct device, then no copy is performed and the original object is returned.
.cpu()
# Returns self tensor as a NumPy ndarray. This tensor and the returned ndarray share the same underlying storage. Changes to self tensor will be reflected in the ndarray and vice versa.
# CPU 메모리에 올려져 있는 tensor만 .numpy() method를 사용할 수 있습니다.
.numpy()
# Order: `.detach().cpu().numpy()`
```
## Operators
### Reshape Tensor
```python
<Tensor>.view()
```
### Permute Dimentions
```python
# `dims`: (Tuple)
torch.permute(input, dims)
<Tensor>.permute(dims)
```
### Remove Dimensions of Size 1
```python
# `dim`: if given, the input will be squeezed only in this dimension.
torch.squeeze(input, [dim])
```
### Arguments of Maxima
```python
# This is the second value returned by `torch.max()`.
torch.argmax(input, dim)
Tensor.argmax(dim)
```

# Layers without Weights
## `tf.stack(values, axis, [name])`
- Reference: https://www.tensorflow.org/api_docs/python/tf/stack
- Stacks a list of tensors of rank R into one tensor of rank (R + 1).
- `axis`: The axis to stack along.
- Same syntax as `np.stack()`
## Add Layers
```python
# TensorFlow
# It takes as input a list of tensors, all of the same shape, and returns a single tensor (also of the same shape).
# 마지막 Deminsion만 동일하면 Input으로 주어진 Tensors 중 하나를 옆으로 늘려서 덧셈을 수행합니다.
Add()()
```
## Multiply Layers
```python
# TensorFlow
Multiply()()
```
## `Dot(axes)`
- `axes` : (integer, tuple of integers) Axis or axes along which to take the dot product. If a tuple, should be two integers corresponding to the desired axis from the first input and the desired axis from the second input, respectively. Note that the size of the two selected axes must match.
## Concatenate Layers
```python
# PyTorch
torch.concat([dim])

# TensorFlow
# Same as `tf.concat(values, [axis], [name])`
Concatenate([axis])()
```
## `Flatten([input_shape])`
## `Input(shape, [name], [dtype], ...)`
- `shape`
	- ***A shape tuple (integers), not including the batch size***. For instance, shape=(32,) indicates that the expected input will be batches of 32-dimensional vectors.
	- ***Elements of this tuple can be None; "None" elements represent dimensions where the shape is not known.***
	- Note that `shape` does not include the batch dimension.
## Dropout Layer
```python
# TensorFlow
# `rate`
	# The Dropout layer randomly sets input units to 0 with a frequency of `rate` at each step during training time, which helps prevent overfitting. Inputs not set to 0 are scaled up by 1/(1 - `rate`) such that the sum over all inputs is unchanged.
	# Note that the `Dropout` layer only applies when `training` is set to `True` such that no values are dropped during inference. When using `model.fit`, `training` will be appropriately set to `True` automatically, and in other contexts, you can set the kwarg explicitly to `True` when calling the layer.
Dropout(rate)

# PyTorch
Dropout(p, [inplace=False])
```
## Pooling Layer
```python
# Tensorflow
# Output Dimension
	# When `padding="valid"`: `(input_dim - pool_size)//strides + 1`
	# When `padding="same"`: `input_dim//strides + 1`
MaxPool1D(pool_size, strides, padding, [data_format]) # Same as `MaxPooling1D()`
MaxPool2D() # Same as `MaxPooling2D()`

# PyTorch
MaxPool1d()
MaxPool2d()
```
```python
# TensorFlow
# Shape: `(a, b, c, d)` -> `(a, d)`.
GlobalMaxPool1D() # Same as `GlobalMaxPooling1D()`
# Downsamples the input representation by taking the maximum value over the time dimension.
# Shape: `(a, b, c)` -> `(b, c)`.
GlobalMaxPool2D() # Same as  `GlobalMaxPooling2D()`
```
```python
# TensorFlow
AveragePooling1D([pool_size], [strides], [padding])
AveragePooling2D()

# PyTorch
AvgPool1d()
AvgPool2d()
```
## `GlobalAveragePooling1D()`, `GlobalAveragePooling2D()`
## `ZeroPadding2D(padding)`
- `padding`:
	- Int: the same symmetric padding is applied to height and width.
	- Tuple of 2 ints: interpreted as two different symmetric padding values for height and width: `(symmetric_height_pad, symmetric_width_pad)`.
	- Tuple of 2 tuples of 2 ints: interpreted as `((top_pad, bottom_pad), (left_pad, right_pad))`.
## `BatchNormalization()`
- Reference: https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization
- Usually used before activation function layers.
## `LayerNormalization([epsilon], axis)`
- Reference: https://www.tensorflow.org/api_docs/python/tf/keras/layers/LayerNormalization
- ***Normalize the activations of the previous layer for each given example in a batch independently, rather than across a batch like Batch Normalization. i.e. applies a transformation that maintains the mean activation within each example close to 0 and the activation standard deviation close to 1.***
- `epsilon`: Small float added to variance to avoid dividing by zero. Defaults to `1e-3`.
## `Reshape()`
## `Activation(activation)`
- `activation`: (`"relu"`)
## `RepeatVector(n)`
- Repeats the input `n` times.

# Layers with Weights
## Embedding Layer
```python
# TensorFlow
# Reference: https://www.tensorflow.org/api_docs/python/tf/keras/layers/Embedding
# `input_dim`: Size of the vocabulary.
# `output_dim`: Dimension of the dense embedding.
# `input_length`: Length of input sequences, when it is constant. This argument is required if you are going to connect `Flatten()` then `Dense ()` layers upstream.
# `mask_zero=True`: Whether or not the input value 0 is a special "padding" value that should be masked out. This is useful when using recurrent layers which may take variable length input. If `mask_zero` is set to `True`, as a consequence, index 0 cannot be used in the vocabulary (`input_dim` should equal to `vocab_size + 1`)).
# Shape: `(batch_size, input_length)` -> `(batch_size, input_length, output_dim)`
Embedding(input_dim, output_dim, [input_length], [mask_zero], [name], [weights], [trainable], ...)

# PyTorch
# Reference: https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html#torch.nn.Embedding
# `padding_idx`: If specified, the entries at `padding_idx` do not contribute to the gradient; therefore, the embedding vector at `padding_idx` is not updated during training, i.e. it remains as a fixed "pad”. For a newly constructed Embedding, the embedding vector at `padding_idx` will default to all zeros, but can be updated to another value to be used as the padding vector.
Embedding(num_embeddings, embedding_dim, padding_idx)
```
## Fully Connected Layer
```python
# Tensorflow
# Reference: https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense
# `units`: Dimensionality of the output space.
# `activation`: Activation function to use. If you don't specify anything, no activation is applied (ie. "linear" activation)
# Shape: `(batch_size, ..., input_dim)` -> `(batch_size, ..., units)`
# Note that after the first layer, you don't need to specify the size of the input anymore.
Dense(units, [activation])

# PyTorch
nn.Linear(in_features, out_features)
```
## Convolution Layer
```python
# TensorFlow
# `kernal_size`: window_size
# `padding="valid"`: No padding. 
# `padding="same"`: Results in padding with zeros evenly to the left/right or up/down of the input such that output has the same height/width dimension as the input.
# `data_format`: (`"channels_last"`, `"channels_first"`)
# `activation`: (`"tanh"`)
# Output Dimension
	# When `padding="valid"`: `math.ceil(input_dim - kernel_size + 1)/strides`
	# When `padding="same"`: `math.ceil(input_dim/strides)`
Conv1D(filters, kernel_size, strides, padding, activation, data_format)
Conv2D()
Conv1DTranspose()
Conv2DTranspose()

# PyTorch
nn.Conv1d()
# `padding="valid"` is the same as no padding. `padding="same"` pads the input so the output has the shape as the input. However, this mode doesn’t support any `stride` values other than 1.
# `dilation`: Spacing between kernel elements.
nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1)
nn.ConvTranspose1d()
nn.ConvTranspose2d()
```
## LSTM
```python
# TensorFlow
# Reference: https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM
# `return_sequences`: Whether to return the last output. in the output sequence, or the full sequence.
	# `True`: 모든 timestep에서 Output을 출력합니다. (Output shape: `(batch_size, timesteps, h_size)`)
	# `False` (default): 마지막 timestep에서만 Output을 출력합니다. (Output shape: `(batch_size, h_size)`)
# `return_state`: Whether to return the last state in addition to the output. (`output, h_state, c_state = LSTM(return_state=True)()`)
# Call arguments
	# `mask`
	# `training`
	# `initial_state`: List of initial state tensors to be passed to the first call of the cell (optional, defaults to `None` which causes creation of zero-filled initial state tensors).
LSTM(units, return_sequences, return_state, [dropout])([initial_state])

# PyTorch
LSTM(input_size, hidden_size, num_layers, batch_first, dropout, bidirectional)
```
## `Bidirectional([input_shape])`
```python
z, for_h_state, for_c_state, back_h_state, back_c_state = Bidirectional(LSTM(return_state=True))(z)
```
## `TimeDistributed()`
- Reference: https://www.tensorflow.org/api_docs/python/tf/keras/layers/TimeDistributed
- This wrapper allows to apply a layer to every temporal slice of an input.
- For example, consider a batch of 32 video samples, where each sample is a 128x128 RGB image with channels_last data format, across 10 timesteps. The batch input shape is (32, 10, 128, 128, 3). You can then use `TimeDistributed()` to apply the same `Conv2D()` layer to each of the `10` timesteps, independently. Because `TimeDistributed()` applies the same instance of `Conv2D()` to each of the timestamps, the same set of weights are used at each timestamp.

# Optimizers
## Adam (ADAptive Moment estimation)
```python
# TensorFlow
Adam(learning_rate, beta_1, beta_2, epsilon, name)

# PyTorch
Adam([lr=0.001], [betas=(0.9, 0.999)], [eps=1e-08])
```

# Model
## Build Model
```python
model = Model(inputs, ouputs, [name])
model.summary()
# `to_file`: File name of the plot image.
# `show_layer_activations`: Display layer activations
plot_model(model, [to_file], [show_layer_activations])
```
## Compile
### PyTorch
```python
# `optimizer`: `SGD(model.parameters(), lr, momentum)`
# `criterion`: `BCELoss()`, `CrossEntropyLoss()`
```
## Train Model
## Training History
## Model Methods
```python
# PyTorch
# Iterate model layers
for layer in model.parameters():
	...
```
## Check Model Weights
```python
layer.size()
```

# Inference
## TensorFlow
- Reference: https://www.tensorflow.org/api_docs/python/tf/keras/Model?hl=en, https://stackoverflow.com/questions/60837962/confusion-about-keras-model-call-vs-call-vs-predict-methods
- `model(x)`
	- Calls the model on new inputs and returns the outputs as `tf.Tensor`s.
	- ***For small numbers of inputs that fit in one batch, directly use `__call__()` for faster execution, e.g., `model(x)`, or `model(x, training=False)` if you have layers such as `BatchNormalization()` that behave differently during inference. You may pair the individual model call with a `@tf.function()` for additional performance inside your inner loop.***
	- ***After `model(x)`, you can use `tf.Tensor.numpy()` to get the numpy array value of an eager tensor.***
	- Also, note the fact that test loss is not affected by regularization layers like noise and dropout.
- `model.predict()`
	- ***Computation is done in batches. This method is designed for batch processing of large numbers of inputs. It is not intended for use inside of loops that iterate over your data and process small numbers of inputs at a time.***
- `model.predict_on_batch()`
	- Returns predictions for a single batch of samples.
	- The difference between `model.predict()` and `model.predict_on_batch()` is that the latter runs over a single batch, and the former runs over a dataset that is splitted into batches and the results merged to produce the final `numpy.ndarray` of predictions.
## PyTorch
```python
# Evaluation (Inference) mode로 전환합니다.
# `Dropout()`, `BatchNorm()`은 Training mode에서만 작동하며 Evaluation mode에서는 작동하지 않습니다.
# 메모리와는 관련이 없습니다.
# `model.train()`: Train mode로 전환합니다.
model.eval()

# Disabling gradient calculation is useful for inference, when you are sure that you will not call `Tensor.backward()`.
# 메모리를 줄여주고 연산 속도를 증가시킵니다.
# `Dropout()`, `BatchNorm()`을 비활성화시키지는 않습니다.
with torch.no_grad():
	...
```

# PyTorch `DataLoader`
```python
dl_tr = DataLoader(dataset, batch_size, [shuffle=False], [num_workers=0], [prefetch_factor=2])
...

next(iter(dl_tr))
```

# Save or Load Model
- Reference: https://pytorch.org/tutorials/beginner/saving_loading_models.html
```python
save_dir = Path("...")
model_path = save_dir / "model_name.pth"
# hist_path = save_dir / "model_name_hist.npy"
if os.path.exists(model_path):
	weights = torch.load(model_path)
	# `map_location`: (`"cpu"`, `"cuda"`, `device`)
    model.load_state_dict(weights, map_location=device)
else:
	...
	# Loads a model’s parameter dictionary using a deserialized `state_dict()`.
	# In PyTorch, the learnable parameters (i.e. weights and biases) of an `torch.nn.Module` model are contained in the model’s parameters (accessed with `model.parameters()`). A `state_dict()` is simply a Python dictionary object that maps each layer to its parameter tensor. Note that only layers with learnable parameters (convolutional layers, linear layers, etc.) and registered buffers (batchnorm’s running_mean) have entries in the model’s `state_dict()`. Optimizer objects (`torch.optim``) also have a `state_dict()`, which contains information about the optimizer’s state, as well as the hyperparameters used.
	weights = model.state_dict()
	torch.save(weights, model_path)
```
```python
torch.save(
	{
		'epoch': epoch,
		'model_state_dict': model.state_dict(),
		'optimizer_state_dict': optimizer.state_dict(),
		'loss': loss,
		...
	},
	model_path
)
```

# Save or Load Parameters
```python
torch.load(model_path)
```

# Custrom Model
```python
class ModelName(nn.Module):
	def __init__(self, ...):
		super().__init__()
		# Or `super(ModelName, self).__init__()`
		self.var1 = ...
		self.var2 = ...
		...

	def forward(self, x):
		...
...
model = ModelName()
```

# Custom Learning Rate

# Import
```python
import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader

import torchvision
import torchvision.datasets as D
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
```

# transform
```python
transform = T.Compose(
	[
		# Reference: https://pytorch.org/vision/main/generated/torchvision.transforms.Resize.html
		# If size is a sequence like `(h, w)`, output size will be matched to this.
		# If size is an int, smaller edge of the image will be matched to this number.
		T.Resize(256),
		T.CenterCrop(224),
		# T.RandomHorizontalFlip(),
		T.ToTensor(),
		T.Normalize(
			mean=[0.485, 0.456, 0.406],
			std=[0.229, 0.224, 0.225]
		)
	]
)

# Custom Dataset
```python
transform = 
ds_tr = ImageFolder(root, transform)
ds_te = ImageFolder(root, transform)

batch_size = 64
dl_tr = DataLoader(dataset=ds_tr, batch_size=batch_size, shiffle=True, num_workers=4)
dl_te = DataLoader(dataset=ds_te, batch_size=batch_size, shiffle=False, num_workers=4)
```

# Pre-trained Models
## Keypoint R-CNN
- Reference: https://pytorch.org/vision/stable/models/keypoint_rcnn.html
```python
from torchvision import models

model = models.detection.keypointrcnn_resnet50_fpn(
	pretrained=True
).eval()

trf = T.Compose(
	[
		T.ToTensor()
	]
)
input_img = trf(img)
out = model([input_img])[0]
```

# Data Parallelism
```python
if torch.cuda.device_count() > 1:
	print(f"Number of GPUs: {torch.cuda.device_count()}")

	model = nn.DataParallel(model)
```

# `import torch.backends.cudnn.benchmark`
- Reference: https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936/2
- It enables benchmark mode in cudnn. benchmark mode is good whenever your input sizes for your network do not vary. This way, cudnn will look for the optimal set of algorithms for that particular configuration (which takes some time). This usually leads to faster runtime.
- But if your input sizes changes at each iteration, then cudnn will benchmark every time a new size appears, possibly leading to worse runtime performances.

# Model Summary
```python
# `pip install torchsummary`
from torchsummary import summary

summary(model=model, input_size=(3, 224, 224))
```

# Initilaize Parameters
```python
def initialize_weights(modules):
    for m in modules:
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()
```