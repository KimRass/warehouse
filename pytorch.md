# Set Seed
```python
torch.manual_seed(3)
```

# Training
- Reference: https://discuss.pytorch.org/t/how-are-optimizer-step-and-loss-backward-related/7350
```python
for epoch in range(1, n_epochs + 1):
	running_loss = 0
	for batch, (x, y) in enumerate(dl_tr, start=1):
		...
        # Sets each parameter's gradient zero.
		# Clears x.grad for every parameter x in the optimizer
        # Calling `loss.backward()` mutiple times accumulates the gradient (by addition) for each parameter. This is why you should call `optimizer.zero_grad()` after each `optimizer.step()` call.
		optimizer.zero_grad()
		...

		logits = model(inputs)
		loss = criterion(inputs, logits)

        # Backpropogates.
		# Computes $\frac{\partial loss}{\partial x}$ for every parameter $x$ which has `requires_grad=True`. These are accumulated into `x.grad` for every parameter `x`.
		loss.backward()
        # Performs a parameter update based on the current gradient (stored in `.grad` attribute of a parameter)
		# Updates the value of `x` using the gradient `x.grad`.
		optimizer.step()

		running_loss += loss.item()
		...
		if batch % ... == 0:
			...
			running_loss = 0
```

# GPU
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
## To CPU or NumPy
```python
# Returns a copy of this object in CPU memory. If this object is already in CPU memory and on the correct device, then no copy is performed and the original object is returned.
.cpu()
# Returns self tensor as a NumPy ndarray. This tensor and the returned ndarray share the same underlying storage. Changes to self tensor will be reflected in the ndarray and vice versa.
# CPU 메모리에 올려져 있는 tensor만 .numpy() method를 사용할 수 있습니다.
.numpy()
# Order: `.detach().cpu().numpy()`
```
## Reshape Tensor
```python
<Tensor>.view()
```
## Permute Dimentions
```python
# `dims`: (Tuple)
torch.permute(input, dims)
<Tensor>.permute(dims)
```
## Remove Dimensions of Size 1
```python
# `dim`: if given, the input will be squeezed only in this dimension.
torch.squeeze(input, [dim])
```
## Arguments of Maxima
```python
# This is the second value returned by `torch.max()`.
torch.argmax(input, dim)
Tensor.argmax(dim)
```
## Normalize
- https://pytorch.org/docs/stable/generated/torch.nn.functional.normalize.html
```python
# Below two code are the same
# `p`: the exponent value in the norm formulation
F.normalize(input=image_features, dim=1)
image_features / image_features.norm(dim=1, keepdim=True)
```

# Layers without Weights
## Add
## Multiply
## `Dot(axes)`
- `axes` : (integer, tuple of integers) Axis or axes along which to take the dot product. If a tuple, should be two integers corresponding to the desired axis from the first input and the desired axis from the second input, respectively. Note that the size of the two selected axes must match.
## Concatenate
```python
torch.cat(tensors, [dim])
```
## `Flatten([input_shape])`
## `Input(shape, [name], [dtype], ...)`
- `shape`
	- ***A shape tuple (integers), not including the batch size***. For instance, shape=(32,) indicates that the expected input will be batches of 32-dimensional vectors.
	- ***Elements of this tuple can be None; "None" elements represent dimensions where the shape is not known.***
	- Note that `shape` does not include the batch dimension.
## Dropout
```python
Dropout(p, [inplace=False])
```
## Pooling
```python
MaxPool1d()
MaxPool2d()
```
```python
AvgPool1d()
AvgPool2d()
```
## Global Average Pooling
```python
nn.AdaptiveAvgPool2d((1, 1))
```
## `ZeroPadding2D(padding)`
- `padding`:
	- Int: the same symmetric padding is applied to height and width.
	- Tuple of 2 ints: interpreted as two different symmetric padding values for height and width: `(symmetric_height_pad, symmetric_width_pad)`.
	- Tuple of 2 tuples of 2 ints: interpreted as `((top_pad, bottom_pad), (left_pad, right_pad))`.
## Batch Normalization
```python
nn.BatchNorm()
```
- If a `nn.Conv2d` layer is directly followed by a `nn.BatchNorm2d` layer, then the bias in the convolution is not needed, instead use `nn.Conv2d(..., bias=False, ....)`. Bias is not needed because in the first step `nn.BatchNorm` subtracts the mean, which effectively cancels out the effect of bias.
## Layer Normalization
## `Reshape()`
## `Activation(activation)`
- `activation`: (`"relu"`)
## `RepeatVector(n)`
- Repeats the input `n` times.

# Layers with Weights
## Embedding Layer
```python
# Reference: https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html#torch.nn.Embedding
# `padding_idx`: If specified, the entries at `padding_idx` do not contribute to the gradient; therefore, the embedding vector at `padding_idx` is not updated during training, i.e. it remains as a fixed "pad”. For a newly constructed Embedding, the embedding vector at `padding_idx` will default to all zeros, but can be updated to another value to be used as the padding vector.
Embedding(num_embeddings, embedding_dim, padding_idx)
```
## Fully Connected Layer
```python
nn.Linear(in_features, out_features)
```
## Convolution Layer
```python
# Reference: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
# `padding="valid"` is the same as no padding. `padding="same"` pads the input so the output has the shape as the input. However, this mode doesn’t support any `stride` values other than 1.
# `dilation`: Spacing between kernel elements.
nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1)
```
```python
nn.ConvTranspose1d()
nn.ConvTranspose2d()
```
## LSTM
```python
# PyTorch
LSTM(input_size, hidden_size, num_layers, batch_first, dropout, bidirectional)
```
## `Bidirectional([input_shape])`
```python
z, for_h_state, for_c_state, back_h_state, back_c_state = Bidirectional(LSTM(return_state=True))(z)
```

# Interpolate
```python
# `mode`: (`"nearest"`, `"linear"`, `"bilinear"`, `"bicubic"`, `"trilinear"`, `"area"`, `"nearest-exact"`, Default: `"nearest"`)Algorithm used for upsampling
# `align_corners`
	# Reference: https://deep-learning-study.tistory.com/908
F.interpolate(input, size, mode="nearest")
```

# Optimizers
## Adam (ADAptive Moment estimation)
```python
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
	torch.save(obj=weights, f=model_path)
```
```python
torch.save(
	obj={
		'epoch': epoch,
		'model_state_dict': model.state_dict(),
		'optimizer_state_dict': optimizer.state_dict(),
		'loss': loss,
		...
	},
	f=model_path
)
```
```python
from collections import OrderedDict


def _get_state_dict(
    ckpt_path, key="state_dict", include="", delete="", cuda=False
):
    ckpt = torch.load(ckpt_path, map_location="cuda" if cuda else "cpu")

    if key in ckpt:
        state_dict = ckpt[key]
    else:
        state_dict = ckpt

    new_state_dict = OrderedDict()
    for old_key in list(state_dict.keys()):
        if old_key and old_key.startswith(include):
            new_key = old_key.split(delete, 1)[1] if delete else old_key
            new_state_dict[new_key] = state_dict[old_key]
    return new_state_dict
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
		# i.e, if height > width, then image will be rescaled to (size * height / width, size).
		T.Resize(256),
		# If a single int is provided this is used to pad all borders.
		# If sequence of length 2 is provided this is the padding on left/right and top/bottom respectively.
		# If a sequence of length 4 is provided this is the padding for the left, top, right and bottom borders respectively.
		T.Pad(padding=(..., ...)),
		# If size is an int instead of sequence like (h, w), a square crop (size, size) is made.
		T.CenterCrop(224),
		# T.RandomHorizontalFlip(),
		T.ToTensor(),
		T.Normalize(
			mean=(0.485, 0.456, 0.406),
			std=(0.229, 0.224, 0.225)
		)
	]
)
```

# Custom Dataset
```python
transform = 
ds_tr = ImageFolder(root, transform)
ds_te = ImageFolder(root, transform)

batch_size = 64
# Reference: https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html
	# `num_workers`:
		# `DataLoader` supports asynchronous data loading and data augmentation in separate worker subprocesses. The default setting for DataLoader is `num_workers=0`, which means that the data loading is synchronous and done in the main process. As a result the main training process has to wait for the data to be available to continue the execution.
		# Setting `num_workers` > 0 enables asynchronous data loading and overlap between the training and data loading. num_workers should be tuned depending on the workload, CPU, GPU, and location of training data.
	# `pin_memory`:
		# `DataLoader` accepts `pin_memory` argument, which defaults to `False`. When using a GPU it’s better to set `pin_memory=True`, this instructs DataLoader to use pinned memory and enables faster and asynchronous memory copy from the host to the GPU.
dl_tr = DataLoader(dataset=ds_tr, batch_size=batch_size, shuffle=True, num_workers=4)
dl_te = DataLoader(dataset=ds_te, batch_size=batch_size, shuffle=False, num_workers=4)
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

# cuDNN
```python
import torch.backends.cudnn as cudnn

# If `True`, causes cuDNN to benchmark multiple convolution algorithms and select the fastest.
# Reference: https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936/2
# It enables benchmark mode in cudnn. benchmark mode is good whenever your input sizes for your network do not vary. This way, cudnn will look for the optimal set of algorithms for that particular configuration (which takes some time). This usually leads to faster runtime.
# But if your input sizes changes at each iteration, then cudnn will benchmark every time a new size appears, possibly leading to worse runtime performances.
cudnn.benchmark = True
# If `True`, causes cuDNN to only use deterministic convolution algorithms.
# Sets whether PyTorch operations must use “deterministic” algorithms. That is, algorithms which, given the same input, and when run on the same software and hardware, always produce the same output.
# When enabled, operations will use deterministic algorithms when available, and if only nondeterministic algorithms are available they will throw a RuntimeError when called.
cudnn.deterministic = False
```

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

# Move to Device
```python
def move_to_device(obj, device):
    if isinstance(obj, nn.Module) or torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, (tuple, list)):
        return [move_to_device(el, device) for el in obj]
    if isinstance(obj, dict):
        return {name: move_to_device(val, device) for name, val in obj.items()}
    raise ValueError(f"Unexpected type {type(obj)}")
```