# TensorFlow Graph Excution
- Reference: https://towardsdatascience.com/eager-execution-vs-graph-execution-which-is-better-38162ea4dbf6
- In TensorFlow 2.0, you can decorate a Python function using `tf.function()` to run it as a single graph object. With this new method, you can easily build models and gain all the graph execution benefits.
- For simple operations, graph execution does not perform well because it has to spend the initial computing power to build a graph. We see the power of graph execution in complex calculations.
```python
import time

# (`SGD()`, `Adagrad()`, `Adam()`, ...)
optimizer = ...

loss_obj = losses....
def loss_func(y_true, y_pred):
    ...
    return ...

def acc_func(real, pred):
    ...
    return ...

tr_loss = metrics....
tr_acc = metrics....

# `input_signature`: A possibly nested sequence of `tf.TensorSpec()` objects specifying the `shape`s and `dtype`s of the Tensors that will be supplied to this function. If `None`, a separate function is instantiated for each inferred `input_signature`. If `input_signature` is specified, every input to func must be a Tensor, and `func` cannot accept `**kwargs`.
    # The input signature specifies the shape and type of each Tensor argument to the function using a tf.TensorSpec object. More general shapes can be used. This ensures only one ConcreteFunction is created, and restricts the GenericFunction to the specified shapes and types. It is an effective way to limit retracing when Tensors have dynamic shapes.
# Since TensorFlow matches tensors based on their shape, using a `None` dimension as a wildcard will allow functions to reuse traces for variably-sized input. Variably-sized input can occur if you have sequences of different length, or images of different sizes for each batch.
# The @tf.function trace-compiles train_step into a TF graph for faster execution. The function specializes to the precise shape of the argument tensors. To avoid re-tracing due to the variable sequence lengths or variable batch sizes (the last batch is smaller), use input_signature to specify more generic shapes.
# tf.function only allows creating new tf.Variable objects when it is called for the first time:
@tf.function(input_signature=...)
def train_step(x, y):
	...
    with tf.GradientTape() as tape:
        y_pred = model(x, ..., training=True)
        loss = loss_func(y, y_pred)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    tr_loss(loss)
    tr_acc(acc_func(y, y_pred))
    
ckpt_path = "..."
# TensorFlow objects may contain trackable state, such as `tf.Variables`, `tf.keras.optimizers.Optimizer` implementations, `tf.data.Dataset` iterators, `tf.keras.Layer` implementations, or `tf.keras.Model` implementations. These are called trackable objects.
# A `Checkpoint` object can be constructed to save either a single or group of trackable objects to a checkpoint file. It maintains a `save_counter` for numbering checkpoints.
ckpt = tf.train.Checkpoint(optimizer=optimizer, model=model)
# Manages multiple checkpoints by keeping some and deleting unneeded ones.
ckpt_manager = tf.train.CheckpointManager(ckpt, directory=ckpt_path, max_to_keep=...)
# The prefix of the most recent checkpoint in directory.
if ckpt_manager.latest_checkpoint:
    # `save_path`: The path to the checkpoint, as returned by `save` or `tf.train.latest_checkpoint`.
    ckpt.restore(save_path=ckpt_manager.latest_checkpoint)
    print ("Latest checkpoint restored!")
	
epochs = ...
for epoch in range(1, epochs + 1):
    start = time.time()
    # Resets all of the metric state variables to a predefined constant (typically 0). This function is called between epochs/steps, when a metric is evaluated during training.
    tr_loss.reset_states()
    tr_acc.reset_states()
    for batch, (x, y) in dataset_tr.enumerate(start=1):
        train_step(x, y)
        if batch % 50 == 0:
            print(f"Epoch: {epoch:3d} | Batch: {batch:5d} | Loss: {tr_loss.result():5.4f} | Accuracy: {tr_acc.result():5.4f}")
    if epoch % 1 == 0:
        # Every time `ckpt_manager.save()` is called, `save_counter` is increased.
        # `save_path`: The path to the new checkpoint. It is also recorded in the `checkpoints` and `latest_checkpoint` properties. `None` if no checkpoint is saved.
        save_path = ckpt_manager.save()
        print(f"Saving checkpoint for epoch {epoch} at {save_path}")
        print(f"Epoch: {epoch:3d} | Loss: {tr_loss.result():5.4f} | Accuracy: {tr_acc.result():5.4f}")
        print(f"Time taken for 1 epoch: {time.time() - start:5.0f} secs\n")
```

# PyTorch
```python
for epoch in range(1, n_epochs + 1):
	running_loss = 0
	for batch, (x, y) in enumerate(dl_tr, 1):
		...
		optimizer.zero_grad()
		...
		outputs = model(inputs)
		loss = criterion(inputs, outputs)
		loss.backward()
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
## TensorFlow
- References: https://www.tensorflow.org/api_docs/python/tf/Tensor, https://stackoverflow.com/questions/57660214/what-is-the-utility-of-tensor-as-opposed-to-eagertensor-in-tensorflow-2-0, https://www.tensorflow.org/api_docs/python/tf/shape
- `EagerTensor`
	```python
	# During eager execution, you may discover your Tensors are actually of type `EagerTensor` (`tensorflow.python.framework.ops.EagerTensor`)
	tf.Tensor(..., shape=..., dtype=...)

	<EagerTensor>.numpy()

	# `tf.shape()` and `Tensor.shape` should be identical in eager mode.
	<EagerTensor>.shape
	```
- `Tensor`
	```python
	# During graph execution
	# In `tf.function` definitions, the shape may only be partially known. Most operations produce tensors of fully-known shapes if the shapes of their inputs are also fully known, but in some cases it's only possible to find the shape of a tensor at execution time.
	# `Tensor` (`tensorflow.python.framework.ops.Tensor`) represents a tensor node in a graph that may not yet have been calculated.

	# During graph execution, not all dimensions may be known until execution time. Hence when defining custom layers and models for graph mode, prefer the dynamic `tf.shape(x)` over the static `x.shape`.
	tf.shape(<Tensor>)
	```
## PyTorch
```python
# Returns a copy of this object in CPU memory. If this object is already in CPU memory and on the correct device, then no copy is performed and the original object is returned.
.cpu()
# Returns self tensor as a NumPy ndarray. This tensor and the returned ndarray share the same underlying storage. Changes to self tensor will be reflected in the ndarray and vice versa.
# CPU 메모리에 올려져 있는 tensor만 .numpy() method를 사용할 수 있습니다.
.numpy()
# Order: `.detach().cpu().numpy()`
```

# Operators
## `tf.identity()`
## `tf.constant()`
## `tf.convert_to_tensor()`
## `tf.cast([dtype])`
- Casts a tensor to a new type.
- Returns `1` if `True` else `0`.
## Reshape Tensor
```python
# TensorFlow
tf.reshape(<Tensor>, shape)
<Tensor>.reshape(shape)

# PyTorch
<Tensor>.view()
```
## Permute Dimentions
```python
# TensorFLow
# `perm`: (Tuple)
tf.transpose(a, perm)
<Tensor>.transpose(perm)

# PyTorch
# `dims`: (Tuple)
torch.permute(input, dims)
<Tensor>.permute(dims)
```
## `tf.range()`
## `tf.tile()`
## `tf.constant_initializer()`
## `tf.argsort()`
- `direction`: (`"ASCENDING"`, `"DESCENDING"`).
## `tf.math.add()`, `tf.math.subtract()`, `tf.math.multiply()`, `tf.math.divide()`
- Adds, substract, multiply or divide two input tensors element-wise.
## `tf.math.add_n(inputs)`
- Adds all input tensors element-wise.
- `inputs`: A list of Tensors, each with the same shape and type.
## `tf.math.square()`
- Compute square of x element-wise.
## `tf.math.sqrt()`
## Remove Dimensions of Size 1
```python
# TensorFlow
tf.squeeze(input, [axis])
# PyTorch
# `dim`: if given, the input will be squeezed only in this dimension.
torch.squeeze(input, [dim])
```

## Arguments of Maxima
```python
# TensorFlow
tf.math.argmax(axis)
# Pytorch
# This is the second value returned by `torch.max()`.
torch.argmax(input, dim)
Tensor.argmax(dim)
```
## `tf.math.sign`
## `tf.math.exp()`
## `tf.math.log()`
## `tf.math.equal()`
```python
seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
```
## `tf.math.reduce_sum([axis])`, `tf.math.reduce_mean()`
- Reference: https://www.tensorflow.org/api_docs/python/tf/math/reduce_sum#returns_1
- `axis=None`: Reduces all dimensions.
- Reduces `input_tensor` along the dimensions given in `axis`. Unless `keepdims=True`, the rank of the tensor is reduced by `1` for each of the entries in `axis`, which must be unique. If `keepdims=True`, the reduced dimensions are retained with length `1`.
## `tf.math.logical_and()`, `tf.math.logical_or()`
## `tf.math.logical_not(x)`
- Returns the truth value of `NOT x` element-wise.
## `tf.linalg.matmul(a, b, [transpose_a], [transpose_b])`

# Create Tensors
## `tf.Variable(initial_value, [shape=None], [trainable=True], [validate_shape=True], [dtype], [name])`
- Reference: https://www.tensorflow.org/api_docs/python/tf/Variable
- `initial_value`: This initial value defines the type and shape of the variable. After construction, the type and shape of the variable are fixed.
- [`shape`]: The shape of this variable. If `None`, the shape of `initial_value` will be used.
- `validate_shape`: If `False`, allows the variable to be initialized with a value of unknown shape. If `True`, the default, the shape of `initial_value` must be known.
- [`dtype`]: If set, `initial_value` will be converted to the given type. If `None`, either the datatype will be kept (if `initial_value` is a Tensor), or `convert_to_tensor()` will decide.
## `tf.zeros()`
```python
W = tf.Variable(tf.zeros([2, 1], dtype=tf.float32), name="weight")
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
Linear(in_features, out_features)
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
Conv1d()
Conv2d()
ConvTranspose1d()
ConvTranspose2d()
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

# Optimizer
## Stochastic Gradient Descent (SGD)
```python
from tensorflow.keras.optimizers import SGD
```
## Adagrad
```python
from tensorflow.keras.optimizers import Adagrad
```
- Reference: https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adagrad
- *Adagrad tends to benefit from higher initial learning rate values compared to other optimizers.*
## RMSprop (Root Mean Square ...)
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
### TensorFlow
```python
# `optimizer`: (`"sgd"`, `"adam"`, `Adam(learning_rate)`, "rmsprop"`, Adagrad(learning_rate)]
# `loss`: (`"mse"`, `"mae"`, `"binary_crossentropy"`, `"categorical_crossentropy"`, `"sparse_categorical_crossentropy"`)
	# If the model has multiple outputs, you can use a different `loss` on each output by passing a dictionary or a list of `loss`es.
	# `"categorical_crossentropy"`: Produces a one-hot array containing the probable match for each category.
	# `"sparse_categorical_crossentropy"`: Produces a category index of the most likely matching category.
# `metrics`: (`["mse"]`, `["mae"]`, `["binary_accuracy"]`, `["categorical_accuracy"]`, `["sparse_categorical_crossentropy"]`, `["acc"]`)
# When you pass the strings "accuracy" or "acc", we convert this to one of ``BinaryAccuracy()`, ``CategoricalAccuracy()`, `SparseCategoricalAccuracy()` based on the loss function used and the model output shape.
# `loss_weights`: The `loss` value that will be minimized by the model will then be the weighted sum of all individual `loss`es, weighted by the `loss_weights` coefficients. 
model.compile(optimizer, loss, [loss_weights], [metrics], [loss_weights])
```
### PyTorch
```python
# `optimizer`: `SGD(model.parameters(), lr, momentum)`
# `criterion`: `BCELoss()`, `CrossEntropyLoss()`
```
## Train Model
### TensorFlow
- Reference: https://keras.io/api/models/model_training_apis/
```python
# `mode`: (`"auto"`, `"min"`, `"max"`).
	# `"min"`: Training will stop when the quantity monitored has stopped decreasing;
	# `"max"`: It will stop when the quantity monitored has stopped increasing;
	# `"auto"`: The direction is automatically inferred from the name of the monitored quantity.
# `patience`: Number of epochs with no improvement after which training will be stopped.
es = EarlyStopping(monitor="val_loss", mode="auto", verbose=1, patience=2)
model_path = "model_path.h5"
# `save_best_only=True`: `monitor` 기준으로 가장 좋은 값으로 모델이 저장됩니다.
# `save_best_only=False`: 매 epoch마다 모델이 filepath{epoch}으로 저장됩니다.
# `save_weights_only`: If `True`, then only the model's weights will be saved (`model.save_weights(filepath)`), else the full model is saved (`model.save(filepath)`).
mc = ModelCheckpoint(filepath=model_path, monitor="val_acc", mode="auto", verbose=1, save_best_only=True)
# `verbose=2`: One line per epoch. recommended.
hist = model.fit(x, y, [validation_split], [validation_data], batch_size, epochs, verbose=2, [shuffle], callbacks=[es, mc])
```
### PyTorch
```python

```
## Training History
```python
fig, axes = plt.subplots(1, 2, figsize=(12, 8))
axes[0].plot(hist.history["loss"][1:], label="loss")
axes[0].plot(hist.history["val_loss"][1:], label="val_loss")
axes[0].legend()

axes[1].plot(hist.history["acc"][1:], label="acc")
axes[1].plot(hist.history["val_acc"][1:], label="val_acc")
axes[1].legend()
```
## Evaluate Model
```python
te_loss, te_acc = model.evaluate(X_te, y_te, batch_size)
```
## Model Methods
```python
# TensorFlow
model.inputs
model.trainable_variables
# Iterate model layers
for layer in model.layers:
	...
# Get the layer by its name
layer = model.get_layer("<layer_name>")

# PyTorch
# Iterate model layers
for layer in model.parameters():
	...
```
## Check Model Weights
```python
# TensorFlow
layer.name # Layer name
layer.output # Output
layer.input_shape # Input shape
layer.output_shape # Output shape
layer.get_weights()[0] # Weight
layer.get_weights()[1] # Bias

# PyTorch
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

# TensorFlow `Dataset`
- Reference: https://www.tensorflow.org/api_docs/python/tf/data/Dataset
- Dataset usage follows a common pattern:
	- Create a Reference dataset from your input data.
	- Apply dataset transformations to preprocess the data.
	- Iterate over the dataset and process the elements. Iteration happens in a streaming fashion, so the full dataset does not need to fit into memory. (Element: A single output from calling `next()` on a dataset iterator. Elements may be nested structures containing multiple components.)
- Methods
	- `as_numpy_iterator()`
		- Returns an iterator which converts all elements of the dataset to numpy.
		- This will preserve the nested structure of dataset elements.
		- This method requires that you are running in eager mode and the dataset's element_spec contains only `tf.TensorSpec` components.
	- `batch(batch_size, [drop_remainder=False])`
		- Combines consecutive elements of this dataset into batches.
		- The components of the resulting element will have an additional outer dimension, which will be `batch_size` (or `N % batch_size` for the last element if `batch_size` does not divide the number of input elements `N` evenly and `drop_remainder=False`). If your program depends on the batches having the same outer dimension, you should set the `drop_remainder=True` to prevent the smaller batch from being produced.
	- `padded_batch()`
		- Pad to the smallest per-`batch size` that fits all elements.
		- Unlike `batch()`, the input elements to be batched may have different shapes, and this transformation will pad each component to the respective shape in `padded_shapes`. The `padded_shapes` argument determines the resulting shape for each dimension of each component in an output element.
		- `padded_shapes`:
			- If `None`: The dimension is unknown, the component will be padded out to the maximum length of all elements in that dimension.
			- If not `None`: The dimension is a constant, the component will be padded out to that length in that dimension.
		- `padding_values`
		- `drop_remainder`
	- `cache(filename)`
		- Caches the elements in this dataset.
		- The first time the dataset is iterated over(e.g., `map()`, `filter()`, etc.), its elements will be cached either in the specified file or in memory. Subsequent iterations will use the cached data.
		- For the cache to be finalized, the input dataset must be iterated through in its entirety. Otherwise, subsequent iterations will not use cached data.
		- `filename`: When caching to a file, the cached data will persist across runs. Even the first iteration through the data will read from the cache file. Changing the input pipeline before the call to `cache()` will have no effect until the cache file is removed or the `filename` is changed. If a `filename` is not provided, the dataset will be cached in memory.
		- `cache()` will produce exactly the same elements during each iteration through the dataset. If you wish to randomize the iteration order, make sure to call `shuffle()` after calling `cache()`.
	- `prefetch(buffer_size)`
		- Most dataset input pipelines should end with a call to prefetch. This allows later elements to be prepared while the current element is being processed. This often improves latency and throughput, at the cost of using additional memory to store prefetched elements.
		- `buffer_size`: The maximum number of elements that will be buffered when prefetching. If the value `tf.data.AUTOTUNE` is used, then the buffer size is dynamically tuned.
	- `enumerate([start=0])`
	- `filter(predicate)`
		- `predicate`: A function mapping a dataset element to a boolean.
		- Returns the dataset containing the elements of this dataset for which `predicate` is `True`.
	- `from_tensor_slices()`
	- `from_tensors()`
	- `from_generator()`
		- `generator`: Must be a callable object that returns an object that supports the `iter()` protocol (e.g. a generator function).
		- `output_types`: A (nested) structure of `tf.DType` objects corresponding to each component of an element yielded by generator.
		- `ouput_signature`: A (nested) structure of tf.TypeSpec objects corresponding to each component of an element yielded by generator.
		```python
		def gen():
			yield ...
		dataset = tf.data.Dataset.from_generator(gen, ...)
		```
	- `map(map_func)`
		- This transformation applies `map_func` to each element of this dataset, and returns a new dataset containing the transformed elements, in the same order as they appeared in the input. `map_func` can be used to change both the values and the structure of a dataset's elements.
	- `random()`
	- `range()`
	- `repeat()`
	- `shuffle(buffer_size, [seed=None], [reshuffle_each_iteration=None])`
		- `buffer_size`: For perfect shuffling, greater than or equal to the full size of the dataset is required. If not, only the first `buffer_size` elements will be selected randomly.
		- `reshuffle_each_iteration`: Controls whether the shuffle order should be different for each epoch.
	- `skip(count)`
	- `take(count)`
	- `unique()`
	- `zip()`

# PyTorch `DataLoader`
```python
dl_tr = DataLoader(dataset, batch_size, [shuffle=False], [num_workers=0], [prefetch_factor=2])
...

next(iter(dl_tr))
```

# Save or Load Model
## TensorFlow
- Reference: https://www.tensorflow.org/tutorials/keras/save_and_load
```python
save_dir = Path("...")
model_path = save_dir / "model_name.h5"
hist_path = save_dir / "model_name_hist.npy"
if os.path.exists(model_path):
    model = load_model(model_path)
    hist = np.load(hist_path, allow_pickle="TRUE").item()
else:
	...
	# The weight values
	# The model's architecture
	# The model's training configuration (what you pass to the .compile() method)
	# The optimizer and its state, if any (this enables you to restart training where you left off)
	model.save(model_path)
	np.save(hist_path, hist.history)
```
## PyTorch
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

# Save or Load Weights
## TensorFlow
```python
model.compile(...)
...
model.load_weights(model_path)
```
- As long as two models share the same architecture you can share weights between them. So, when restoring a model from weights-only, create a model with the same architecture as the original model and then set its weights.
## PyTorch
```python
torch.load(model_path)
```

# Custrom Model
## TensorFlow
- Reference: https://www.tensorflow.org/api_docs/python/tf/keras/Model
```python
class ModelName(Model):
	# You should define your layers in `__init__()`.
	def __init__(self, ...):
		super().__init__()
		self.var1 = ...
		self.var2 = ...
		...
	# You should implement the model's forward pass in `__call__()`.
	# If you subclass `Model`, you can optionally have a `training` argument (boolean) in `__call__()`, which you can use to specify a different behavior in training and inference.
	def __call__(self, ..., [training]):
		...
		return ...
...
model = ModelName()
```
## PyTorch
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

# Custom Layer
## TensorFlow
```python
class LayerName(Layer):
	def __init__(self, ...):
		super().__init__()
		self.var1 = ...
		self.var2= ...
		...
	def __call__(self, ...):
		...
		return ...
```

# Custom Learning Rate
## TensorFlow
```python
class LearningRate(LearningRateSchedule):
    def __init__(self, warmup_steps=4000):
        super(self).__init__()

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        return (d_model**-0.5)*tf.math.minimum(step**-0.5, step*(self.warmup_steps**-1.5))

lr = LearningRate()
```

# `tf.keras.utils.image_dataset_from_directory()`
- References: https://www.tensorflow.org/api_docs/python/tf/keras/utils/image_dataset_from_directory, https://www.tensorflow.org/tutorials/load_data/images
```python
ds_tr = image_dataset_from_directory(
    "computer_vision_task/train",
    validation_split=0.2,
    subset="training",
    seed=seed,
    shuffle=True,
    image_size=(img_size, img_size),
    batch_size=64
)
ds_val = image_dataset_from_directory(
    "computer_vision_task/train",
    validation_split=0.2,
    subset="validation",
    seed=seed,
    shuffle=True,
    image_size=(img_size, img_size),
    batch_size=64
)
ds_te = image_dataset_from_directory(
    "computer_vision_task/test",
    seed=seed,
    shuffle=False,
    image_size=(img_size, img_size),
    batch_size=64
)
```

# Import Machine Learning Libraries
## scikit-learn
```python
from sklearn.manifold import TSNE
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, KFold, GroupKFold, LeaveOneOut, LeaveOneGroupOut
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler, Normalizer
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
```
## TensorFLow
```python
import tensorflow as tf
from tensorflow.keras import Input, Model, Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer, Dense, Flatten, Dropout, Concatenate, Add, Dot, Multiply, Reshape, Activation, BatchNormalization, LayerNormalization, SimpleRNNCell, RNN, SimpleRNN, LSTM, Embedding, Bidirectional, TimeDistributed, Conv1D, Conv2D, Conv1DTranspose, Conv2DTranspose, MaxPool1D, MaxPool2D, GlobalMaxPool1D, GlobalMaxPool2D, AveragePooling1D, AveragePooling2D, GlobalAveragePooling1D, GlobalAveragePooling2D, ZeroPadding2D, RepeatVector, Resizing, Rescaling, RandomContrast, RandomCrop, RandomFlip, RandomRotation, RandomTranslation, RandomZoom, RandomWidth, RandomHeight, RandomBrightness
from tensorflow.keras.utils import get_file, to_categorical, plot_model, image_dataset_from_directory
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from tensorflow.keras.optimizers import SGD, Adagrad, Adam
from tensorflow.keras.optimizers.schedules import LearningRateSchedule
# MeanSquaredError, MeanAbsoluteError, MeanAbsolutePercentageError, BinaryCrossentropy, CategoricalCrossentropy, SparseCategoricalCrossentropy, CosineSimilarity
from tensorflow.keras import losses
# MeanSquaredError, RootMeanSquaredError, MeanAbsoluteError, MeanAbsolutePercentageError, BinaryCrossentropy, CategoricalCrossentropy, SparseCategoricalCrossentropy, TopKCategoricalAccuracy, SparseTopKCategoricalAccuracy, CosineSimilarity
from tensorflow.keras import metrics
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.activations import linear, sigmoid, relu
from tensorflow.keras.initializers import RandomNormal, glorot_uniform, he_uniform, Constant
```
## PyTorch
```python
import torch
from torch.nn import Module, Linear, Dropout, Conv1d, Conv2d, ConvTranspose1d, ConvTranspose2d, MaxPool1d, MaxPool2d, AvgPool1d, AvgPool2d, CrossEntropyLoss()
import torch.nn.functional as F
from torch.optim import SGD, RMSprop, Adagrad, Adam
from torch.utils.data import Dataset, DataLoader

import torchvision
import torchvision.dataset
import torchvision.transforms as transforms
```

# `torchvision`
```python
import torch
import torchvision
from torchvision import models
import torchvision.transformers as T
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

## Determine Vocabulary Size
```python
tokenizer = Tokenizer()
tokenizer.fit_on_texts(tr_X)
word2idx = tokenizer.word_index
cnts = sorted(tokenizer.word_counts.values(), reverse=True)
ratio = 0.99
for vocab_size, value in enumerate(np.cumsum(cnts)/np.sum(cnts)):
    if value >= ratio:
        break
print(f"{vocab_size:,}개의 단어로 전체 data의 {ratio:.0%}를 표현할 수 있습니다.")
print(f"{len(word2idx):,}개의 단어 중 {vocab_size/len(word2idx):.1%}에 해당합니다.")
```
## Determine Sequence Length
- Reference: https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/text/Tokenizer
```python
# `num_words`: The maximum number of words to keep, based on word frequency. Only the most common `num_words - 1` words will be kept.
# `filters`: A string where each element is a character that will be filtered from the texts. The default is all punctuation, plus tabs and line breaks, minus the `'` character.
# `oov_token`: If given, it will be added to `word_index` and used to replace out-of-vocabulary words during `texts_to_sequence()` calls
tokenizer = Tokenizer(num_words=vocab_size + 2, oov_token="UNK")
tokenizer.fit_on_texts(train_text)
word2idx = tokenizer.word_index
word2cnt = dict(sorted(tokenizer.word_counts.items(), key=lambda x:x[1], reverse=True))

X_tr = tokenizer.texts_to_sequences(train_text)
X_te = tokenizer.texts_to_sequences(test_text)

lens = sorted([len(doc) for doc in X_tr])
ratio = 0.99
max_len = int(np.quantile(lens, 0.99))
print(f"길이가 가장 긴 문장의 길이는 {np.max(lens)}이고 길이가 {max_len} 이하인 문장이 전체의 {ratio:.0%}를 차지합니다.")
```
## Padding
```python
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences

tr_X = pad_sequences(tr_X, padding="post", maxlen=max_len)
tr_y = pad_sequences(tr_y, padding="post", maxlen=max_len)

# tr_X = to_categorical(tr_X)
# tr_y = to_categorical(tr_y)
```