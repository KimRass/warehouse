```python
import numpy as np
```
# Print Options
```python
np.set_printoptions([edgeitems], [infstr], [linewidth], [nanstr], [precision], [suppress], [threshold], [formatter])
```

# Create Array
## `np.full(shape, fill_value)`
## `np.eye(a)`
- `a`: (Array-like)
## `np.ones_like()`, `np.zeros_like()`
## `np.zeros(shape)`, `np.ones(shape)`
## `np.arange([start], stop, [step])`
- `start`: (default 0) Start of interval. The interval includes this value.
- `stop`: End of interval. The interval does not include this value, except in some cases where step is not an integer and floating point round-off affects the length of out.
- `step`: (default 1)
## `np.linspace()`
## `np.meshgrid()`

# Change Array Shape
## Insert New Axis
```python
# Insert a new axis that will appear at the `axis` position in the expanded array shape.
np.expand_dims(a, axis)
```
## Remove Axes of Length One
```python
np.squeeze(a, axis=None)
```
## `Array.ravel()`
```python
# `order="C"`: row 기준
# `order="F"`: column 기준
arr.ravel(order="F")	
```
## Flatten
```python
# In-place function
Array.flatten()
```
## Transpose
```python
# Example
conv_weights = np.fromfile(f, dtype=np.float32, count=np.prod(conv_shape)).reshape(conv_shape).transpose((2, 3, 1, 0))
```

# Diagonal
## `np.fill_diagonal()`
## `np.diag(v, [k=0])`
- If `v` is a 2-D array, returns a copy of its `k`-th diagonal.
- If `v` is a 1-D array, returns a 2-D array with `v` on the `k`-th diagonal.
## `np.triu(m, k=0)`
- Upper triangle of an array.
## `tf.linalg.band_part()`
- `tf.linalg.band_part(input, 0, -1)`: Upper triangular part.
- `tf.linalg.band_part(input, -1, 0)`: Lower triangular part.
- `tf.linalg.band_part(input, 0, 0)`: Diagonal.

# Statistics
## Mean without Missing Values
```python
np.nanmean()
```

# Linear Algebra
## Normalization
```python
# `ord=1`: L1 normalization.
# `ord=2`: L2 normalization.
np.linalg.norm(x, axis=1, ord=2)
```

# Mathematics
## `np.sqrt()`
## `np.power()`
## `np.exp()`
## `np.tanh()`
## `np.inf`

# Logical Operators
## True If Two Arrays Have The Same Shape and Elements
```python
# Example
np.array_equal(temp, local_max)
```
## `np.isnan()`
## `np.any()`
## `np.all()`
## `np.where()`
```python
# 첫 번째 인자를 만족하면 두 번째 인자에서 값을 가져오고, 만족하지 못하면 세 번째 인자에서 값을 가져옵니다.
# Example
temp1 = np.where(
    (segmap_char_mask == idx), segmap_mask, 0
)
```
## `np.logical_and()`, `np.logical_or()`
## `np.array_equal()`

# Functions for Manipulating Matrices
## `np.add.outer()`, `np.multiply.outer()`
```python
euc_sim_item = 1 / (1 + np.sqrt(np.add.outer(square, square) - 2*dot))
```

# `np.quantile()`
```python
lens = sorted([len(doc) for doc in train_X])
ratio = 0.99
max_len = int(np.quantile(lens, ratio))
print(f"가장 긴 문장의 길이는 {np.max(lens)}입니다.")
print(f"길이가 {max_len} 이하인 문장이 전체의 {ratio:.0%}를 차지합니다.")
```

# Load .npy File
```python
np.load("....npy")
```


# `np.isin()`
# `np.digitize()`
```python		
bins=range(0, 55000, 5000)
data["price_range"]=np.digitize(data["money"], bins)
```
# `np.reshape(newshape)`, `Array.reshape(newshape)`

# Number of Elements
```python
# Example
items, counts = np.unique(intersected_movie_ids, return_counts=True)
```

# `np.fromfile()`
- `count`: Number of items to read. `-1` means all items (i.e., the complete file).

## Clip
```python
np.clip(a, a_min, a_max)
```