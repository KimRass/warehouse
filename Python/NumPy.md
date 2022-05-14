```python
import numpy as np
```
# `np.set_printoptions([edgeitems], [infstr], [linewidth], [nanstr], [precision], [suppress], [threshold], [formatter])`
# `Array.size`
# `Array.astype()`
```python
x_train = x_train.astype("float32")
```
# `np.inf`
# `np.load("file_name.npy")`
# `np.logical_and()`, `np.logical_or()`
# `np.array_equal()`
# `np.linspace()`
# `np.meshgrid()`
# `np.isin()`
# `np.digitize()`
```python		
bins=range(0, 55000, 5000)
data["price_range"]=np.digitize(data["money"], bins)
```
# `np.reshape(newshape)`, `Array.reshape(newshape)`
# `np.unique()`
```python
items, counts = np.unique(intersected_movie_ids, return_counts=True)
```
# `np.fromfile()`
- `count`: Number of items to read. `-1` means all items (i.e., the complete file).

# Array 생성 함수
# `np.full(shape, fill_value)`
# `np.eye(a)`
- `a`: (Array-like)
# `np.ones_like()`, `np.zeros_like()`
# `np.zeros(shape)`, `np.ones(shape)`
# `np.arange([start], stop, [step])`
- `start`: (default 0) Start of interval. The interval includes this value.
- `stop`: End of interval. The interval does not include this value, except in some cases where step is not an integer and floating point round-off affects the length of out.
- `step`: (default 1)
# `np.split(ary, indices_or_sections, [axis=0])`
- `indices_or_sections`
	- (int) `ary` will be divided into `indices_or_sections` equal arrays along `axis`. If such a split is not possible, an error is raised.
# `np.sqrt()`
# `np.power()`
# `np.exp()`
# `np.isnan()`
# `np.nanmean()`
# `np.sort()`
# `np.any()`
# `np.all()`
# `np.where()`
```python
np.min(np.where(cumsum >= np.cumsum(cnts)[-1]*ratio))
```
# `np.tanh()`
# `np.shape()`
# `np.empty()`

# Functions for Manipulating Matrices
# `np.add.outer()`, `np.multiply.outer()`
```python
euc_sim_item = 1 / (1 + np.sqrt(np.add.outer(square, square) - 2*dot))
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
# Linear Algebra
## `np.linalg.norm()`
```python
np.linalg.norm(x, axis=1, ord=2)
```
- `ord=1`: L1 normalization.
- `ord=2`: L2 normalization.
# 모양 변화
## `np.expand_dims()`
```python
np.expand_dims(mh_df.values, axis=1)
```
# `np.einsum()`
# `np.concatenate()`
# `np.stack()`
# `np.delete()`
# `np.argmax()`
# `np.swapaxes()`
# `np.max()`, `np.min()`
# `np.maximum()`, `np.minimum()`
- Element-wise minimum(maximum) of Array elements.
# `np.cumsum()`
- `axis`
# `np.prod()`
- Return the product of Array elements over a given axis.
- `axis`
# `np.quantile()`
```python
lens = sorted([len(doc) for doc in train_X])
ratio = 0.99
max_len = int(np.quantile(lens, ratio))
print(f"가장 긴 문장의 길이는 {np.max(lens)}입니다.")
print(f"길이가 {max_len} 이하인 문장이 전체의 {ratio:.0%}를 차지합니다.")
```
# `Array.ravel()`
```python
arr.ravel(order="F")	
```
- `order="C"` : row 기준
- `order="F"` : column 기준
# `Array.flatten()`
- 복사본 반환
# `Array.transpose()`
```python
conv_weights = np.fromfile(f, dtype=np.float32, count=np.prod(conv_shape)).reshape(conv_shape).transpose((2, 3, 1, 0))
```
