```python
import numpy as np
```
# np
## np.set_printoptions()
```python
np.set_printoptions(precision=3)
```
## np.arange
```python
np.arange(5, 101, 5)
```
## np.ones()
```python
np.ones((2, 3, 4))
```
## np.zeros()
## np.empty()
## np.full()
```python
np.full((2, 3, 4), 7))
```
## np.eye()
```python
np.eye(4)
```
## np.ones_like(), np.zeros_like()
```python
np.ones_like(arr)
```
## np.linspace()
```python
np.linspace(-5, 5, 100)
```
## np.any()
```python
np.any(arr>0)
```
## np.all()
## np.where()
```python
np.where(arr>0, arr, 0)
```
## np.isin()
```python
data[np.isin(data["houses"], list)]
```
## np.transpose()
## np.swapaxes()
```python
feature_maps = np.transpose(conv2d, (3, 1, 2, 0))
```
```python
feature_maps = np.swapaxes(conv2d, 0, 3)
```
## np.random
### np.random.seed()
```python
np.random.seed(23)
```
### np.random.rand()
```python
np.random.rand(2, 3, 4)
```
### np.random.randint()
```python
np.random.randint(1, 100, size=(2, 3, 4))
```
### np.random.choice()
```python
np.random.choice(arr(1d), size=(2, 3, 4), replace=False)
```
### np.random.normal()
```python
np.random.normal(mean, std, size=(3, 4))
```
## np.digitize()
```python
bins=range(0, 55000, 5000)
data["price_range"]=np.digitize(data["money"], bins)
```
## np.isnan()
## np.nanmean()
## np.sort()
## np.expand_dims()
```python
train_data = np.expand_dims(train_data, axis=-1)
```
## np.unique()
## np.linalg
### np.linalg.norm()
```python
np.linalg.norm(x, axis=1, ord=2)
```
- ord=1 : L1 normalization.
- ord=2 : L2 normalization.
## np.sqrt()
## np.power()
## np.fill_diagonal()
```python
np.fill_diagonal(cos_sim_item, 0)
```

# arr
## arr.ravel()
```python
arr.ravel(order="F")
```
- order="C" : row 기준
- order="F" : column 기준
##  arr.flatten()
- 복사본 반환
## arr.reshape()
- tuple로 차원 지정
## arr.T
## arr.shape