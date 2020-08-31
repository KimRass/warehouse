# os
```python
import os
```
## os.getcwd()
```python
cur_dir = os.getcwd()
```
## os.makedirs()
```python
os.makedirs(ckpt_dir, exist_ok=True)
```
## os.path
### os.path.join()
```python
checkpoint_dir = os.path.join(cur_dir, ckpt_dir_name, model_dir_name)
```
### os.path.exists()
```python
if os.path.exists("C:/Users/5CG7092POZ/train_data.json"):
```
