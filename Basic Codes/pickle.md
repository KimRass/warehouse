# pickle
```python
import pickle as pk
```
## pk.dump()
```python
with open("filename.pk", 'wb') as f:
    pk.dump(list, f)
```
## pk.load()
```python
with open("filename.pk", "rb") as f:
    data = pk.load(f)
```
- 한 줄씩 load
