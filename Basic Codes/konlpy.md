# konlpy
## Okt()
```python
from konlpy.tag import Okt
```
```python
okt = Okt()
```
### okt.add_dictionary
```python
okt.add_dictionary(["대금", "지급", "근재", "사배책", "건설", "기계"], "Noun")
```
### okt.morphs()
```python
codes_tokenized = [" ".join(okt.morphs(code)) for code in codes]
items_tokenized = [" ".join(okt.morphs(item)) for item in items]
```
