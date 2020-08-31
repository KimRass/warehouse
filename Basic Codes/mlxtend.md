# mlxtend
```python
import mlxtend
```
## mlxtend.preprocessing
### TransactionEncoder
```python
from mlxtend.preprocessing import TransactionEncoder
```
```python
te = TransactionEncoder()
te.fit_transform([{0, 1, 2}, {0, 1, 4}, {1, 8}, {3, 4, 10}, {2, 3}])
```
#### te.columns_
## mlxtend.frequent_patterns
### apriori
```python
from mlxtend.frequent_patterns import apriori
```
```python
freq_sets_df = apriori(baskets_df_over5000.sample(frac=0.05), min_support=0.01, max_len=2, use_colnames=True, verbose=1)
```
### association_rules
```python
from mlxtend.frequent_patterns import association_rules
```
```python
asso_rules = association_rules(sups, metric="support", min_threshold=0.01)
```
