# variance_inflation_factor
```python
from statsmodels.stats.outliers_influence import variance_inflation_factor
```
```python
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(data_for_corr.values, i) for i in range(data_for_corr.shape[1])]
vif["features"] = data_for_corr.columns
```
