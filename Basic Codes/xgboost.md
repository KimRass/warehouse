# xgboost
```python
import xgboost as xgb
```
## xgb.XGBRegressor()
```python
model = xgb.XGBRegressor(base_score=0.5, booster="gbtree", colsample_bylevel=1, colsample_bynode=1, colsample_bytree=1, gamma=0, importance_type='gain', learning_rate=0.1, max_delta_step=0, max_depth=max_depth, min_child_weight=1, missing=-1, n_estimators=n_estimators, n_jobs=1, nthread=None, objective="reg:linear", random_state=1, reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None, silent=None, subsample=1, verbosity=1, warning="ignore")

model.fit(train_X, train_y, eval_set=[(train_X, train_y), (val_X, val_y)], early_stopping_rounds=50, verbose=True)
```
## xgb.to_graphviz()
```python
def plot_tree(model, filename, rankdir="UT"):
    import os
    gviz = xgb.to_graphviz(model, num_trees = model.best_iteration, rankdir=rankdir)
    _, file_extension = os.path.splitext(filename)
    format = file_extension.strip(".").lower()
    data = gviz.pipe(format=format)
    full_filename = filename
    with open(full_filename, "wb") as f:
        f.write(data)
```
## xgb.plot_importance()
```python
plt.rcParams["figure.figsize"]=(15, len(train_X.columns)/4)
plt.rcParams["font.size"]=15
...
xgb.plot_importance(model, height=len(train_X.columns)/50)
```
