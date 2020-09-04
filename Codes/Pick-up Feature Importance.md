```python
import numpy as np
import pandas as pd
import seaborn as sb
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.inspection import permutation_importance
from sklearn.metrics import fbeta_score, make_scorer
import random
import matplotlib as mpl
import matplotlib.pyplot as plt
import time
import os
import pickle
from itertools import combinations
from tqdm.notebook import tqdm
import platform

path = "C:/Windows/Fonts/malgun.ttf"
if platform.system()=="Darwin":
    mpl.rc("font", family="AppleGothic")
elif platform.system()=="Windows":
    font_name = mpl.font_manager.FontProperties(fname=path).get_name()
    mpl.rc('font', family=font_name)
```

```python
num = 2

def custom_obj(observ, pred):
    per_residual = (pred-observ)/observ*100
    grad = np.power((per_residual), num-1).astype(np.float64)
    hess = np.power((per_residual), num-2).astype(np.float64)
    return np.where(per_residual>=0, grad, grad), np.where(per_residual>=0, hess, hess)

def custom_met(pred, label):
    observ = label.get_label()
    per_residual = (pred-observ)/observ*100
    return "error", np.power(np.sum(np.power(per_residual, num)/len(observ)), 1/num)

model = xgb.XGBRegressor(booster="gbtree", max_delta_step=0, importance_type="gain", missing=None, seed=None, base_score=0.5, verbosity=1, disable_default_eval_metric=1)

model.objective = custom_obj
model.n_jobs = -1

model.learning_rate = 30
model.n_estimators = 3000
model.max_depth = 3
model.min_child_weight = 10
model.gamma = 4
model.subsample = 0.5
model.colsample_bytree = 1
model.reg_lambda = 5
```
```python
train_base, val_base = train_test_split(data_kor, train_size=0.8, shuffle=True, random_state=1)

train_X_base = train_base.drop(["feature1"], axis=1)
train_y_base = train_base[["feature1"]]

val_X_base = val_base.drop(["feature1"], axis=1)
val_y_base = val_base[["feature1"]]

model.fit(train_X_base, train_y_base, eval_set=[(train_X_base, train_y_base), (val_X_base, val_y_base)], eval_metric=custom_met, early_stopping_rounds=50, verbose=False)

val_base["pred"] = model.predict(val_X_base)
val_base["error"] = (val_base["pred"]-val_base["feature1"])/val_base["feature1"]*100
base_error = abs(val_base["error"]).mean().round(1)
print("base error :", base_error)

feature_list = train_X_base.columns.tolist()
result = []
imps = []
for feat in tqdm(feature_list):
    for _ in range(1):
        train_new = train_base[[feat, "feature1"]]
        train_X_new = train_new.drop(["feature1"], axis=1)
        train_y_new = train_new[["feature1"]]
        
        val_new = val_base[[feat, "feature1"]]
        val_X_new = val_new.drop(["feature1"], axis=1)
        val_y_new = val_new[["feature1"]]

        model.fit(train_X_new, train_y_new, eval_set=[(train_X_new, train_y_new), (val_X_new, val_y_new)], eval_metric=custom_met, early_stopping_rounds=50, verbose=True)
        
        val_new["pred"] = model.predict(val_X_new)
        val_new["error"] = (val_new["pred"] - val_new["feature1"])/val_new["feature1"] * 100
        new_error = abs(val_new["error"]).mean().round(1)
        
        imp = new_error/base_error
        imps.append(imp)
        
        result.append([new_error, imp, feat])
        imp_result = pd.DataFrame(result, columns=["new_error", "imp", "feat"])
        imp_result = imp_result.sort_values(by="imp", ascending=True)
        print(feat, imp)
        
imp_result = imp_result.reset_index(drop=True)
imp_result["imp_rev"] = 1/imp_result["imp"]
```
```python
print("base_error :", round(base_error, 1))

fig, ax = plt.subplots(figsize=(9, len(imp_result)/3))
ax.tick_params(axis="y", labelsize=15)

sb.barplot(ax=ax, data=imp_result, x="imp_rev", y=imp_result.index, color="cornflowerblue", edgecolor="black", orient="h")

for idx in imp_result.index:
    ax.text(y=idx+0.05, x=imp_result.loc[idx, "imp_rev"]+0.002, s=f"{imp_result.loc[idx, 'feat']}", ha="left", va="center", fontsize=13)

ax.set_xlabel("특성 중요도", size=15)

ax.tick_params(axis="y", labelsize=0)
ax.set_xlim([0, 0.65])
ax.grid(axis="x", alpha=0.3, linestyle="--", linewidth=2)

fig.tight_layout()
fig.savefig("특성 중요도_200903.png", bbox_inches="tight")
```
