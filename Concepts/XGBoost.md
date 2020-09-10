* [https://bcho.tistory.com/1354](https://bcho.tistory.com/1354)
* XGBoost regression 개념 : [https://www.datacamp.com/community/tutorials/xgboost-in-python](https://www.datacamp.com/community/tutorials/xgboost-in-python)
* [https://xgboost.readthedocs.io/en/latest/parameter.html](https://xgboost.readthedocs.io/en/latest/parameter.html)
* [https://brunch.co.kr/@snobberys/137](https://brunch.co.kr/@snobberys/137)
* [https://www.datacamp.com/community/tutorials/xgboost-in-python](https://www.datacamp.com/community/tutorials/xgboost-in-python)
# algorithm
- 출처 : https://www.youtube.com/watch?v=OtD8wVaFm6E
- 출처 : https://www.youtube.com/watch?v=8b1JEDvenQU&t=3s
- 출처 : https://www.youtube.com/watch?v=ZVFeW798-2I
- 출처 : https://www.youtube.com/watch?v=oRrKeUCEbq8
### 공식 문서
* [https://xgboost.readthedocs.io/en/latest/parameter.html](https://xgboost.readthedocs.io/en/latest/parameter.html)
## early_stopping
- 출처 : https://xgboost.readthedocs.io/en/latest/python/python_intro.html
- If there’s more than one metric in the eval_metric parameter given in params, the last metric will be used for early stopping.
## customized objective/metric function
- 출처 : https://xgboost.readthedocs.io/en/latest/tutorials/custom_metric_obj.html
# Python API reference
- 출처 : https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.train
# hyperparameters tunning

* 출처 : [https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/](https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/)

### booster [default=gbtree]

* Select the type of model to run at each iteration. It has 2 options:
gbtree: tree-based models
gblinear: linear models

### silent [default=0]:

* Silent mode is activated is set to 1, i.e. no running messages will be printed. It’s generally good to keep it 0 as the messages might help in understanding the model.

### eta [default=0.3] == learning_rate

* Analogous to learning rate in GBM
* Makes the model more robust by shrinking the weights on each step
* Typical final values to be used: 0.01-0.2

### min\_child\_weight [default=1]

* Used to control over-fitting. **Higher values prevent a model from learning relations which might be highly specific to the particular sample selected for a tree.**
* Too high values can lead to under-fitting hence, **it should be tuned using CV.**

### max\_depth [default=6]

* Used to control over-fitting as higher depth will allow model to learn relations very specific to a particular sample.
* **Should be tuned using CV.**

### max\_leaf\_nodes

* Can be defined in place of max\_depth. Since binary trees are created, a depth of ‘n’ would produce a maximum of 2^n leaves. If this is defined, GBM will ignore max\_depth.

### gamma [default=0]

* A node is split only when the resulting split gives a positive reduction in the loss function. Gamma specifies the minimum loss reduction required to make a split.
* Makes the algorithm conservative. The values can vary depending on the loss function and **should be tuned**.

### max\_delta\_step [default=0]

* In maximum delta step we allow each tree’s weight estimation to be. If the value is set to 0, it means there is no constraint. If it is set to a positive value, it can help making the update step more conservative.
* Usually this parameter is not needed, but it might help in logistic regression when class is extremely imbalanced.
This is generally not used but you can explore further if you wish.

### subsample [default=1]

* Same as the subsample of GBM. Denotes the fraction of observations to be randomly sampled for each tree.
* **Lower values make the algorithm more conservative and prevents overfitting but too small values might lead to under-fitting.**
* Typical values: 0.5-1

### colsample_bytree [default=1]
- Similar to max_features in GBM. Denotes the fraction of columns to be randomly sampled for each tree.
Typical values: 0.5-1
### colsample_bylevel [default=1]
- Denotes the subsample ratio of columns for each split, in each level. I don’t use this often because subsample and colsample_bytree will do the job for you. but you can explore further if you feel so.
### lambda [default=1]==reg_lambda
- L2 regularization term on weights (analogous to Ridge regression) .This is used to handle the regularization part of XGBoost. Though many data scientists don’t use it often, **it should be explored to reduce overfitting.**
### alpha [default=0]==reg_alpha
- L1 regularization term on weight (analogous to Lasso regression). **Can be used in case of very high dimensionality so that the algorithm runs faster when implemented.**
### scale_pos_weight [default=1]
- **A value greater than 0 should be used in case of high class imbalance as it helps in faster convergence.**
### n_jobs
- int Number of parallel threads used to run xgboost.

### objective?

* Additional parameters for Dart Booster (booster=dart)
Using predict() with DART booster
If the booster object is DART type, predict() will perform dropouts, i.e. only some of the trees will be evaluated. This will produce incorrect results if data is not the training data. To obtain correct results on test sets, set ntree\_limit to a nonzero value, e.g.
preds = bst.predict(dtest, ntree\_limit=num\_round)

### General Approach for Parameter Tuning

1. Choose a relatively **high learning rate**. Generally a learning rate of 0.1 works but somewhere between 0.05 to 0.3 should work for different problems. Determine the optimum **number of trees for this learning rat**e. XGBoost has a very useful function called as “cv” which performs cross-validation at each boosting iteration and thus returns the optimum number of trees required.
2. Tune tree-specific parameters (**max\_depth, min\_child\_weight, gamma, subsample, colsample\_bytree**) for decided learning rate and number of trees. Note that we can choose different parameters to define a tree and I’ll take up an example here.
3. Tune **regularization parameters(lambda, alpha)** for xgboost which can help reduce model complexity and enhance performance.
4. **Lower the learning rate and decide the optimal parameters**.
Let us look at a more detailed step by step approach.
