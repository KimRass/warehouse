# dataset
- 출처 : https://tykimos.github.io/2017/03/25/Dataset_and_Fit_Talk/

# splitting dataset
- 출처 : https://davinci-ai.tistory.com/18
- 데이터가 독립적이고 동일한 분포를 가진 경우
KFold, RepeatedKFold, LeaveOneOut(LOO), LeavePOutLeaveOneOut(LPO)
- 동일한 분포가 아닌 경우
StratifiedKFold, RepeatedStratifiedKFold, StratifiedShuffleSplit
- 그룹화된 데이터의 경우
GroupKFold, LeaveOneGroupOut, LeavePGroupsOut, GroupShuffleSplit
- 시계열 데이터의 경우
TimeSeriesSplit


출처: https://davinci-ai.tistory.com/18 [DAVINCI - AI]
- 출처 : https://towardsdatascience.com/validating-your-machine-learning-model-25b4c8643fb7
- 예를 들어 제목, 본문, 보낸 사람의 이메일 주소를 특성을 사용하여 스팸 메일을 가려내는 모델이 있다고 가정해 보겠습니다. 데이터를 80:20 비율로 학습 세트와 평가 세트로 배분했습니다. 학습 후에 모델은 학습 세트와 평가 세트 모두에서 99%의 정확성을 보입니다. 평가 세트에서는 정확성이 이보다 낮아야 하므로, 데이터를 다시 살펴본 결과 평가 세트의 예 중 다수가 학습 세트의 예와 중복되는 것으로 나타났습니다. 데이터를 분할하기 전에 입력 데이터베이스에서 동일한 스팸 메일의 중복 항목을 솎아내지 않았던 것입니다. 따라서 테스트 데이터 중 일부가 의도치 않게 학습에 사용되어, 모델이 새 데이터로 얼마나 효과적으로 일반화되는지 정확히 측정할 수 없게 되었습니다.
- 출처 : https://developers.google.com/machine-learning/crash-course/training-and-test-sets/splitting-data
## random sampling
- The benefit of this approach is that we can see how the model reacts to previously unseen data.
However, what if one subset of our data only have people of a certain age or income levels? This is typically referred to as a sampling bias:
Sampling bias is systematic error due to a non-random sample of a population, causing some members of the population to be less likely to be included than others, resulting in a biased sample.
- If only use a train/test split, then I would advise comparing the distributions of your train and test sets. If they differ significantly, then you might run into problems with generalization. Use Facets to easily compare their distributions.
## holdout set
- When optimizing the hyperparameters of your model, you might overfit your model if you were to optimize using the train/test split.
Why? Because the model searches for the hyperparameters that fit the specific train/test you made.
## k-fold cv
- We typically choose either i=5 or k=10 as they find a nice balance between computational complexity and validation accuracy:
## leave-one-out cv
- This variant is identical to k-fold CV when k = n (number of observations).
## leave-one-group-out cv
## nested cv
- When you are optimizing the hyperparameters of your model and you use the same k-Fold CV strategy to tune the model and evaluate performance you run the risk of overfitting. You do not want to estimate the accuracy of your model on the same split that you found the best hyperparameters for.
- Instead, we use a Nested Cross-Validation strategy allowing to separate the hyperparameter tuning step from the error estimation step.
- The inner loop for hyperparameter tuning and
the outer loop for estimating accuracy.
- You are free to select the cross-validation approaches you use in the inner and outer loops. For example, you can use Leave-one-group-out for both the inner and outer loops if you want to split by specific groups.
## time series cv
- Overfitting would be a major concern since your training data could contain information from the future. It is important that all your training data happens before your test data.
One way of validating time series data is by using k-fold CV and making sure that in each fold the training data takes place before the test data.
