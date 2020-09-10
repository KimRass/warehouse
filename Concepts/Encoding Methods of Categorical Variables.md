### One-hot encoding
- One-Hot encoding should not be performed if the number of categories are high. This would result in a sparse data.
- Decision trees does not require doing one-hot encoding. Since xgboost, AFAIK, is a boosting of decision trees, I assume the encoding is not required.
- 피처내 값들이 서로 분리 되어있기 때문에, 우리가 모를 수 있는 어떤 관계나 영향을 주지 않는다.
- features 내 값의 종류가 많을 경우(High Cardinaliry), 매우 많은 Feature 들을 만들어 낸다. 이는, 모델 훈련의 속도를 낮추고 훈련에 더 많은 데이터를 필요로 하게 한다.(차원의 저주 문제)
- 단순히 0과 1로만 결과를 내어 큰 정보이득 없이 Tree 의 depth 만 깊게 만든다. 중요한건, Tree Depth 를 증가시키는 것에 비해, 2가지 경우로만 트리를 만들어 나간다는 것이다.
- Random Forest 와 같이, 일부 Feature 만 Sampling 하여 트리를 만들어나가는 경우, One-hot Feature 로 생성된 Feature 의 수가 많기 때문에 이 features가 다른 features보다 더 많이 쓰인다.
### Label encoding
- just mapping randomly to ints often works very well, especially if there aren’t to many.
- Another way is to map them to their frequency in the dataset. If you’re afraid of collisions, map to counts + a fixed (per category) small random perturbation.
- 값의 크기가 의미가 없는데 부여되어서 bias가 생길 수 있음.
- One major issue with this approach is there is no relation or order between these classes, but the algorithm might consider them as some order, or there is some relationship.
- pandas factorize는 NAN값을 자동으로 -1로 채우는데 반해 scikitlearn Label Encoder는 에러를 발생.
- Categorical 값들은 순서개념이 없음에도, 모델은 순서개념을 전제하여 학습하게 된다.
### Ordinal encoding
- We do Ordinal encoding to ensure the encoding of variables retains the ordinal nature of the variable. This is reasonable only for ordinal variables, as I mentioned at the beginning of this article. This encoding looks almost similar to Label Encoding but slightly different as Label coding would not consider whether variable is ordinal or not and it will assign sequence of integers
- If we consider in the temperature scale as the order, then the ordinal value should from cold to “Very Hot. “ Ordinal encoding will assign values as ( Cold(1) <Warm(2)<Hot(3)<”Very Hot(4)). Usually, we Ordinal Encoding is done starting from 1.
### Mean encoding
- 일반적인 방법인 category를 label encoding으로 얻은 숫자는 머신러닝 모델이 오해하기 쉽다.
- target value의 더 큰 값에 해당하는 범주가 더 큰 숫자로 인코딩 되게 하는 것이다.
- Target encoding can lead to data leakage. To avoid that, fit the encoder only on your train set, and then use its transform method to encode categories in both train and test sets.(train에 encoding fitting 후 그것을 test와 train 적용)
- 특히 Gradient Boosting Tree 계열에 많이 쓰이고 있다.
- 오버피팅의 문제가 있다. 하나는 Data Leakage 문제인데, 사실 훈련 데이터에는 예측 값에 대한 정보가 전혀 들어가면 안되는게 일반적이다. 그런데, Mean encoding 과정을 보면, 사실 encoding 된 값에는 예측 값에 대한 정보가 포함되어있다. 이러한 문제는 모델을 Training set 에만 오버피팅 되도록 만든다.
- 다른 하나는 하나의 label 값의 대표값을 trainset의 하나의 mean 으로만 사용한다는 점이다. 만약 testset 에 해당 label 값의 통계적인 분포가 trainset 과 다르다면, 오버피팅이 일어날 수 밖에 없다. 특히, 이런 상황은 Categorical 변수 내 Label의 분포가 매우 극단적인 경우에 발생한다. 예를 들어 Trainset 에는 남자가 100명, 여자가 5명이고, Testset 에는 50명, 50명이라고 하자. 우리는 Trainset 으로 Mean encoding 할텐데, 여자 5명의 평균값이 Testset 의 여자 50명을 대표할 수 있을까? 어려울 수 밖에 없다.
- Usually, Mean encoding is notorious for over-fitting; thus, a regularization with cross-validation or some other approach is a must on most occasions.
### frequency encoding
- 값 분포에 대한 정보가 잘 보존. 값의 빈도가 타겟과 연관이 있으면 아주 유용.
-Encoding한 Category끼리 같은 Frequency를 가진다면 Feature로 사용하지 않아도 됨.(?)
### grouping
- you’ll need to do some exploratory data analysis to do some feature engineering like grouping categories or tactfully assigning appropriate integer values to match the relation of the variable with the output.
- if you know something about the categories you can perhaps group them and add an additional feature group id then order them by group id.
### 참고자료
- https://homeproject.tistory.com/4
- http://blog.naver.com/PostView.nhn?blogId=choco_9966&logNo=221374544814&parentCategoryNo=&categoryNo=77&viewDate=&isShowPopularPosts=false&from=postView
- https://dailyheumsi.tistory.com/120
- https://towardsdatascience.com/all-about-categorical-variable-encoding-305f3361fd02
