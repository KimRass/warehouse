# Header
- Reference: https://www.webopedia.com/definitions/header/
- In many disciplines of computer science, a header is a unit of information that precedes a data object.

# Dataset & Data Set
- Reference: https://english.stackexchange.com/questions/2120/which-is-correct-dataset-or-data-set
- Dataset for certain datasets
- Data set for any set for data in general.

# Data Density (or Sparsity)
- Reference: https://datascience.foundation/discussion/data-science/data-sparsity
- In a database, sparsity and density describe the number of cells in a table that are empty (sparsity) and that contain information (density), though sparse cells are not always technically empty—they often contain a "0” digit.
## Sparse Matrix & Dense Matrix
- Reference: https://en.wikipedia.org/wiki/Sparse_matrix
- ***A sparse matrix or sparse array is a matrix in which most of the elements are zero.*** *There is no strict definition regarding the proportion of zero-value elements for a matrix to qualify as sparse but a common criterion is that the number of non-zero elements is roughly equal to the number of rows or columns.* ***By contrast, if most of the elements are non-zero, the matrix is considered dense. The number of zero-valued elements divided by the total number of elements is sometimes referred to as the sparsity of the matrix.***
### Compressed Sparse Row (CSR)
- *The compressed sparse row (CSR) or compressed row storage (CRS) or Yale format represents a matrix M by three (one-dimensional) arrays, that respectively contain nonzero values, the extents of rows, and column indices.*
```python
from scipy.sparse import csr_matrix

vals = [2, 4, 3, 4, 1, 1, 2]
rows = [0, 1, 2, 2, 3, 4, 4]
cols = [0, 2, 5, 6, 14, 0, 1]
sparse_mat = csr_matrix((vals,  (rows,  cols)))
dense_mat = sparse_mat.todense()
```

# Prediction & Forecasting
- Reference: https://www.datascienceblog.net/post/machine-learning/forecasting_vs_prediction/#:~:text=Prediction%20is%20concerned%20with%20estimating%20the%20outcomes%20for%20unseen%20data.&text=Forecasting%20is%20a%20sub%2Ddiscipline,we%20consider%20the%20temporal%20dimension.
- *Prediction is concerned with estimating the outcomes for unseen data.*
- *Forecasting is a sub-discipline of prediction in which we are making predictions about the future, on the basis of time-series data.* Thus, the only difference between prediction and forecasting is that we consider the temporal dimension.

# Categories of Variables
- Continuous
- Categorical

# Metadata
- Reference: https://en.wikipedia.org/wiki/Metadata
- Metadata is "data that provides information about other data", but not the content of the data, such as the text of a message or the image itself.

# Embedding
- Reference: https://analyticsindiamag.com/machine-learning-embedding/#:~:text=An%20embedding%20is%20a%20low,of%20a%20high%2Ddimensional%20vector.&text=Embedding%20is%20the%20process%20of,the%20two%20are%20semantically%20similar.
- *Embedding is the process of converting high-dimensional data to low-dimensional data in the form of a vector in such a way that the two are semantically similar.*
- Embeddings of neural networks are advantageous because they can lower the dimensionality of categorical variables and represent them meaningfully in the altered space.
- Reference: https://developers.google.com/machine-learning/crash-course/embeddings/video-lecture
- An embedding is a relatively low-dimensional space into which you can translate high-dimensional vectors. Embeddings make it easier to do machine learning on large inputs like sparse vectors representing words. Ideally, an embedding captures some of the semantics of the input by placing semantically similar inputs close together in the embedding space. An embedding can be learned and reused across models.

# Feature Scaling (Data Normalization)
- Reference: https://en.wikipedia.org/wiki/Feature_scaling
- Feature scaling is a method used to normalize the range of independent variables or features of data. In data processing, it is also known as data normalization and is generally performed during the data preprocessing step.
- ***Since the range of values of raw data varies widely, in some machine learning algorithms, objective functions will not work properly without normalization. For example, many classifiers calculate the distance between two points by the Euclidean distance. If one of the features has a broad range of values, the distance will be governed by this particular feature. Therefore, the range of all features should be normalized so that each feature contributes approximately proportionately to the final distance.***
- ***Another reason why feature scaling is applied is that gradient descent converges much faster with feature scaling than without it.***
- It's also important to apply feature scaling if regularization is used as part of the loss function (so that coefficients are penalized appropriately).
```python
sc.fit()
sc.transform()
sc.fit_transform()
```
## Min-Max Scaling
- Min-max scaling is the simplest method and consists in rescaling the range of features to scale the range in [0, 1].
- `numpy` Implementation
	```python
	x_new = (x - min(X))/(max(X) - min(X))
	```
- Using `from sklearn.preprocessing.MinMaxScaler()`
	```python
	from sklearn.preprocessing import MinMaxScaler

	# `feature_range`: (default `(0, 1)`) Desired range of transformed data.
	sc = MinMaxScaler()
	```
## Standard Scaling
- `numpy` Implementation
	```python
	import numpy as np

	x_new = (x - np.mean(X))/np.std(X)
	```
- Using `sklearn.preprocessing.StandardScaler()`
	```python
	from sklearn.preprocessing import StandardScaler

	sc = StandardScaler()
	...
	mu = sc.mean_
	sigma = sc.scale_
	```
## Robust Scaler
```python
from sklearn.preprocessing import RobustScaler

sc = RobustScaler()
```
## Normalizer
```python
from sklearn.preprocessing import Normalizer

sc = Normalizer()
```

# PCA (Principle Component Analysis)
```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
user_embs_pca = pca.fit_transform(user_embs)
user_embs_pca = pd.DataFrame(user_embs_pca, index=user_embs.index, columns=["x", "y"])
```

# Parameter
## Hyperparameter
- Reference: https://en.wikipedia.org/wiki/Artificial_neural_network
- A hyperparameter is a constant parameter whose value is set before the learning process begins. The values of parameters are derived via learning.

# Learning
- Reference: https://en.wikipedia.org/wiki/Artificial_neural_network
- Learning is the adaptation of the network to better handle a task by considering sample observations. *Learning involves adjusting the weights (and optional thresholds) of the network to improve the accuracy of the result. This is done by minimizing the observed errors. Learning is complete when examining additional observations does not usefully reduce the error rate. Even after learning, the error rate typically does not reach 0.* If after learning, the error rate is too high, the network typically must be redesigned. Practically this is done by defining a cost function that is evaluated periodically during learning. *As long as its output continues to decline, learning continues.*
## Learning Rate
- *The learning rate defines the size of the corrective steps that the model takes to adjust for errors in each observation. A high learning rate shortens the training time, but with lower ultimate accuracy, while a lower learning rate takes longer, but with the potential for greater accuracy. In order to avoid oscillation inside the network such as alternating connection weights, and to improve the rate of convergence, refinements use an adaptive learning rate that increases or decreases as appropriate. The concept of momentum allows the balance between the gradient and the previous change to be weighted such that the weight adjustment depends to some degree on the previous change. A momentum close to 0 emphasizes the gradient, while a value close to 1 emphasizes the last change.*

# Discriminative & Generative Model
- Reference: https://analyticsindiamag.com/what-are-discriminative-generative-models-how-do-they-differ/
- *Discriminative models draw boundaries in the data space, while generative ones model how data is placed throughout the space. Mathematically speaking, a discriminative machine learning trains a model by learning parameters that maximize the conditional probability P(Y|X), but a generative model learns parameters by maximizing the joint probability P(X,Y).*
## Discriminative Model
- ***The discriminative model is used particularly for supervised machine learning. Also called a conditional model, it learns the boundaries between classes or labels in a dataset. It creates new instances using probability estimates and maximum likelihood. However, they are not capable of generating new data points. The ultimate goal of discriminative models is to separate one class from another.***
## Generative Model
- ***Generative models are a class of statistical models that generate new data instances. These models are used in unsupervised machine learning to perform tasks such as probability and likelihood estimation, modelling data points, and distinguishing between classes using these probabilities. Generative models rely on the Bayes theorem to find the joint probability.***

# Convergence
- Reference: https://en.wikipedia.org/wiki/Artificial_neural_network
- ***Models may not consistently converge on a single solution, firstly because local minima may exist, depending on the cost function and the model. Secondly, the optimization method used might not guarantee to converge when it begins far from any local minimum. Thirdly, for sufficiently large data or parameters, some methods become impractical.***

# Batch Normalization
- Reference: https://en.wikipedia.org/wiki/Batch_normalization 
- Batch normalization (also known as batch norm) is a method used to make artificial neural networks faster and more stable through normalization of the layers' inputs by re-centering and re-scaling. It was proposed by Sergey Ioffe and Christian Szegedy in 2015.[1]
While the effect of batch normalization is evident, the reasons behind its effectiveness remain under discussion. It was believed that it can mitigate the problem of internal covariate shift, where parameter initialization and changes in the distribution of the inputs of each layer affect the learning rate of the network.[1] Recently, some scholars have argued that batch normalization does not reduce internal covariate shift, but rather smooths the objective function, which in turn improves the performance.[2] However, at initialization, batch normalization in fact induces severe gradient explosion in deep networks, which is only alleviated by skip connections in residual networks.[3] Others sustain that batch normalization achieves length-direction decoupling, and thereby accelerates neural networks.[4] More recently a normalize gradient clipping technique and smart hyperparameter tuning has been introduced in Normalizer-Free Nets, so called "NF-Nets" which mitigates the need for batch normalization.[5][6

# MLOps
- Reference: https://en.wikipedia.org/wiki/MLOps
- MLOps or ML Ops is a set of practices that aims to deploy and maintain machine learning models in production reliably and efficiently.[1] The word is a compound of "machine learning" and the continuous development practice of DevOps in the software field. Machine learning models are tested and developed in isolated experimental systems. When an algorithm is ready to be launched, MLOps is practiced between Data Scientists, DevOps, and Machine Learning engineers to transition the algorithm to production systems.
- Model Serving
	- Reference: https://medium.com/daria-blog/%EB%AA%A8%EB%8D%B8-%EC%84%9C%EB%B9%99%EC%9D%B4%EB%9E%80-21f970e6cfa5
	- MLOps에서는 특히 새로운 모델을 서빙하는 건 상당히 두려운 일입니다. 이 문제를 해결하기 위해 카나리 모델은 새로운 모델을 완전히 올리기 전 일부 트래픽을 새로운 모델에 흘려보냅니다. 카나리 모델이 어떻게 반응하는지 살펴보고 서빙을 할지 말지 판단하는 데 도와줍니다. 앞서 설명해 드린 모델 컨테이너라는 규격화된 애플리케이션이 있다면 이 안에 모델을 바꾸는 것만으로도 간단히 카나리 모델을 적용할 수 있습니다. 모델 컨테이너가 없다면 카나리 모델을 서빙하는 또 다른 애플리케이션이 담긴 도커 이미지를 반복적으로 빌드 해야 합니다.
	- ![Canary Test](https://miro.medium.com/max/421/1*1AUtwG0MqUEzGlfRndvnnw.png)
	- 위와 같이 카나리 테스트를 적용한다고 하더라도 모델은 시간이 흐르며 성능은 변합니다. 현재 운영 중인 모델이 기준 성능을 만족하지 못하는 상황은 언제든지 발생할 수 있습니다. 이러한 상황이 발생한다면 MLOps가 해야 하는 작업은 크게 (1) 서비스 중지, (2) 특정 모델로 교체입니다. (1) 서비스 중지는 잘못된 예측값을 주느니 차라리 예측하지 않는 것이 좋은 경우입니다. (2) 특정 모델로 교체하는 것을 롤백이라고 합니다.

# Activation Functions
## Sigmoid Function
```python
def sigmoid(x):
    return 1/(1 + np.exp(-x))
```
```python
tf.math.sigmoid()
```
```python
tf.keras.activations.sigmoid(logits)
```
### Derivative of Sigmoid Function
```python
def deriv_sigmoid(x):
	return sigmoid(x)(1 - sigmoid(x))
```
## Softmax
- Reference: https://www.tensorflow.org/api_docs/python/tf/nn/softmax
```python
# TensorFlow
tf.nn.softmax([axis=-1])
# Pytorch
F.softmax(input, [dim=None])
```
```python
# Applies a softmax followed by a logarithm.
F.log_softmax(input, [dim=None])
```
## Hyperbolic Tangent
- Using `tensorflow.nn.tanh()`
	```python
	tf.nn.tanh()
	```
## Categorical Cross-Entropy
## Relu
- Using `tensorflow.nn.relu()`
	```python
	tf.nn.relu
	```
## `tensorflow_addons.optimizers.RectifiedAdam()`
```python
import tensorflow_addons as tfa

tfa.optimizers.RectifiedAdam(lr, total_steps, warmup_proportion, min_lr, epsilon, clipnorm)
```

# Gradient Descent
- Reference: https://en.wikipedia.org/wiki/Gradient_descent
- ***Gradient descent is a first-order iterative optimization algorithm for finding a local minimum of a differentiable function. The idea is to take repeated steps in the opposite direction of the gradient (or approximate gradient) of the function at the current point, because this is the direction of steepest descent.*** Conversely, stepping in the direction of the gradient will lead to a local maximum of that function; the procedure is then known as gradient ascent.
## Vanishing Gradient
- Reference: https://en.wikipedia.org/wiki/Vanishing_gradient_problem
- In machine learning, the vanishing gradient problem is encountered when training artificial neural networks with gradient-based learning methods and backpropagation. In such methods, during each iteration of training each of the neural network's weights receives an update proportional to the partial derivative of the error function with respect to the current weight. *The problem is that in some cases, the gradient will be vanishingly small, effectively preventing the weight from changing its value. In the worst case, this may completely stop the neural network from further training. As one example of the problem cause, traditional activation functions such as the hyperbolic tangent function have gradients in the range (0,1], and backpropagation computes gradients by the chain rule. This has the effect of multiplying `n` of these small numbers to compute gradients of the early layers in an `n`-layer network, meaning that the gradient (error signal) decreases exponentially with `n` while the early layers train very slowly.*
When activation functions are used whose derivatives can take on larger values, one risks encountering the related exploding gradient problem.
- Solutions
	- Residual networks
		- One of the newest and most effective ways to resolve the vanishing gradient problem is with residual neural networks, or ResNets (not to be confused with recurrent neural networks). ResNets refer to neural networks where skip connections or residual connections are part of the network architecture. *These skip connections allow gradient information to pass through the layers, by creating "highways" of information, where the output of a previous layer/activation is added to the output of a deeper layer. This allows information from the earlier parts of the network to be passed to the deeper parts of the network, helping maintain signal propagation even in deeper networks. Skip connections are a critical component of what allowed successful training of deeper neural networks.* ResNets yielded lower training error (and test error) than their shallower counterparts simply by reintroducing outputs from shallower layers in the network to compensate for the vanishing data.
	- Other activation functions
		- Rectifiers such as ReLU suffer less from the vanishing gradient problem, because they only saturate in one direction.
- Reference: https://www.analyticsvidhya.com/blog/2021/06/understanding-resnet-and-analyzing-various-models-on-the-cifar-10-dataset/
- While backpropagating, we follow the chain rule, the derivatives of each layer are multiplied down the network. When we use a lot of deeper layers, and we have hidden layers like sigmoid, we could have derivatives being scaled down below 0.25 for each layer. So when the number of layers derivatives are multiplied the gradient decreases exponentially as we propagate down to the initial layers.

# Variable Encoding
## One-Hot Encoding
- One-hot encoding should not be performed if the number of categories are high. This would result in a sparse data.
- Decision trees does not require doing one-hot encoding. Since xgboost, AFAIK, is a boosting of decision trees, I assume the encoding is not required.
- 피처내 값들이 서로 분리 되어있기 때문에, 우리가 모를 수 있는 어떤 관계나 영향을 주지 않는다.
- features 내 값의 종류가 많을 경우(High Cardinaliry), 매우 많은 Feature 들을 만들어 낸다. 이는, 모델 훈련의 속도를 낮추고 훈련에 더 많은 데이터를 필요로 하게 한다.(차원의 저주 문제)
- 단순히 0과 1로만 결과를 내어 큰 정보이득 없이 Tree 의 depth 만 깊게 만든다. 중요한건, Tree Depth 를 증가시키는 것에 비해, 2가지 경우로만 트리를 만들어 나간다는 것이다.
- Random Forest 와 같이, 일부 Feature 만 Sampling 하여 트리를 만들어나가는 경우, One-hot Feature 로 생성된 Feature 의 수가 많기 때문에 이 features가 다른 features보다 더 많이 쓰인다.
- Using `tensorflow.keras.utils.to_categorical()`
	```python
	from tensorflow.keras.utils import to_categorical

	to_categorical([2, 5, 1, 6, 3, 7])
	```
- Using `sklearn.preprocessing.OneHotEncoder()`
	```python
	from sklearn.preprocessing import OneHotEncoder
	```
## Label Encoding
- just mapping randomly to ints often works very well, especially if there aren’t to many.
- Another way is to map them to their frequency in the dataset. If you’re afraid of collisions, map to counts + a fixed (per category) small random perturbation.
- 값의 크기가 의미가 없는데 부여되어서 bias가 생길 수 있음.
- One major issue with this approach is there is no relation or order between these classes, but the algorithm might consider them as some order, or there is some relationship.
- pandas factorize는 NAN값을 자동으로 -1로 채우는데 반해 scikitlearn Label Encoder는 에러를 발생.
- Categorical 값들은 순서개념이 없음에도, 모델은 순서개념을 전제하여 학습하게 된다.
- Using `pandas`
	```python
	data["var"] = pd.Categorical(data["var"])
	
	vars = data["var"]
	vars_enc = data["var"].cat.codes
	```
- Using `sklearn.preprocessing.LabelEncoder()`
	```python
	from sklearn.preprocessing import LabelEncoder
	
	enc = LabelEncoder()

	enc.fit()
	enc.transform()
	enc.fit_transform()
	enc.inverse_transform()
	enc.classes_
	```
## Ordinal Encoding
- We do Ordinal encoding to ensure the encoding of variables retains the ordinal nature of the variable. This is reasonable only for ordinal variables, as I mentioned at the beginning of this article. This encoding looks almost similar to Label Encoding but slightly different as Label coding would not consider whether variable is ordinal or not and it will assign sequence of integers
- If we consider in the temperature scale as the order, then the ordinal value should from cold to "Very Hot. " Ordinal encoding will assign values as ( Cold(1) <Warm(2)<Hot(3)<”Very Hot(4)). Usually, we Ordinal Encoding is done starting from 1.
- Using `sklearn.preprocessing.OrdinalEncoder()`
	```python
	from sklearn.preprocessing import OrdinalEncoder
	```
## Mean Encoding
- 일반적인 방법인 category를 label encoding으로 얻은 숫자는 머신러닝 모델이 오해하기 쉽다.
- target value의 더 큰 값에 해당하는 범주가 더 큰 숫자로 인코딩 되게 하는 것이다.
- Target encoding can lead to data leakage. To avoid that, fit the encoder only on your train set, and then use its transform method to encode categories in both train and test sets.(train에 encoding fitting 후 그것을 test와 train 적용)
- 특히 Gradient Boosting Tree 계열에 많이 쓰이고 있다.
- 오버피팅의 문제가 있다. 하나는 Data Leakage 문제인데, 사실 훈련 데이터에는 예측 값에 대한 정보가 전혀 들어가면 안되는게 일반적이다. 그런데, Mean encoding 과정을 보면, 사실 encoding 된 값에는 예측 값에 대한 정보가 포함되어있다. 이러한 문제는 모델을 Training set 에만 오버피팅 되도록 만든다.
- 다른 하나는 하나의 label 값의 대표값을 trainset의 하나의 mean 으로만 사용한다는 점이다. 만약 testset 에 해당 label 값의 통계적인 분포가 trainset 과 다르다면, 오버피팅이 일어날 수 밖에 없다. 특히, 이런 상황은 Categorical 변수 내 Label의 분포가 매우 극단적인 경우에 발생한다. 예를 들어 Trainset 에는 남자가 100명, 여자가 5명이고, Testset 에는 50명, 50명이라고 하자. 우리는 Trainset 으로 Mean encoding 할텐데, 여자 5명의 평균값이 Testset 의 여자 50명을 대표할 수 있을까? 어려울 수 밖에 없다.
- Usually, Mean encoding is notorious for over-fitting; thus, a regularization with cross-validation or some other approach is a must on most occasions.
## Frequency Encoding
- 값 분포에 대한 정보가 잘 보존. 값의 빈도가 타겟과 연관이 있으면 아주 유용.
-Encoding한 Category끼리 같은 Frequency를 가진다면 Feature로 사용하지 않아도 됨.(?)
## Grouping
- you’ll need to do some exploratory data analysis to do some feature engineering like grouping categories or tactfully assigning appropriate integer values to match the relation of the variable with the output.
- if you know something about the categories you can perhaps group them and add an additional feature group id then order them by group id.
# `category_encoders`
```python
!pip install --upgrade category_encoders
```
```python
import category_encoders as ce
```
## `ce.target_encoder`
### `ce.target_encoder.TargetEncoder()`
```python
encoder = ce.target_encoder.TargetEncoder(cols=["company1"])
encoder.fit(data["company1"], data["money"]);
data["company1_label"] = encoder.transform(data["company1"]).round(0)
```

# Split Dataset
- Reference: https://developers.google.com/machine-learning/crash-course/training-and-test-sets/splitting-data
## Random Sample
- The benefit of this approach is that we can see how the model reacts to previously unseen data.
However, what if one subset of our data only have people of a certain age or income levels? This is typically referred to as a sampling bias:
Sampling bias is systematic error due to a non-random sample of a population, causing some members of the population to be less likely to be included than others, resulting in a biased sample.
- If only use a train/test split, then I would advise comparing the distributions of your train and test sets. If they differ significantly, then you might run into problems with generalization. Use Facets to easily compare their distributions.
- Using `sklearn.model_selection.train_test_split()`
	```python
	from sklearn.model_selection import train_test_split

	tr_X, te_X, tr_y, te_y = train_test_split(arrays, test_size, [shuffle=True], [stratify], [random_state])
	```
	- [`test_size`]: If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the train split. If int, represents the absolute number of train samples. 
	- [`stratify`]: If not None, data is split in a stratified fashion, using this as the class labels.

# Cross Validation (CV)
## Holdout Set
- When optimizing the hyperparameters of your model, you might overfit your model if you were to optimize using the train/test split.
Why? Because the model searches for the hyperparameters that fit the specific train/test you made.
## K-Fold CV
- We typically choose either i=5 or k=10 as they find a nice balance between computational complexity and validation accuracy:
```python
from sklearn.model_selection import KFold
```
### Stratified K-Fold CV
```python
from sklearn.model_selection import StratifiedKFold
```
## Group K-Fold CV
```python
from sklearn.model_selection import GroupKFold
```
### Stratified Group K-Fold CV
```python
from sklearn.model_selection import StratifiedGroupKFold
```
## Leave-One-Out CV
- This variant is identical to k-fold CV when k = n (number of observations).
```python
from sklearn.model_selection import LeaveOneOut

loo = LeaveOneOut()
```
## Leave-One-Group-Out CV
```python
from sklearn.model_selection import LeaveOneGroupOut
```
## Nested CV
- When you are optimizing the hyperparameters of your model and you use the same k-Fold CV strategy to tune the model and evaluate performance you run the risk of overfitting. You do not want to estimate the accuracy of your model on the same split that you found the best hyperparameters for.
- Instead, we use a Nested Cross-Validation strategy allowing to separate the hyperparameter tuning step from the error estimation step.
- The inner loop for hyperparameter tuning and
the outer loop for estimating accuracy.
- You are free to select the cross-validation approaches you use in the inner and outer loops. For example, you can use Leave-one-group-out for both the inner and outer loops if you want to split by specific groups.

# Evaluation Metrics
## MSE (Mean Squared Error)
- TensorFlow implementation
	```python
	from tensorflow.keras import metrics

	mse = metrics.MeanSquaredError([name])().numpy()
	```
- Scikit-learn implementation
	```python
	from sklearn.metrics import mean_squared_error
	
	mse = mean_squared_error()
	```
## RMSE (Root Mean Squared Error)
```python
from tensorflow.keras import metrics

rmse = metrics.RootMeanSquaredError([name])().numpy()
```
## MAE (Mean Absolute Error)
```python
from tensorflow.keras import metrics

mae = metrics.MeanAbsoluteError([name])().numpy()
```
## MAPE (Mean Absolute Percentage Error)
```python
from tensorflow.keras import metrics

mape = metrics.MeanAbsolutePercentageError([name])().numpy()
```
## Accuracy, Recall (Confusion Matrix)
- ![confusion_matrix](https://www.popit.kr/wp-content/uploads/2017/04/table-1024x378.png)
## Error Rate (CER (Character Error Rate), WER (Word Error Rate))
- Reference: https://towardsdatascience.com/evaluating-ocr-output-quality-with-character-error-rate-cer-and-word-error-rate-wer-853175297510
- Edit distance
  - Reference: https://en.wikipedia.org/wiki/Edit_distance
  - Edit distance is a way of quantifying how dissimilar two strings (e.g., words) are to one another by counting the minimum number of operations required to transform one string into the other.
  - Levenshtein distance
    - Allows deletion, insertion and substitution.
    - It is the minimum number of single-character (or word) edits (i.e., insertions, deletions, or substitutions) required to change one word (or sentence) into another. The more different the two text sequences are, the higher the number of edits needed, and thus the larger the Levenshtein distance.
- The usual way of evaluating prediction output is with the accuracy metric, where we indicate a match (1) or a no match (0). However, this does not provide enough granularity to assess OCR performance effectively. We should instead use error rates to determine the extent to which the OCR transcribed text and ground truth text (i.e., reference text labeled manually) differ from each other. A common intuition is to see how many characters were misspelled. While this is correct, the actual error rate calculation is more complex than that. This is because the OCR output can have a different length from the ground truth text.
- The question now is, how do you measure the extent of errors between two text sequences? This is where Levenshtein distance enters the picture.
- CER calculation is based on the concept of Levenshtein distance, where we count the minimum number of character-level operations required to transform the ground truth text (aka reference text) into the OCR output.
- CER = (S + D + I) / N
  - N: Number of characters in reference text (aka ground truth)
- Normalized CER = (S + D + I) / (S + D + I + C)
  - C: Number of correct characters
```python
import fastwer

cer = fastwer.score_sent(output, ref, char_level=True)
wer = fastwer.score_sent(output, ref, char_level=False)
```

# Recurrent Neural Network
- Reference: https://wikidocs.net/22886
- 앞서 배운 신경망들은 전부 은닉층에서 활성화 함수를 지난 값은 오직 출력층 방향으로만 향했습니다. 이와 같은 신경망들을 피드 포워드 신경망(Feed Forward Neural Network)이라고 합니다. 그런데 그렇지 않은 신경망들이 있습니다. RNN(Recurrent Neural Network) 또한 그 중 하나입니다. RNN은 은닉층의 노드에서 활성화 함수를 통해 나온 결과값을 출력층 방향으로도 보내면서, 다시 은닉층 노드의 다음 계산의 입력으로 보내는 특징을 갖고있습니다.
- 메모리 셀이 출력층 방향 또는 다음 시점인 t+1의 자신에게 보내는 값을 은닉 상태(hidden state) 라고 합니다. 다시 말해 t 시점의 메모리 셀은 t-1 시점의 메모리 셀이 보낸 은닉 상태값을 t 시점의 은닉 상태 계산을 위한 입력값으로 사용합니다.
## LSTM (Long Short-Term Memory)
- Reference: https://en.wikipedia.org/wiki/Long_short-term_memory
- *A common LSTM unit is composed of a cell, an input gate, an output gate and a forget gate. The cell remembers values over arbitrary time intervals and the three gates regulate the flow of information into and out of the cell.*
- ***LSTMs were developed to deal with the vanishing gradient problem that can be encountered when training traditional RNNs. Relative insensitivity to gap length is an advantage of LSTM over RNNs.***
- In theory, classic (or "vanilla") RNNs can keep track of arbitrary long-term dependencies in the input sequences. The problem with vanilla RNNs is computational (or practical) in nature: *when training a vanilla RNN using back-propagation, the long-term gradients which are back-propagated can "vanish" (that is, they can tend to zero) or "explode" (that is, they can tend to infinity), because of the computations involved in the process, which use finite-precision numbers. RNNs using LSTM units partially solve the vanishing gradient problem, because LSTM units allow gradients to also flow unchanged. However, LSTM networks can still suffer from the exploding gradient problem.*
## Bidirectional Recurrent Neural Network
- 양방향 순환 신경망은 시점 t에서의 출력값을 예측할 때 이전 시점의 입력뿐만 아니라, 이후 시점의 입력 또한 예측에 기여할 수 있다는 아이디어에 기반합니다.
- 양방향 RNN은 하나의 출력값을 예측하기 위해 기본적으로 두 개의 메모리 셀을 사용합니다. 첫번째 메모리 셀은 앞에서 배운 것처럼 앞 시점의 은닉 상태(Forward States) 를 전달받아 현재의 은닉 상태를 계산합니다. 위의 그림에서는 주황색 메모리 셀에 해당됩니다. 두번째 메모리 셀은 앞에서 배운 것과는 다릅니다. 앞 시점의 은닉 상태가 아니라 뒤 시점의 은닉 상태(Backward States) 를 전달 받아 현재의 은닉 상태를 계산합니다. 입력 시퀀스를 반대 방향으로 읽는 것입니다. 위의 그림에서는 초록색 메모리 셀에 해당됩니다. 그리고 이 두 개의 값 모두가 현재 시점의 출력층에서 출력값을 예측하기 위해 사용됩니다.

# Feature Importance
## Permutation Feature Importance
- Reference: https://scikit-learn.org/stable/modules/permutation_importance.html
- Permutation feature importance is a model inspection technique that can be used for any fitted estimator when the data is tabular.
- Features that are important on the training set but not on the held-out set might cause the model to overfit.
- Tree-based models provide an alternative measure of feature importances based on the mean decrease in impurity (MDI). Impurity is quantified by the splitting criterion of the decision trees (Gini, Entropy or Mean Squared Error). However, this method can give high importance to features that may not be predictive on unseen data when the model is overfitting. Permutation-based feature importance, on the other hand, avoids this issue, since it can be computed on unseen data.
- Furthermore, impurity-based feature importance for trees are strongly biased and favor high cardinality features (typically numerical features) over low cardinality features such as binary features or categorical variables with a small number of possible categories.
- Permutation-based feature importances do not exhibit such a bias. Additionally, the permutation feature importance may be computed performance metric on the model predictions and can be used to analyze any model class (not just tree-based models).
- When two features are correlated and one of the features is permuted, the model will still have access to the feature through its correlated feature. This will result in a lower importance value for both features, where they might actually be important.
- One way to handle this is to cluster features that are correlated and only keep one feature from each cluster.
- Reference: https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance_multicollinear.html#sphx-glr-auto-examples-inspection-plot-permutation-importance-multicollinear-py
- Because this dataset contains multicollinear features, the permutation importance will show that none of the features are important. One approach to handling multicollinearity is by performing hierarchical clustering on the features’ Spearman rank-order correlations, picking a threshold, and keeping a single feature from each cluster.
- The permutation importance plot shows that permuting a feature drops the accuracy by at most 0.012, which would suggest that none of the features are important.
- When features are collinear, permutating one feature will have little effect on the models performance because it can get the same information from a correlated feature. One way to handle multicollinear features is by performing hierarchical clustering on the Spearman rank-order correlations, picking a threshold, and keeping a single feature from each cluster.
- Reference: https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance.html#sphx-glr-auto-examples-inspection-plot-permutation-importance-py
- We will show that the impurity-based feature importance can inflate the importance of numerical features.
- Furthermore, the impurity-based feature importance of random forests suffers from being computed on statistics derived from the training dataset: the importances can be high even for features that are not predictive of the target variable, as long as the model has the capacity to use them to overfit.
- random_num is a high cardinality numerical variable (as many unique values as records).
random_cat is a low cardinality categorical variable (3 possible values).
The impurity-based feature importance ranks the numerical features to be the most important features. As a result, the non-predictive random_num variable is ranked the most important!
This problem stems from two limitations of impurity-based feature importances:
impurity-based importances are biased towards high cardinality features;
impurity-based importances are computed on training set statistics and therefore do not reflect the ability of feature to be useful to make predictions that generalize to the test set (when the model has enough capacity).
- It is also possible to compute the permutation importances on the training set. This reveals that random_num gets a significantly higher importance ranking than when computed on the test set. The difference between those two plots is a confirmation that the RF model has enough capacity to use that random numerical feature to overfit. You can further confirm this by re-running this example with constrained RF with min_samples_leaf=10.
- Reference: https://christophm.github.io/interpretable-ml-book/feature-importance.html#feature-importance-data
- We measure the importance of a feature by calculating the increase in the model's prediction error after permuting the feature. A feature is "important" if shuffling its values increases the model error, because in this case the model relied on the feature for the prediction. A feature is "unimportant" if shuffling its values leaves the model error unchanged, because in this case the model ignored the feature for the prediction.
- If you want a more accurate estimate, you can estimate the error of permuting feature j by pairing each instance with the value of feature j of each other instance (except with itself). This gives you a dataset of size n(n-1) to estimate the permutation error, and it takes a large amount of computation time. I can only recommend using the n(n-1) method if you are serious about getting extremely accurate estimates.
- **The feature importance based on training data makes us mistakenly believe that features are important for the predictions, when in reality the model was just overfitting and the features were not important at all.**
- Feature importance based on the training data tells us which features are important for the model in the sense that it depends on them for making predictions.
- If you would use (nested) cross-validation for the feature importance estimation, you would have the problem that the feature importance is not calculated on the final model with all the data, but on models with subsets of the data that might behave differently.
-  **You need to decide whether you want to know how much the model relies on each feature for making predictions (-> training data) or how much the feature contributes to the performance of the model on unseen data (-> test data).**
## Drop-out Feature Importance
- Reference: https://towardsdatascience.com/explaining-feature-importance-by-example-of-a-random-forest-d9166011959e
## On Train Set or Test Set?
- Reference: https://christophm.github.io/interpretable-ml-book/feature-importance.html
### On Test Set
- Really, it is one of the first things you learn in machine learning: If you measure the model error (or performance) on the same data on which the model was trained, the measurement is usually too optimistic, which means that the model seems to work much better than it does in reality. And since the permutation feature importance relies on measurements of the model error, we should use unseen test data. The feature importance based on training data makes us mistakenly believe that features are important for the predictions, when in reality the model was just overfitting and the features were not important at all.

# Feature Engineering
- Reference: https://adataanalyst.com/machine-learning/comprehensive-guide-feature-engineering/
## Time-Related Variables
- Time-stamp attributes are usually denoted by the EPOCH time or split up into multiple dimensions such as (Year, Month, Date, Hours, Minutes, Seconds). But in many applications, a lot of that information is unnecessary. Consider for example a supervised system that tries to predict traffic levels in a city as a function of Location+Time. In this case, trying to learn trends that vary by seconds would mostly be misleading. The year wouldn’t add much value to the model as well. Hours, day and month are probably the only dimensions you need. So when representing the time, try to ensure that your model does require all the numbers you are providing it.
- Here is an example hypothesis: An applicant who takes days to fill in an application form is likely to be less interested / motivated in the product compared to some one who fills in the same application with in 30 minutes. Similarly, for a bank, time elapsed between dispatch of login details for Online portal and customer logging in might show customers’ willingness to use Online portal. Another example is that a customer living closer to a bank branch is more likely to have a higher engagement than a customer living far off.
## Creating New Ratios and Proportions
- For example, in order to predict future performance of credit card sales of a branch, ratios like credit card sales / Sales person or Credit card Sales / Marketing spend would be more powerful than just using absolute number of card sold in the branch
## Creating Weights
- There may be domain knowledge that items with a weight above 4 incur a higher taxation rate. That magic domain number could be used to create a new binary feature Item_Above_4kg with a value of "1” for our example of 6289 grams.
## Creating Aggregated Values
- You may also have a quantity stored as a rate or an aggregate quantity for an interval. For example, Num_Customer_Purchases aggregated over a year.
- For example, the following new binary features could be created: Purchases_Summer, Purchases_Fall, Purchases_Winter and Purchases_Spring.
## Splitting Features
- The Item_Weight could be split into two features: Item_Weight_Kilograms and Item_Weight_Remainder_Grams, with example values of 6 and 289 respectively.
## Bucketing, Binning, Discretization
- Sometimes, it makes more sense to represent a numerical attribute as a categorical one.
- Consider the problem of predicting whether a person owns a certain item of clothing or not. Age might definitely be a factor here. What is actually more pertinent, is the Age Group. So what you could do, is have ranges such as 1-10, 11-18, 19-25, 26-40, etc.
- It reduces overfitting in certain applications, where you don’t want your model to try and distinguish between values that are too close by – for example, you could club together all latitude values that fall in a city, if your property of interest is a function of the city as a whole.
- Binning also reduces the effect of tiny errors, by ’rounding off’ a given value to the nearest representative. Binning does not make sense if the number of your ranges is comparable to the total possible values, or if precision is very important to you.
- For example, you may have Item_Weight in grams, with a value like 6289. You could create a new feature with this quantity in kilograms as 6.289 or rounded kilograms like 6. If the domain is shipping data, perhaps kilograms is sufficient or more useful (less noisy) a precision for Item_Weight.
## Variables Transformation
- Transform complex non-linear relationships into linear relationships.Existence of a linear relationship between variables is easier to comprehend compared to a non-linear or curved relation. Transformation helps us to convert a non-linear relation into linear relation. Scatter plot can be used to find the relationship between two continuous variables. These transformations also improve the prediction. Log transformation is one of the commonly used transformation technique used in these situations
- For right skewed distribution, we take square / cube root or logarithm of variable and for left skewed, we take square / cube or exponential of variables.
- Cube root can be applied to negative values including zero. Square root can be applied to positive values including zero.
## Feature Crosses
- Feature crosses are a unique way to combine two or more categorical attributes into a single one. This is extremely useful a technique, when certain features together denote a property better than individually by themselves. Mathematically speaking, you are doing a cross product between all possible values of the categorical features.
Consider a feature A, with two possible values {A1, A2}. Let B be a feature with possibilities {B1, B2}. Then, a feature-cross between A & B (lets call it AB) would take one of the following values: {(A1, B1), (A1, B2), (A2, B1), (A2, B2)}. You can basically give these ‘combinations’ any names you like. Just remember that every combination denotes a synergy between the information contained by the corresponding values of A and B.

# Distance Features
- Reference: https://www.tandfonline.com/doi/full/10.1080/10095020.2018.1503775
- The Euclidean function is unrealistic for some (notably urban) settings which contain complex physical restrictions and social structures for example road and path networks, large restricted areas of private land and legal road restrictions such as speed limits and one-way systems.
## House prices in space
- Most contemporary analysis mimics this trend, for example predicting property value by using (1) the average sales price of other properties in the local comparable markets, (2) a spatial clustering of properties and demographics (Malczewski 2004) and (3) a local demographic "trade area” (Daniel 1994).
- In the case of spatially dependent data, cross-validation is optimistic due to its inherent IID assumption.
- Euclidean distances are exclusively considered in all of the above work. This paper hypothesizes that house prices are related to a more complex structural network relating to (restricted) road distance and travel time; hence, we introduce an approximate (restricted) road distance and travel time metric using the Minkowski distance function for a valid house price Kriging predictor (Matheron 1963; Cressie 1990).
## Data Description
- Figure 1. A comparison of an Euclidean distance matrix versus a drive time distance matrix and a road distance matrix around the center point of Coventry. (a) Euclidean distance buffer from 0 to 4 miles around the centre of Coventry; (b) Travel time distance buffer from 0 to 10 minutes drive time around the centre of Coventry; (c) Road distance buffer from 0 to 4 miles around the centre of Coventry.
## Collapsing Time
- The price paid data for 2016 are addressed only (herewithin named ). This accounts for 3669 sales in Coventry. Stage 1 predicts each property’s sale price based on its value on the 1 January 2017 (for time singularity). This process involves each property being assigned some percentage price change based on the date that it was sold and the lower super output area that the property is contained within to produce a value for all 3669 properties at the date 1 January 2017 (). The error for the purposes of this experiment is minimal or nonexistent due to the small temporal and spatial aggregate areas being considered.
- Figure 2 shows an exact example where the distance between houses  to  is 0.24 mi along the red dotted line which takes a route along "Brownshill Green Road” and is marked as a one-way system, this means that the route  to  must be different, which, in this case, is further; hence, the distance matrix is not symmetric. The same reasoning applies for a travel time matrix

# Model Linearity
## Difference between Linear and Nonlinear
- Reference: https://statisticsbyjim.com/regression/difference-between-linear-nonlinear-regression-models/
- The form is linear in the parameters because all terms are either the constant or a parameter multiplied by an independent variable (IV). A linear regression equation simply sums the terms. While the model must be linear in the parameters, you can raise an independent variable by an exponent to fit a curve. For instance, you can include a squared or cubed term.
Nonlinear regression models are anything that doesn’t follow this one form.
While both types of models can fit curvature, nonlinear regression is much more flexible in the shapes of the curves that it can fit. After all, the sky is the limit when it comes to the possible forms of nonlinear models.
- While the independent variable is squared, the model is still linear in the parameters. Linear models can also contain log terms and inverse terms to follow different kinds of curves and yet continue to be linear in the parameters.
- If a regression equation doesn’t follow the rules for a linear model, then it must be a nonlinear model
- Reference: https://brunch.co.kr/@gimmesilver/18
- 비선형 모델은 데이터를 어떻게 변형하더라도 파라미터를 선형 결합식으로 표현할 수 없는 모델을 말합니다. 이런 비선형 모델 중 단순한 예로는 아래와 같은 것이 있습니다. 이 식은 아무리 x, y 변수를 변환하더라도 파라미터를 선형식으로 표현할 수 없습니다.
    선형 회귀 모델은 파라미터 계수에 대한 해석이 단순하지만 비선형 모델은 모델의 형태가 복잡할 경우 해석이 매우 어렵습니다. 그래서 보통 모델의 해석을 중시하는 통계 모델링에서는 비선형 회귀 모델을  잘 사용하지 않습니다. 
    그런데 만약 회귀 모델의 목적이 해석이 아니라 예측에 있다면 비선형 모델은 대단히 유연하기 때문에 복잡한 패턴을 갖는 데이터에 대해서도 모델링이 가능합니다. 그래서 충분히 많은 데이터를 갖고 있어서 variance error를 충분히 줄일 수 있고 예측 자체가 목적인 경우라면 비선형 모델은 사용할만한 도구입니다. 기계 학습 분야에서는 실제 이런 비선형 모델을 대단히 많이 사용하고 있는데 가장 대표적인 것이 소위 딥 러닝이라고 부르는 뉴럴 네트워크입니다.
- 정리하자면, 선형 회귀 모델은 파라미터가 선형식으로 표현되는 회귀 모델을 의미합니다. 그리고 이런 선형 회귀 모델은 파라미터를 추정하거나 모델을 해석하기가 비선형 모델에 비해 비교적 쉽기 때문에, 데이터를 적절히 변환하거나 도움이 되는 feature들을 추가하여 선형 모델을 만들 수 있다면 이렇게 하는 것이 적은 개수의 feature로 복잡한 비선형 모델을 만드는 것보다 여러 면에서 유리합니다. 반면 선형 모델은 표현 가능한 모델의 가짓수(파라미터의 개수가 아니라 파라미터의 결합 형태)가 한정되어 있기 때문에 유연성이 떨어집니다. 따라서 복잡한 패턴을 갖고 있는 데이터에 대해서는 정확한 모델링이 불가능한 경우가 있습니다. 그래서 최근에는 모델의 해석보다는 정교한 예측이 중요한 분야의 경우 뉴럴 네트워크와 같은 비선형 모델이 널리 사용되고 있습니다.

# Missing Value
- Reference: https://adataanalyst.com/machine-learning/comprehensive-guide-feature-engineering/
## Data Imputation
- Using `sklearn.impute.SimpleImputer()`
	```python
	# The `SimpleImputer()` class provides basic strategies for imputing missing values. Missing values can be imputed with a provided constant value, or using the statistics (mean, median or most frequent) of each column in which the missing values are located. This class also allows for different missing values encodings.
	from sklearn.impute import SimpleImputer
	
	# `missing_values`: (default `np.nan`)
	# `strategy`: (`"mean"`, `"median"`, `"most_frequent"`, `"constant"`)
		# `strategy="most_frequent"`: Can be used with strings or numeric data. If there is more than one such value, only the smallest is returned.
		# `strategy="constatn"`: Replace missing values with `fill_value`.
	imp = SimpleImputer(missing_values, strategy, fill_value)
	imp.fit()
	# imp.transform()
	# imp.fit_transform()
	```
## Prediction
- In this case, we divide our data set into two sets: One set with no missing values for the variable and another one with missing values. First data set become training data set of the model while second data set with missing values is test data set and variable with missing values is treated as target variable.
## Interpolation
## K-Nearest Neighbors Imputation
- KNN algorithm is very time-consuming in analyzing large database. It searches through all the dataset looking for the most similar instances.
- Choice of k-value is very critical. Higher value of k would include attributes which are significantly different from what we need whereas lower value of k implies missing out of significant attributes. 

# Outliers
## Types of Outliers
- Reference: https://adataanalyst.com/machine-learning/comprehensive-guide-feature-engineering/
- Univariate outliers can be found when we look at distribution of a single variable.
Multi-variate outliers are outliers in an n-dimensional space. In order to find them, you have to look at distributions in multi-dimensions.
### Data Entry Errors
- Human errors such as errors caused during data collection, recording, or entry can cause outliers in data.
### Data Processing Errors
- It is possible that some manipulation or extraction errors may lead to outliers in the dataset.
### Measurement Errors
- It is the most common Reference of outliers. This is caused when the measurement instrument used turns out to be faulty. 
E## Experimental Errors
- For example: In a 100m sprint of 7 runners, one runner missed out on concentrating on the ‘Go’ call which caused him to start late. Hence, this caused the runner’s run time to be more than other runners.
### Intentional Outliers
- For example: Teens would typically under report the amount of alcohol that they consume. Only a fraction of them would report actual value. Here actual values might look like outliers because rest of the teens are under reporting the consumption.
### Sampling Errors
- For instance, we have to measure the height of athletes. By mistake, we include a few basketball players in the sample.
### Natural Outliers
## Outliers Treatment
### Deleting
### Transforming
- Natural log of a value reduces the variation caused by extreme values.
- Decision Tree algorithm allows to deal with outliers well due to binning of variable. We can also use the process of assigning weights to different observations.
### Binning
- Decision Tree algorithm allows to deal with outliers well due to binning of variable. We can also use the process of assigning weights to different observations.
### Imputing
- We can use mean, median, mode imputation methods. Before imputing values, we should analyse if it is natural outlier or artificial. If it is artificial, we can go with imputing values. We can also use statistical model to predict values of outlier.
### Treating Separately
- If there are significant number of outliers, we should treat them separately in the statistical model. One of the approach is to treat both groups as two different groups and build individual model for both groups and then combine the output.

# K-Means Clustering
```python
# 초기 중심점을 랜덤으로 설정.
flags = cv2.KMEANS_RANDOM_CENTERS
compactness, labels, centers = cv2.kmeans(z, 3, None, criteria, 10, flags)
```

# Zero-Shot Learning (ZSL)
- Reference: https://en.wikipedia.org/wiki/Zero-shot_learning
- Zero-shot learning (ZSL) is a problem setup in machine learning, where at test time, a learner observes samples from classes, which were not observed during training, and needs to predict the class that they belong to. Zero-shot methods generally work by associating observed and non-observed classes through some form of auxiliary information, which encodes observable distinguishing properties of objects. For example, given a set of images of animals to be classified, along with auxiliary textual descriptions of what animals look like, an artificial intelligence model which has been trained to recognize horses, but has never been given a zebra, can still recognize a zebra when it also knows that zebras look like striped horses.

# Variational AutoEncoder (VAE)
- Reference: https://www.tensorflow.org/tutorials/generative/cvae
- A VAE is a probabilistic take on the autoencoder, a model which takes high dimensional input data and compresses it into a smaller representation. Unlike a traditional autoencoder, which maps the input onto a latent vector, a VAE maps the input data into the parameters of a probability distribution, such as the mean and variance of a Gaussian. This approach produces a continuous, structured latent space, which is useful for image generation.
- Reference: https://jaan.io/what-is-variational-autoencoder-vae-tutorial/
- Local latent variables: these are the z_i for each datapoint x_i. There are no global latent variables. Because there are only local latent variables, we can easily decompose the ELBO into terms L_i that depend only on a single datapoint x_i. This enables stochastic gradient descent.
- Inference: in neural nets, inference usually means prediction of latent representations given new, never-before-seen datapoints. In probability models, inference refers to inferring the values of latent variables given observed data.
- The latent variable  is now generated by a function of ,  and , which would enable the model to backpropagate gradients in the encoder through  and  respectively, while maintaining stochasticity through .
- Encoder
	- q(z|x)
- Decoder
	- p(x|z)

# Regularization
## L1 Regularization (= Lasso Regression)
## L2 Regularization (= Ridge Regression)

# Transfer Learning
- Reference: https://www.toptal.com/machine-learning/semi-supervised-image-classification
- Transfer learning means using knowledge from a similar task to solve a problem at hand. In practice, it usually means using as initializations the deep neural network weights learned from a similar task, rather than starting from a random initialization of the weights, and then further training the model on the available labeled data to solve the task at hand.
- Transfer learning enables us to train models on datasets as small as a few thousand examples, and it can deliver a very good performance. Transfer learning from pretrained models can be performed in three ways:
	1. Feature Extraction 
		- *Usually, the last layers of the neural network are doing the most abstract and task-specific calculations, which are generally not easily transferable to other tasks. By contrast, the initial layers of the network learn some basic features like edges and common shapes, which are easily transferable across tasks.*
		- A common practice is to take a model pretrained on large labeled image datasets (such as ImageNet) and chop off the fully connected layers at the end. New, fully connected layers are then attached and configured according to the required number of classes. *Transferred layers are frozen, and the new layers are trained on the available labeled data for your task.*
		- In this setup, the pretrained model is being used as a feature extractor, and the fully connected layers on the top can be considered a shallow classifier. *This setup is more robust than overfitting as the number of trainable parameters is relatively small, so this configuration works well when the available labeled data is very scarce.* What size of dataset qualifies as a very small dataset is usually a tricky problem with many aspects of consideration, including the problem at hand and the size of the model backbone. Roughly speaking, I would use this strategy for a dataset consisting of a couple of thousand images.
	2. Fine-tuning
		- *Alternatively, we can transfer the layers from a pretrained network and train the entire network on the available labeled data. This setup needs a little more labeled data because you are training the entire network and hence a large number of parameters. This setup is more prone to overfitting when there is a scarcity of data.*
	3. Two-stage Transfer Learning
		- This approach is my personal favorite and usually yields the best results, at least in my experience. Here, ***we train the newly attached layers while freezing the transferred layers for a few epochs before fine-tuning the entire network. Fine-tuning the entire network without giving a few epochs to the final layers can result in the propagation of harmful gradients from randomly initialized layers to the base network. Furthermore, fine-tuning requires a comparatively smaller learning rate, and a two-stage approach is a convenient solution to it.***

# Droupout
- Reference: https://leimao.github.io/blog/Dropout-Explained/#:~:text=During%20inference%20time%2C%20dropout%20does,were%20multiplied%20by%20pkeep%20.
- During training time, dropout randomly sets node values to zero. In the original implementation, we have "keep probability" p. So dropout randomly kills node values with "dropout probability" 1 − p. During inference time, dropout does not kill node values, but all the weights in the layer were multiplied by keep probability. One of the major motivations of doing so is to make sure that the distribution of the values after affine transformation during inference time is close to that during training time. Equivalently, This multiplier could be placed on the input values rather than the weights.
- TensorFlow has its own implementation of dropout which only does work during training time.

# `annoy`
- Install
	- On Windows
		```python
		# Reference: https://www.lfd.uci.edu/~gohlke/pythonlibs/#annoy
		!pip install "D:/annoy-1.17.0-cp38-cp38-win_amd64.whl"
		```
	- On MacOS
		```python
		pip install --user annoy
		```
- Reference: https://github.com/spotify/annoy
- Annoy (Approximate Nearest Neighbors Oh Yeah) is a C++ library with Python bindings to *search for points in space that are close to a given query point.*
- Tree Building
	```python
	from annoy import AnnoyIndex

	# `AnnoyIndex()` Returns a new index that's read-write and stores vector of f dimensions.
		# `f`: Length of item vector that will be indexed.
		# `metric`: (`"angular"`, `"euclidean"`, `"manhattan"`, `"hamming"`, `"dot"`)
	tree = AnnoyIndex(f, metric)
	for i, value in enumerate(embs):
		tree.add_item(i, value)
	# Builds a forest of `n_trees` trees. More trees gives higher precision when querying. After calling `build()`, no more items can be added.
		# `n_jobs`: Specifies the number of threads used to build the trees. `-1` uses all available CPU cores.
	tree.build(n_trees=20)
	```
- Similarity Measure
	```python
	# Returns the `n` closest items.
	# During the query it will inspect up to `search_k` nodes which defaults to `n_trees*n` if not provided. `search_k` gives you a run-time tradeoff between better accuracy and speed.
	sim_vecs = tree.get_nns_by_vector(v, n, search_k, include_distances)
	sim_items = tree.get_nns_by_item(i, n, search_k, include_distances)
	
	sim_vec = tree.get_item_vector(i)
	dist = tree.get_distance(i, j)
	```

# Greedy Search and Beam Search
```python
probs = [1, 2, 3, 4, 5]
data = np.array([random.sample(probs, k=5) for _ in range(10)])
```
## Greedy Search
```python
def greedy_search(data):
    return np.argmax(data, axis=1)
```
## Beam Search
- Python Implementation
	```python
	def beam_search(data, k):
		seq_score = [[list(), 0]]ㅠ
		for probs in data:
			cands = list()
			for seq, score in seq_score:
				for i, prob in enumerate(probs):
					cands.append([seq + [i], score - np.log(prob)])
			seq_score.extend(cands)
			seq_score = sorted(seq_score, key=lambda x:x[1])[:k]
		return [np.array(i[0]) for i in seq_score]
	```
- `ctcdecode` Implementation
  - Install
		```sh
		git clone --recursive https://github.com/parlance/ctcdecode.git
		cd ctcdecode
		pip install .
		# Or
		CFLAGS=-stdlib=libc++ pip install .
		```
	```python
	from ctcdecode import CTCBeamDecoder
	```

# `statsmodels`
```python
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
```
## Variance Inflation Factor (VIF)
```python
vif = pd.DataFrame()
vif["vif"] = [variance_inflation_factor(x_tr.values, i) for i in range(x_tr.shape[1])]
vif["feat"] = x_tr.columns
```
## Q-Q Plot
```python
fig = sm.qqplot(ax=axes[0], data=data["value"], fit=True, line="45")
```
## Oridinary Least Squares
```python
sm.OLS(y_tr, x_tr).fit()

# `coef`: The measurement of how change in the independent variable affects the dependent variable. Negative sign means inverse relationship. As one rises, the other falls.
# `R-squared`, `Adj. R-squared`
# `P>|t|`: p-value. Measurement of how likely the dependent variable is measured through the model by chance. That is, there is a (p-value) chance that the independent variable has no effect on the dependent variable, and the results are produced by chance. Proper model analysis will compare the p-value to a previously established threshold with which we can apply significance to our coefficient. A common threshold is 0.05.
# `Durbin-Watson`: Durbin-Watson statistic.
# `Skew`: Skewness.
# `Kurtosis`: Kurtosis.
# `Cond. No.`: Condition number of independent variables.
sm.OLS().fit().summary()

# Return `Adj. R-squared` of the independent variables.
sm.OLS().fit().rsquared_adj
sm.OLS().fit().fvalue
sm.OLS().fit().f_pvalue
sm.OLS().fit().aic
sm.OLS().fit().bic
# Return `coef`s of the independent variables.
sm.OLS().fit().params
# Return `P>|t|`s of the independent variables.
sm.OLS().fit().pvalues
sm.OLS().fit().predict()
```

# `scipy`
```python
import scipy
from scipy import stats
```
## `stats.shapiro()`
```python
Normality = pd.DataFrame([stats.shapiro(resid_tr["resid"])], index=["Normality"], columns=["Test Statistic", "p-value"]).T
```
- Return test statistic and p-value.
## `stats.boxcox_normalplot()`
```python
x, y = stats.boxcox_normplot(data["value"], la=-3, lb=3)
```
## `stats.boxcox()`
```python
y_trans, l_opt = stats.boxcox(data["value"])
```
