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

# Google Colab
## Mount Google Drive
```python
from google.colab import drive
import os
drive.mount("/content/drive")
os.chdir("/content")
# os.chdir("/content/drive/MyDrive/Libraries")
```
## Download Files to Local
```
from google.colab import files

files.download(path)
```
## Display Hangul
```python
import matplotlib as mpl
import matplotlib.pyplot as plt

%config InlineBackend.figure_format = "retina"
!apt -qq -y install fonts-nanum
fpath = "/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf"
fpath = "/NanumBarunGothic.ttf"
font = mpl.font_manager.FontProperties(fname=fpath, size=9)
plt.rc("font", family="NanumBarunGothic") 
mpl.font_manager._rebuild()
mpl.rcParams["axes.unicode_minus"] = False
```
## Prevent from Disconnecting
```
function ClickConnect(){
    console.log("코랩 연결 끊김 방지");
	document.querySelector("colab-toolbar-button#connect").click()}
setInterval(ClickConnect, 60*1000)
```
## Install Libraries Permanently
```python
!pip install --target=TARGET_PATH LIBRARY_NAME
```
## Use TPU in tensorflow
```python
resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu="grpc://" + os.environ["COLAB_TPU_ADDR"])
tf.config.experimental_connect_to_cluster(resolver)
tf.tpu.experimental.initialize_tpu_system(resolver)
strategy = tf.distribute.experimental.TPUStrategy(resolver)
...
with strategy.scope():
	...
```
```python
strategy = tf.distribute.experimental.TPUStrategy(resolver)
with strategy.scope():
    model = create_model()
    hist = model.fit()
```
## Substitution for `cv2.imshow()`
```python
from google.colab.patches import cv2_imshow
```

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
- Same as `tf.exp(logits) / tf.math.reduce_sum(tf.exp(logits), [axis])`
- Using `tensorflow.nn.softmax([axis])`
	```python
	tf.nn.softmax()
	```
	- `axis`: The dimension softmax would be performed on. The default is `-1` which indicates the last dimension.
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
- Using `tensorflow.keras.metrics.MeanSquaredError()`
	```python
	mse = metrics.MeanSquaredError([name])().numpy()
	```
- Using `sklearn.metrics.mean_squared_error()`
	```python
	from sklearn.metrics import mean_squared_error
	
	mse = mean_squared_error()
	```
## RMSE (Root Mean Squared Error)
```python
rmse = metrics.RootMeanSquaredError([name])().numpy()
```
## MAE (Mean Absolute Error)
```python
mae = metrics.MeanAbsoluteError([name])().numpy()
```
## MPE (Mean Percentage Error)
## MAPE (Mean Absolute Percentage Error)
```python
mape = metrics.MeanAbsolutePercentageError([name])().numpy()
```
## SMAPE (Symmetric Mean Absolute Percentage Error)
- Source: https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error
- The absolute difference between the actual value and forecast value is divided by half the sum of absolute values of the actual value and the forecast value. The value of this calculation is summed for every fitted point and divided again by the number of fitted points.
## R-Squared
- Source: https://statisticsbyjim.com/regression/r-squared-invalid-nonlinear-regression/
- Explained variance + Error variance = Total variance.
- However, this math works out correctly only for linear regression models. In nonlinear regression, these underlying assumptions are incorrect. Explained variance + Error variance DO NOT add up to the total variance! The result is that R-squared isn’t necessarily between 0 and 100%.
- If you use R-squared for nonlinear models, their study indicates you will experience the following problems:
R-squared is consistently high for both excellent and appalling models.
R-squared will not rise for better models all of the time.
If you use R-squared to pick the best model, it leads to the proper model only 28-43% of the time.
## Adjusted R-Squared
## RMSLE(Root Mean Squared Logarithmic Error)
- Source: https://shryu8902.github.io/machine%20learning/error/
## Binary Classification
- ![classification](https://www.popit.kr/wp-content/uploads/2017/04/table-1024x378.png)
## Binary Cross Entropy
```python
bc = metrics.BinaryCrossentropy()
```
## Categorical Cross Entropy
```python
scc = metrics.SparseCategoricalCrossentropy()
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
### Turkey Fences
### Z-score

# K-Nearest Neighbors (KNN)
- NumPy Implementation
	```python
	import numpy as np


	from collections import Counter, defaultdict


	def euclidean_distance(vec1, vec2):
		return np.linalg.norm(vec1 - vec2, ord=2)


	def get_score_from_distance(distance):
		return 1/distance


	def predict_label_by_knn(X_test, X_train, y_train, k, weights="uniform"):
		y_test = list()
		for x_test in X_test:
			dist_label = list()
			for x_train, label in zip(X_train, y_train):
				dist = euclidean_distance(x_test, x_train)
				dist_label.append((dist, label))

			dist_label.sort()
			dist_label = dist_label[:k]

			if weights == "uniform":
				counter = Counter([i[1] for i in dist_label])
				pred = counter.most_common(1)[0][0]        
			elif weights == "distance":
				label2score = defaultdict(int)
				for dist, label in dist_label:
					score = get_score_from_distance(dist)
					label2score[label] += score
				pred = sorted(label2score.items(), key=lambda x: x[1])[-1][0]
			else:
				raise ValueError("`weights` shoud be one of the following; ('unifrom', 'distance')")

			y_test.append(pred)

		return np.array(y_test)


	# Example
	size_tr = 30
	n_classes = 5
	X_tr = np.array([[np.random.rand(), np.random.rand()] for _ in range(size_tr)])
	y_tr = np.array([np.random.choice(range(n_classes)) for _ in range(size_tr)])

	size_te = 10
	X_te = np.array([[np.random.rand(), np.random.rand()] for _ in range(size_te)])

	predict_label_by_knn(X_te, X_tr, y_tr, k=5, weights="uniform")
	predict_label_by_knn(X_te, X_tr, y_tr, k=5, weights="distance")
	```


# K-Means Clustering
```python
# 초기 중심점을 랜덤으로 설정.
flags = cv2.KMEANS_RANDOM_CENTERS
compactness, labels, centers = cv2.kmeans(z, 3, None, criteria, 10, flags)
```

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

# TensorFlow Graph Excution
- Reference: https://towardsdatascience.com/eager-execution-vs-graph-execution-which-is-better-38162ea4dbf6
- In TensorFlow 2.0, you can decorate a Python function using `tf.function()` to run it as a single graph object. With this new method, you can easily build models and gain all the graph execution benefits.
- For simple operations, graph execution does not perform well because it has to spend the initial computing power to build a graph. We see the power of graph execution in complex calculations.
```python
import time

# (`SGD()`, `Adagrad()`, `Adam()`, ...)
optimizer = ...

loss_obj = losses....
def loss_func(y_true, y_pred):
    ...
    return ...

def acc_func(real, pred):
    ...
    return ...

tr_loss = metrics....
tr_acc = metrics....

# `input_signature`: A possibly nested sequence of `tf.TensorSpec()` objects specifying the `shape`s and `dtype`s of the Tensors that will be supplied to this function. If `None`, a separate function is instantiated for each inferred `input_signature`. If `input_signature` is specified, every input to func must be a Tensor, and `func` cannot accept `**kwargs`.
    # The input signature specifies the shape and type of each Tensor argument to the function using a tf.TensorSpec object. More general shapes can be used. This ensures only one ConcreteFunction is created, and restricts the GenericFunction to the specified shapes and types. It is an effective way to limit retracing when Tensors have dynamic shapes.
# Since TensorFlow matches tensors based on their shape, using a `None` dimension as a wildcard will allow functions to reuse traces for variably-sized input. Variably-sized input can occur if you have sequences of different length, or images of different sizes for each batch.
# The @tf.function trace-compiles train_step into a TF graph for faster execution. The function specializes to the precise shape of the argument tensors. To avoid re-tracing due to the variable sequence lengths or variable batch sizes (the last batch is smaller), use input_signature to specify more generic shapes.
# tf.function only allows creating new tf.Variable objects when it is called for the first time:
@tf.function(input_signature=...)
def train_step(x, y):
	...
    with tf.GradientTape() as tape:
        y_pred = model(x, ..., training=True)
        loss = loss_func(y, y_pred)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    tr_loss(loss)
    tr_acc(acc_func(y, y_pred))
    
ckpt_path = "..."
# TensorFlow objects may contain trackable state, such as `tf.Variables`, `tf.keras.optimizers.Optimizer` implementations, `tf.data.Dataset` iterators, `tf.keras.Layer` implementations, or `tf.keras.Model` implementations. These are called trackable objects.
# A `Checkpoint` object can be constructed to save either a single or group of trackable objects to a checkpoint file. It maintains a `save_counter` for numbering checkpoints.
ckpt = tf.train.Checkpoint(optimizer=optimizer, model=model)
# Manages multiple checkpoints by keeping some and deleting unneeded ones.
ckpt_manager = tf.train.CheckpointManager(ckpt, directory=ckpt_path, max_to_keep=...)
# The prefix of the most recent checkpoint in directory.
if ckpt_manager.latest_checkpoint:
    # `save_path`: The path to the checkpoint, as returned by `save` or `tf.train.latest_checkpoint`.
    ckpt.restore(save_path=ckpt_manager.latest_checkpoint)
    print ("Latest checkpoint restored!")
	
epochs = ...
for epoch in range(1, epochs + 1):
    start = time.time()
    # Resets all of the metric state variables to a predefined constant (typically 0). This function is called between epochs/steps, when a metric is evaluated during training.
    tr_loss.reset_states()
    tr_acc.reset_states()
    for batch, (x, y) in dataset_tr.enumerate(start=1):
        train_step(x, y)
        if batch % 50 == 0:
            print(f"Epoch: {epoch:3d} | Batch: {batch:5d} | Loss: {tr_loss.result():5.4f} | Accuracy: {tr_acc.result():5.4f}")
    if epoch % 1 == 0:
        # Every time `ckpt_manager.save()` is called, `save_counter` is increased.
        # `save_path`: The path to the new checkpoint. It is also recorded in the `checkpoints` and `latest_checkpoint` properties. `None` if no checkpoint is saved.
        save_path = ckpt_manager.save()
        print(f"Saving checkpoint for epoch {epoch} at {save_path}")
        print(f"Epoch: {epoch:3d} | Loss: {tr_loss.result():5.4f} | Accuracy: {tr_acc.result():5.4f}")
        print(f"Time taken for 1 epoch: {time.time() - start:5.0f} secs\n")
```

# PyTorch
```python
for epoch in range(1, n_epochs + 1):
	running_loss = 0
	for batch, (x, y) in enumerate(dl_tr, 1):
		...
		optimizer.zero_grad()
		...
		outputs = model(inputs)
		loss = criterion(inputs, outputs)
		loss.backward()
		optimizer.step()

		running_loss += loss.item()
		...
		if batch % ... == 0:
			...
			running_loss = 0
```

# GPU on PyTorch
```python
# `torch.cuda.is_available()`: Returns a bool indicating if CUDA is currently available.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
...

model = ModelName().to(device)
```
```python
# Returns the index of a currently selected device.
torch.cuda.current_device()
# Returns the number of GPUs available.
torch.cuda.device_count()
# Gets the name of the device.
torch.cuda.get_device_name(<index>)
```

# TensorFlow Tensors
- References: https://www.tensorflow.org/api_docs/python/tf/Tensor, https://stackoverflow.com/questions/57660214/what-is-the-utility-of-tensor-as-opposed-to-eagertensor-in-tensorflow-2-0, https://www.tensorflow.org/api_docs/python/tf/shape
## `EagerTensor`
```python
# During eager execution, you may discover your Tensors are actually of type `EagerTensor` (`tensorflow.python.framework.ops.EagerTensor`)
tf.Tensor(..., shape=..., dtype=...)

<EagerTensor>.numpy()

# `tf.shape()` and `Tensor.shape` should be identical in eager mode.
<EagerTensor>.shape
```
## `Tensor`
```python
# During graph execution
# In `tf.function` definitions, the shape may only be partially known. Most operations produce tensors of fully-known shapes if the shapes of their inputs are also fully known, but in some cases it's only possible to find the shape of a tensor at execution time.
# `Tensor` (`tensorflow.python.framework.ops.Tensor`) represents a tensor node in a graph that may not yet have been calculated.

# During graph execution, not all dimensions may be known until execution time. Hence when defining custom layers and models for graph mode, prefer the dynamic `tf.shape(x)` over the static `x.shape`.
tf.shape(<Tensor>)
```

# Operators
## `tf.identity()`
## `tf.constant()`
## `tf.convert_to_tensor()`
## `tf.cast([dtype])`
- Casts a tensor to a new type.
- Returns `1` if `True` else `0`.
## Reshape Tensor
```python
# TensorFlow
tf.reshape(<Tensor>, shape)
<Tensor>.reshape(shape)

# PyTorch
<Tensor>.view()
```
## `tf.transpose(a, perm)`
## `tf.range()`
## `tf.tile()`
## `tf.constant_initializer()`
## `tf.argsort()`
- `direction`: (`"ASCENDING"`, `"DESCENDING"`).
## `tf.math`
### `tf.math.add()`, `tf.math.subtract()`, `tf.math.multiply()`, `tf.math.divide()`
- Adds, substract, multiply or divide two input tensors element-wise.
### `tf.math.add_n(inputs)`
- Adds all input tensors element-wise.
- `inputs`: A list of Tensors, each with the same shape and type.
### `tf.math.square()`
- Compute square of x element-wise.
### `tf.math.sqrt()`
### `tf.math.argmax(axis)`
### `tf.math.sign`
### `tf.math.exp()`
### `tf.math.log()`
### `tf.math.equal()`
```python
seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
```
### `tf.math.reduce_sum([axis])`, `tf.math.reduce_mean()`
- Reference: https://www.tensorflow.org/api_docs/python/tf/math/reduce_sum#returns_1
- `axis=None`: Reduces all dimensions.
- Reduces `input_tensor` along the dimensions given in `axis`. Unless `keepdims=True`, the rank of the tensor is reduced by `1` for each of the entries in `axis`, which must be unique. If `keepdims=True`, the reduced dimensions are retained with length `1`.
## `tf.math.logical_and()`, `tf.math.logical_or()`
## `tf.math.logical_not(x)`
- Returns the truth value of `NOT x` element-wise.
## `tf.linalg.matmul(a, b, [transpose_a], [transpose_b])`

# Create Tensors
## `tf.Variable(initial_value, [shape=None], [trainable=True], [validate_shape=True], [dtype], [name])`
- Reference: https://www.tensorflow.org/api_docs/python/tf/Variable
- `initial_value`: This initial value defines the type and shape of the variable. After construction, the type and shape of the variable are fixed.
- [`shape`]: The shape of this variable. If `None`, the shape of `initial_value` will be used.
- `validate_shape`: If `False`, allows the variable to be initialized with a value of unknown shape. If `True`, the default, the shape of `initial_value` must be known.
- [`dtype`]: If set, `initial_value` will be converted to the given type. If `None`, either the datatype will be kept (if `initial_value` is a Tensor), or `convert_to_tensor()` will decide.
## `tf.zeros()`
```python
W = tf.Variable(tf.zeros([2, 1], dtype=tf.float32), name="weight")
```

# Layers without Weights
## `tf.stack(values, axis, [name])`
- Reference: https://www.tensorflow.org/api_docs/python/tf/stack
- Stacks a list of tensors of rank R into one tensor of rank (R + 1).
- `axis`: The axis to stack along.
- Same syntax as `np.stack()`
## Add Layers
```python
# TensorFlow
# It takes as input a list of tensors, all of the same shape, and returns a single tensor (also of the same shape).
# 마지막 Deminsion만 동일하면 Input으로 주어진 Tensors 중 하나를 옆으로 늘려서 덧셈을 수행합니다.
Add()()
```
## Multiply Layers
```python
# TensorFlow
Multiply()()
```
## `Dot(axes)`
- `axes` : (integer, tuple of integers) Axis or axes along which to take the dot product. If a tuple, should be two integers corresponding to the desired axis from the first input and the desired axis from the second input, respectively. Note that the size of the two selected axes must match.
## `Concatenate([axis])()` (= `tf.concat(values, [axis], [name])`)
## `Flatten([input_shape])`
## `Input(shape, [name], [dtype], ...)`
- `shape`
	- ***A shape tuple (integers), not including the batch size***. For instance, shape=(32,) indicates that the expected input will be batches of 32-dimensional vectors.
	- ***Elements of this tuple can be None; "None" elements represent dimensions where the shape is not known.***
	- Note that `shape` does not include the batch dimension.
## Dropout Layer
```python
# TensorFlow
# `rate`
	# The Dropout layer randomly sets input units to 0 with a frequency of `rate` at each step during training time, which helps prevent overfitting. Inputs not set to 0 are scaled up by 1/(1 - `rate`) such that the sum over all inputs is unchanged.
	# Note that the `Dropout` layer only applies when `training` is set to `True` such that no values are dropped during inference. When using `model.fit`, `training` will be appropriately set to `True` automatically, and in other contexts, you can set the kwarg explicitly to `True` when calling the layer.
Dropout(rate)

# PyTorch
Dropout(p, [inplace=False])
```
## Pooling Layer
```python
# Tensorflow
# Output Dimension
	# When `padding="valid"`: `(input_dim - pool_size)//strides + 1`
	# When `padding="same"`: `input_dim//strides + 1`
MaxPool1D(pool_size, strides, padding, [data_format]) # Same as `MaxPooling1D()`
MaxPool2D() # Same as `MaxPooling2D()`

# PyTorch
MaxPool1d()
MaxPool2d()
```
```python
# TensorFlow
# Shape: `(a, b, c, d)` -> `(a, d)`.
GlobalMaxPool1D() # Same as `GlobalMaxPooling1D()`
# Downsamples the input representation by taking the maximum value over the time dimension.
# Shape: `(a, b, c)` -> `(b, c)`.
GlobalMaxPool2D() # Same as  `GlobalMaxPooling2D()`
```
```python
# TensorFlow
AveragePooling1D([pool_size], [strides], [padding])
AveragePooling2D()

# PyTorch
AvgPool1d()
AvgPool2d()
```
## `GlobalAveragePooling1D()`, `GlobalAveragePooling2D()`
## `ZeroPadding2D(padding)`
- `padding`:
	- Int: the same symmetric padding is applied to height and width.
	- Tuple of 2 ints: interpreted as two different symmetric padding values for height and width: `(symmetric_height_pad, symmetric_width_pad)`.
	- Tuple of 2 tuples of 2 ints: interpreted as `((top_pad, bottom_pad), (left_pad, right_pad))`.
## `BatchNormalization()`
- Reference: https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization
- Usually used before activation function layers.
## `LayerNormalization([epsilon], axis)`
- Reference: https://www.tensorflow.org/api_docs/python/tf/keras/layers/LayerNormalization
- ***Normalize the activations of the previous layer for each given example in a batch independently, rather than across a batch like Batch Normalization. i.e. applies a transformation that maintains the mean activation within each example close to 0 and the activation standard deviation close to 1.***
- `epsilon`: Small float added to variance to avoid dividing by zero. Defaults to `1e-3`.
## `Reshape()`
## `Activation(activation)`
- `activation`: (`"relu"`)
## `RepeatVector(n)`
- Repeats the input `n` times.

# Layers with Weights
## Embedding Layer
```python
# TensorFlow
# Reference: https://www.tensorflow.org/api_docs/python/tf/keras/layers/Embedding
# `input_dim`: Size of the vocabulary.
# `output_dim`: Dimension of the dense embedding.
# `input_length`: Length of input sequences, when it is constant. This argument is required if you are going to connect `Flatten()` then `Dense ()` layers upstream.
# `mask_zero=True`: Whether or not the input value 0 is a special "padding" value that should be masked out. This is useful when using recurrent layers which may take variable length input. If `mask_zero` is set to `True`, as a consequence, index 0 cannot be used in the vocabulary (`input_dim` should equal to `vocab_size + 1`)).
# Shape: `(batch_size, input_length)` -> `(batch_size, input_length, output_dim)`
Embedding(input_dim, output_dim, [input_length], [mask_zero], [name], [weights], [trainable], ...)

# PyTorch
# Reference: https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html#torch.nn.Embedding
# `padding_idx`: If specified, the entries at `padding_idx` do not contribute to the gradient; therefore, the embedding vector at `padding_idx` is not updated during training, i.e. it remains as a fixed "pad”. For a newly constructed Embedding, the embedding vector at `padding_idx` will default to all zeros, but can be updated to another value to be used as the padding vector.
Embedding(num_embeddings, embedding_dim, padding_idx)
```
## Fully Connected Layer
```python
# Tensorflow
# Reference: https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense
# `units`: Dimensionality of the output space.
# `activation`: Activation function to use. If you don't specify anything, no activation is applied (ie. "linear" activation)
# Shape: `(batch_size, ..., input_dim)` -> `(batch_size, ..., units)`
# Note that after the first layer, you don't need to specify the size of the input anymore.
Dense(units, [activation])

# PyTorch
Linear(in_features, out_features)
```
## Convolution Layer
```python
# TensorFlow
# `kernal_size`: window_size
# `padding="valid"`: No padding. 
# `padding="same"`: Results in padding with zeros evenly to the left/right or up/down of the input such that output has the same height/width dimension as the input.
# `data_format`: (`"channels_last"`, `"channels_first"`)
# `activation`: (`"tanh"`)
# Output Dimension
	# When `padding="valid"`: `math.ceil(input_dim - kernel_size + 1)/strides`
	# When `padding="same"`: `math.ceil(input_dim/strides)`
Conv1D(filters, kernel_size, strides, padding, activation, data_format)
Conv2D()
Conv1DTranspose()
Conv2DTranspose()

# PyTorch
Conv1d()
Conv2d()
ConvTranspose1d()
ConvTranspose2d()
```
## LSTM
```python
# TensorFlow
# Reference: https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM
# `return_sequences`: Whether to return the last output. in the output sequence, or the full sequence.
	# `True`: 모든 timestep에서 Output을 출력합니다. (Output shape: `(batch_size, timesteps, h_size)`)
	# `False` (default): 마지막 timestep에서만 Output을 출력합니다. (Output shape: `(batch_size, h_size)`)
# `return_state`: Whether to return the last state in addition to the output. (`output, h_state, c_state = LSTM(return_state=True)()`)
# Call arguments
	# `mask`
	# `training`
	# `initial_state`: List of initial state tensors to be passed to the first call of the cell (optional, defaults to `None` which causes creation of zero-filled initial state tensors).
LSTM(units, return_sequences, return_state, [dropout])([initial_state])

# PyTorch
LSTM(input_size, hidden_size, num_layers, batch_first, dropout, bidirectional)
```
## `Bidirectional([input_shape])`
```python
z, for_h_state, for_c_state, back_h_state, back_c_state = Bidirectional(LSTM(return_state=True))(z)
```
## `TimeDistributed()`
- Reference: https://www.tensorflow.org/api_docs/python/tf/keras/layers/TimeDistributed
- This wrapper allows to apply a layer to every temporal slice of an input.
- For example, consider a batch of 32 video samples, where each sample is a 128x128 RGB image with channels_last data format, across 10 timesteps. The batch input shape is (32, 10, 128, 128, 3). You can then use `TimeDistributed()` to apply the same `Conv2D()` layer to each of the `10` timesteps, independently. Because `TimeDistributed()` applies the same instance of `Conv2D()` to each of the timestamps, the same set of weights are used at each timestamp.

# Optimizer
## Stochastic Gradient Descent (SGD)
```python
from tensorflow.keras.optimizers import SGD
```
## Adagrad
```python
from tensorflow.keras.optimizers import Adagrad
```
- Reference: https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adagrad
- *Adagrad tends to benefit from higher initial learning rate values compared to other optimizers.*
## RMSprop (Root Mean Square ...)
## Adam (ADAptive Moment estimation)
```python
# TensorFlow
Adam(learning_rate, beta_1, beta_2, epsilon, name)

# PyTorch
Adam([lr=0.001], [betas=(0.9, 0.999)], [eps=1e-08])
```

# Model
## Build Model
```python
model = Model(inputs, ouputs, [name])
model.summary()
# `to_file`: File name of the plot image.
# `show_layer_activations`: Display layer activations
plot_model(model, [to_file], [show_layer_activations])
```
## Compile
### TensorFlow
```python
# `optimizer`: (`"sgd"`, `"adam"`, `Adam(learning_rate)`, "rmsprop"`, Adagrad(learning_rate)]
# `loss`: (`"mse"`, `"mae"`, `"binary_crossentropy"`, `"categorical_crossentropy"`, `"sparse_categorical_crossentropy"`)
	# If the model has multiple outputs, you can use a different `loss` on each output by passing a dictionary or a list of `loss`es.
	# `"categorical_crossentropy"`: Produces a one-hot array containing the probable match for each category.
	# `"sparse_categorical_crossentropy"`: Produces a category index of the most likely matching category.
# `metrics`: (`["mse"]`, `["mae"]`, `["binary_accuracy"]`, `["categorical_accuracy"]`, `["sparse_categorical_crossentropy"]`, `["acc"]`)
# When you pass the strings "accuracy" or "acc", we convert this to one of ``BinaryAccuracy()`, ``CategoricalAccuracy()`, `SparseCategoricalAccuracy()` based on the loss function used and the model output shape.
# `loss_weights`: The `loss` value that will be minimized by the model will then be the weighted sum of all individual `loss`es, weighted by the `loss_weights` coefficients. 
model.compile(optimizer, loss, [loss_weights], [metrics], [loss_weights])
```
### PyTorch
```python
# `optimizer`: `SGD(model.parameters(), lr, momentum)`
# `criterion`: `BCELoss()`, `CrossEntropyLoss()`
```
## Train Model
### TensorFlow
- Reference: https://keras.io/api/models/model_training_apis/
```python
# `mode`: (`"auto"`, `"min"`, `"max"`).
	# `"min"`: Training will stop when the quantity monitored has stopped decreasing;
	# `"max"`: It will stop when the quantity monitored has stopped increasing;
	# `"auto"`: The direction is automatically inferred from the name of the monitored quantity.
# `patience`: Number of epochs with no improvement after which training will be stopped.
es = EarlyStopping(monitor="val_loss", mode="auto", verbose=1, patience=2)
model_path = "model_path.h5"
# `save_best_only=True`: `monitor` 기준으로 가장 좋은 값으로 모델이 저장됩니다.
# `save_best_only=False`: 매 epoch마다 모델이 filepath{epoch}으로 저장됩니다.
# `save_weights_only`: If `True`, then only the model's weights will be saved (`model.save_weights(filepath)`), else the full model is saved (`model.save(filepath)`).
mc = ModelCheckpoint(filepath=model_path, monitor="val_acc", mode="auto", verbose=1, save_best_only=True)
# `verbose=2`: One line per epoch. recommended.
hist = model.fit(x, y, [validation_split], [validation_data], batch_size, epochs, verbose=2, [shuffle], callbacks=[es, mc])
```
### PyTorch
```python

```
## Training History
```python
fig, axes = plt.subplots(1, 2, figsize=(12, 8))
axes[0].plot(hist.history["loss"][1:], label="loss");
axes[0].plot(hist.history["val_loss"][1:], label="val_loss");
axes[0].legend();

axes[1].plot(hist.history["acc"][1:], label="acc");
axes[1].plot(hist.history["val_acc"][1:], label="val_acc");
axes[1].legend();
```
## Evaluate Model
```python
te_loss, te_acc = model.evaluate(X_te, y_te, batch_size)
```
## Inference
### TensorFlow
- Reference: https://www.tensorflow.org/api_docs/python/tf/keras/Model?hl=en, https://stackoverflow.com/questions/60837962/confusion-about-keras-model-call-vs-call-vs-predict-methods
- `model(x)`
	- Calls the model on new inputs and returns the outputs as `tf.Tensor`s.
	- ***For small numbers of inputs that fit in one batch, directly use `__call__()` for faster execution, e.g., `model(x)`, or `model(x, training=False)` if you have layers such as `BatchNormalization()` that behave differently during inference. You may pair the individual model call with a `@tf.function()` for additional performance inside your inner loop.***
	- ***After `model(x)`, you can use `tf.Tensor.numpy()` to get the numpy array value of an eager tensor.***
	- Also, note the fact that test loss is not affected by regularization layers like noise and dropout.
- `model.predict()`
	- ***Computation is done in batches. This method is designed for batch processing of large numbers of inputs. It is not intended for use inside of loops that iterate over your data and process small numbers of inputs at a time.***
- `model.predict_on_batch()`
	- Returns predictions for a single batch of samples.
	- The difference between `model.predict()` and `model.predict_on_batch()` is that the latter runs over a single batch, and the former runs over a dataset that is splitted into batches and the results merged to produce the final `numpy.ndarray` of predictions.
### PyTorch
```python
# Evaluation (Inference) mode로 전환합니다. (`Dropout()`, `BatchNorm()`은 Evaluation mode에서는 작동하지 않습니다.)
# `model.train()`: Train mode로 전환합니다.
model.eval()

# Disabling gradient calculation is useful for inference, when you are sure that you will not call `Tensor.backward()`.
with torch.no_grad():
	...
```
## Model Methods
```python
# TensorFlow
model.inputs
model.trainable_variables
# Iterate model layers
for layer in model.layers:
	...
# Get the layer by its name
layer = model.get_layer("<layer_name>")

# PyTorch
# Iterate model layers
for layer in model.parameters():
	...
```
## Check Model Weights
```python
# TensorFlow
layer.name # Layer name
layer.output # Output
layer.input_shape # Input shape
layer.output_shape # Output shape
layer.get_weights()[0] # Weight
layer.get_weights()[1] # Bias

# PyTorch
layer.size()
```

# TensorFlow `Dataset`
- Reference: https://www.tensorflow.org/api_docs/python/tf/data/Dataset
- Dataset usage follows a common pattern:
	- Create a Reference dataset from your input data.
	- Apply dataset transformations to preprocess the data.
	- Iterate over the dataset and process the elements. Iteration happens in a streaming fashion, so the full dataset does not need to fit into memory. (Element: A single output from calling `next()` on a dataset iterator. Elements may be nested structures containing multiple components.)
- Methods
	- `as_numpy_iterator()`
		- Returns an iterator which converts all elements of the dataset to numpy.
		- This will preserve the nested structure of dataset elements.
		- This method requires that you are running in eager mode and the dataset's element_spec contains only `tf.TensorSpec` components.
	- `batch(batch_size, [drop_remainder=False])`
		- Combines consecutive elements of this dataset into batches.
		- The components of the resulting element will have an additional outer dimension, which will be `batch_size` (or `N % batch_size` for the last element if `batch_size` does not divide the number of input elements `N` evenly and `drop_remainder=False`). If your program depends on the batches having the same outer dimension, you should set the `drop_remainder=True` to prevent the smaller batch from being produced.
	- `padded_batch()`
		- Pad to the smallest per-`batch size` that fits all elements.
		- Unlike `batch()`, the input elements to be batched may have different shapes, and this transformation will pad each component to the respective shape in `padded_shapes`. The `padded_shapes` argument determines the resulting shape for each dimension of each component in an output element.
		- `padded_shapes`:
			- If `None`: The dimension is unknown, the component will be padded out to the maximum length of all elements in that dimension.
			- If not `None`: The dimension is a constant, the component will be padded out to that length in that dimension.
		- `padding_values`
		- `drop_remainder`
	- `cache(filename)`
		- Caches the elements in this dataset.
		- The first time the dataset is iterated over(e.g., `map()`, `filter()`, etc.), its elements will be cached either in the specified file or in memory. Subsequent iterations will use the cached data.
		- For the cache to be finalized, the input dataset must be iterated through in its entirety. Otherwise, subsequent iterations will not use cached data.
		- `filename`: When caching to a file, the cached data will persist across runs. Even the first iteration through the data will read from the cache file. Changing the input pipeline before the call to `cache()` will have no effect until the cache file is removed or the `filename` is changed. If a `filename` is not provided, the dataset will be cached in memory.
		- `cache()` will produce exactly the same elements during each iteration through the dataset. If you wish to randomize the iteration order, make sure to call `shuffle()` after calling `cache()`.
	- `prefetch(buffer_size)`
		- Most dataset input pipelines should end with a call to prefetch. This allows later elements to be prepared while the current element is being processed. This often improves latency and throughput, at the cost of using additional memory to store prefetched elements.
		- `buffer_size`: The maximum number of elements that will be buffered when prefetching. If the value `tf.data.AUTOTUNE` is used, then the buffer size is dynamically tuned.
	- `enumerate([start=0])`
	- `filter(predicate)`
		- `predicate`: A function mapping a dataset element to a boolean.
		- Returns the dataset containing the elements of this dataset for which `predicate` is `True`.
	- `from_tensor_slices()`
	- `from_tensors()`
	- `from_generator()`
		- `generator`: Must be a callable object that returns an object that supports the `iter()` protocol (e.g. a generator function).
		- `output_types`: A (nested) structure of `tf.DType` objects corresponding to each component of an element yielded by generator.
		- `ouput_signature`: A (nested) structure of tf.TypeSpec objects corresponding to each component of an element yielded by generator.
		```python
		def gen():
			yield ...
		dataset = tf.data.Dataset.from_generator(gen, ...)
		```
	- `map(map_func)`
		- This transformation applies `map_func` to each element of this dataset, and returns a new dataset containing the transformed elements, in the same order as they appeared in the input. `map_func` can be used to change both the values and the structure of a dataset's elements.
	- `random()`
	- `range()`
	- `repeat()`
	- `shuffle(buffer_size, [seed=None], [reshuffle_each_iteration=None])`
		- `buffer_size`: For perfect shuffling, greater than or equal to the full size of the dataset is required. If not, only the first `buffer_size` elements will be selected randomly.
		- `reshuffle_each_iteration`: Controls whether the shuffle order should be different for each epoch.
	- `skip(count)`
	- `take(count)`
	- `unique()`
	- `zip()`

# PyTorch `DataLoader`
```python
dl_tr = DataLoader(dataset, batch_size, [shuffle=False], [num_workers=0], [prefetch_factor=2])
...

next(iter(dl_tr))
```

# Save or Load Model
## TensorFlow
- Reference: https://www.tensorflow.org/tutorials/keras/save_and_load
```python
save_dir = Path("...")
model_path = save_dir / "model_name.h5"
hist_path = save_dir / "model_name_hist.npy"
if os.path.exists(model_path):
    model = load_model(model_path)
    hist = np.load(hist_path, allow_pickle="TRUE").item()
else:
	...
	# The weight values
	# The model's architecture
	# The model's training configuration (what you pass to the .compile() method)
	# The optimizer and its state, if any (this enables you to restart training where you left off)
	model.save(model_path)
	np.save(hist_path, hist.history)
```
## PyTorch
- Reference: https://pytorch.org/tutorials/beginner/saving_loading_models.html
```python
save_dir = Path("...")
model_path = save_dir / "model_name.pth"
# hist_path = save_dir / "model_name_hist.npy"
if os.path.exists(model_path):
	weights = torch.load(model_path)
    model.load_state_dict(weights)
else:
	...
	# Loads a model’s parameter dictionary using a deserialized `state_dict()`.
	# In PyTorch, the learnable parameters (i.e. weights and biases) of an `torch.nn.Module` model are contained in the model’s parameters (accessed with `model.parameters()`). A `state_dict()` is simply a Python dictionary object that maps each layer to its parameter tensor. Note that only layers with learnable parameters (convolutional layers, linear layers, etc.) and registered buffers (batchnorm’s running_mean) have entries in the model’s `state_dict()`. Optimizer objects (`torch.optim``) also have a `state_dict()`, which contains information about the optimizer’s state, as well as the hyperparameters used.
	weights = model.state_dict()
	torch.save(weights, model_path)
```

# Save or Load Weights
## TensorFlow
```python
model.compile(...)
...
model.load_weights(model_path)
```
- As long as two models share the same architecture you can share weights between them. So, when restoring a model from weights-only, create a model with the same architecture as the original model and then set its weights.

# Custrom Model
## TensorFlow
- Reference: https://www.tensorflow.org/api_docs/python/tf/keras/Model
```python
class ModelName(Model):
	# You should define your layers in `__init__()`.
	def __init__(self, ...):
		super().__init__()
		self.var1 = ...
		self.var2 = ...
		...
	# You should implement the model's forward pass in `__call__()`.
	# If you subclass `Model`, you can optionally have a `training` argument (boolean) in `__call__()`, which you can use to specify a different behavior in training and inference.
	def __call__(self, ..., [training]):
		...
		return ...
...
model = ModelName()
```
## PyTorch
```python
class ModelName(nn.Module):
	def __init__(self, ...):
		super().__init__()
		# Or `super(ModelName, self).__init__()`
		self.var1 = ...
		self.var2 = ...
		...
	def forward(self, x):
		...
...
model = ModelName()
```

# Custom Layer
## TensorFlow
```python
class LayerName(Layer):
	def __init__(self, ...):
		super().__init__()
		self.var1 = ...
		self.var2= ...
		...
	def __call__(self, ...):
		...
		return ...
```

# Custom Learning Rate
## TensorFlow
```python
class LearningRate(LearningRateSchedule):
    def __init__(self, warmup_steps=4000):
        super(self).__init__()

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        return (d_model**-0.5)*tf.math.minimum(step**-0.5, step*(self.warmup_steps**-1.5))

lr = LearningRate()
```

# Import Machine Learning Libraries
## scikit-learn
```python
from sklearn.manifold import TSNE
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, KFold, GroupKFold, LeaveOneOut, LeaveOneGroupOut
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler, Normalizer
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
```
## TensorFLow
```python
import tensorflow as tf
from tensorflow.keras import Input, Model, Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer, Dense, Flatten, Dropout, Concatenate, Add, Dot, Multiply, Reshape, Activation, BatchNormalization, LayerNormalization, SimpleRNNCell, RNN, SimpleRNN, LSTM, Embedding, Bidirectional, TimeDistributed, Conv1D, Conv2D, Conv1DTranspose, Conv2DTranspose, MaxPool1D, MaxPool2D, GlobalMaxPool1D, GlobalMaxPool2D, AveragePooling1D, AveragePooling2D, GlobalAveragePooling1D, GlobalAveragePooling2D, ZeroPadding2D, RepeatVector, Resizing, Rescaling, RandomContrast, RandomCrop, RandomFlip, RandomRotation, RandomTranslation, RandomZoom, RandomWidth, RandomHeight, RandomBrightness
from tensorflow.keras.utils import get_file, to_categorical, plot_model, image_dataset_from_directory
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from tensorflow.keras.optimizers import SGD, Adagrad, Adam
from tensorflow.keras.optimizers.schedules import LearningRateSchedule
# MeanSquaredError, MeanAbsoluteError, MeanAbsolutePercentageError, BinaryCrossentropy, CategoricalCrossentropy, SparseCategoricalCrossentropy, CosineSimilarity
from tensorflow.keras import losses
# MeanSquaredError, RootMeanSquaredError, MeanAbsoluteError, MeanAbsolutePercentageError, BinaryCrossentropy, CategoricalCrossentropy, SparseCategoricalCrossentropy, TopKCategoricalAccuracy, SparseTopKCategoricalAccuracy, CosineSimilarity
from tensorflow.keras import metrics
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.activations import linear, sigmoid, relu
from tensorflow.keras.initializers import RandomNormal, glorot_uniform, he_uniform, Constant
```
## PyTorch
```python
import torch
from torch.nn import Module, Linear, Dropout, Conv1d, Conv2d, ConvTranspose1d, ConvTranspose2d, MaxPool1d, MaxPool2d, AvgPool1d, AvgPool2d, CrossEntropyLoss()
import torch.nn.functional as F
from torch.optim import SGD, RMSprop, Adagrad, Adam
from torch.utils.data import Dataset, DataLoader

import torchvision
import torchvision.dataset
import torchvision.transforms as transforms
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