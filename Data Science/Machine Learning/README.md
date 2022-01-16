# Dataset & Data Set
- Source: https://english.stackexchange.com/questions/2120/which-is-correct-dataset-or-data-set
- Dataset for certain datasets
- Data set for any set for data in general.

# Data Density (or Sparsity)
- Source: https://datascience.foundation/discussion/data-science/data-sparsity
- In a database, sparsity and density describe the number of cells in a table that are empty (sparsity) and that contain information (density), though sparse cells are not always technically empty—they often contain a “0” digit.
## Sparse Matrix & Dense Matrix
- Source: https://en.wikipedia.org/wiki/Sparse_matrix
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

# Real Data & Test Data

# Embedding
- Source: https://analyticsindiamag.com/machine-learning-embedding/#:~:text=An%20embedding%20is%20a%20low,of%20a%20high%2Ddimensional%20vector.&text=Embedding%20is%20the%20process%20of,the%20two%20are%20semantically%20similar.
- *Embedding is the process of converting high-dimensional data to low-dimensional data in the form of a vector in such a way that the two are semantically similar.*
- Embeddings of neural networks are advantageous because they can lower the dimensionality of categorical variables and represent them meaningfully in the altered space.
- Source: https://developers.google.com/machine-learning/crash-course/embeddings/video-lecture
- An embedding is a relatively low-dimensional space into which you can translate high-dimensional vectors. Embeddings make it easier to do machine learning on large inputs like sparse vectors representing words. Ideally, an embedding captures some of the semantics of the input by placing semantically similar inputs close together in the embedding space. An embedding can be learned and reused across models.

# Datasets
## `mnist`
```python
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
```
## `reuters`
```python
(X_train, y_train), (X_test, y_test) = reuters.load_data(num_words=None, test_split=0.2)
```
## `cifar10`
```python
(x_tr, y_tr), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
```

# Feature Scaling
- Source: https://en.wikipedia.org/wiki/Feature_scaling
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
```python
x_new = (x - min(X))/(max(X) - min(X))
```
```python
from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler()
```
## Standard Scaling
```python
import numpy as np

x_new = (x - np.mean(X))/np.std(X)
```
```python
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
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
- Source: https://en.wikipedia.org/wiki/Artificial_neural_network
- A hyperparameter is a constant parameter whose value is set before the learning process begins. The values of parameters are derived via learning.

# Learning
- Source: https://en.wikipedia.org/wiki/Artificial_neural_network
- Learning is the adaptation of the network to better handle a task by considering sample observations. *Learning involves adjusting the weights (and optional thresholds) of the network to improve the accuracy of the result. This is done by minimizing the observed errors. Learning is complete when examining additional observations does not usefully reduce the error rate. Even after learning, the error rate typically does not reach 0.* If after learning, the error rate is too high, the network typically must be redesigned. Practically this is done by defining a cost function that is evaluated periodically during learning. *As long as its output continues to decline, learning continues.*
## Learning Rate
- *The learning rate defines the size of the corrective steps that the model takes to adjust for errors in each observation. A high learning rate shortens the training time, but with lower ultimate accuracy, while a lower learning rate takes longer, but with the potential for greater accuracy. In order to avoid oscillation inside the network such as alternating connection weights, and to improve the rate of convergence, refinements use an adaptive learning rate that increases or decreases as appropriate. The concept of momentum allows the balance between the gradient and the previous change to be weighted such that the weight adjustment depends to some degree on the previous change. A momentum close to 0 emphasizes the gradient, while a value close to 1 emphasizes the last change.*

# Discriminative & Generative Model
- Source: https://analyticsindiamag.com/what-are-discriminative-generative-models-how-do-they-differ/
- *Discriminative models draw boundaries in the data space, while generative ones model how data is placed throughout the space. Mathematically speaking, a discriminative machine learning trains a model by learning parameters that maximize the conditional probability P(Y|X), but a generative model learns parameters by maximizing the joint probability P(X,Y).*
## Discriminative Model
- ***The discriminative model is used particularly for supervised machine learning. Also called a conditional model, it learns the boundaries between classes or labels in a dataset. It creates new instances using probability estimates and maximum likelihood. However, they are not capable of generating new data points. The ultimate goal of discriminative models is to separate one class from another.***
## Generative Model
- ***Generative models are a class of statistical models that generate new data instances. These models are used in unsupervised machine learning to perform tasks such as probability and likelihood estimation, modelling data points, and distinguishing between classes using these probabilities. Generative models rely on the Bayes theorem to find the joint probability.***

# Convergence
- Source: https://en.wikipedia.org/wiki/Artificial_neural_network
- ***Models may not consistently converge on a single solution, firstly because local minima may exist, depending on the cost function and the model. Secondly, the optimization method used might not guarantee to converge when it begins far from any local minimum. Thirdly, for sufficiently large data or parameters, some methods become impractical.***

# Batch Normalization
- Source: https://en.wikipedia.org/wiki/Batch_normalization 
- Batch normalization (also known as batch norm) is a method used to make artificial neural networks faster and more stable through normalization of the layers' inputs by re-centering and re-scaling. It was proposed by Sergey Ioffe and Christian Szegedy in 2015.[1]
While the effect of batch normalization is evident, the reasons behind its effectiveness remain under discussion. It was believed that it can mitigate the problem of internal covariate shift, where parameter initialization and changes in the distribution of the inputs of each layer affect the learning rate of the network.[1] Recently, some scholars have argued that batch normalization does not reduce internal covariate shift, but rather smooths the objective function, which in turn improves the performance.[2] However, at initialization, batch normalization in fact induces severe gradient explosion in deep networks, which is only alleviated by skip connections in residual networks.[3] Others sustain that batch normalization achieves length-direction decoupling, and thereby accelerates neural networks.[4] More recently a normalize gradient clipping technique and smart hyperparameter tuning has been introduced in Normalizer-Free Nets, so called "NF-Nets" which mitigates the need for batch normalization.[5][6

# MLOps
- Source: https://en.wikipedia.org/wiki/MLOps
- MLOps or ML Ops is a set of practices that aims to deploy and maintain machine learning models in production reliably and efficiently.[1] The word is a compound of "machine learning" and the continuous development practice of DevOps in the software field. Machine learning models are tested and developed in isolated experimental systems. When an algorithm is ready to be launched, MLOps is practiced between Data Scientists, DevOps, and Machine Learning engineers to transition the algorithm to production systems.

# Google Colab
## Mount Google Drive
```python
from google.colab import drive
import os
import sys

drive.mount("/content/drive", force_remount=True)
try:
    my_path = "/content/notebooks"
    os.symlink("/content/drive/MyDrive/ColabNotebooks/my_env", my_path)
    sys.path.insert(0, my_path)
except:
    print("Failed!")
os.chdir(my_path)
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
## Prevent from Disconnecting.
```
function ClickConnect(){
    console.log("코랩 연결 끊김 방지");
	document.querySelector("colab-toolbar-button#connect").click()}
setInterval(ClickConnect, 60*1000)
```
## Install Libraries Permanently
```python
!pip install --target=$my_path LIBRARY_NAME
```
## Use TPU
```python
resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu="grpc://" + os.environ["COLAB_TPU_ADDR"])
tf.config.experimental_connect_to_cluster(resolver)
tf.tpu.experimental.initialize_tpu_system(resolver)
```
```python
strategy = tf.distribute.experimental.TPUStrategy(resolver)
with strategy.scope():
    model = create_model()
    hist = model.fit()
```
## Install `khaiii`
```python
!git clone https://github.com/kakao/khaiii.git
!pip install cmake
!mkdir build
!cd build && cmake /content/khaiii
!cd /content/build/ && make all
!cd /content/build/ && make resource
!cd /content/build && make install
!cd /content/build && make package_python
!pip install /content/build/package_python
```

# Activation Function
- Source: https://leedakyeong.tistory.com/entry/%EB%B0%91%EB%B0%94%EB%8B%A5%EB%B6%80%ED%84%B0-%EC%8B%9C%EC%9E%91%ED%95%98%EB%8A%94-%EB%94%A5%EB%9F%AC%EB%8B%9D-%ED%99%9C%EC%84%B1%ED%99%94%ED%95%A8%EC%88%98%EB%9E%80-What-is-activation-function?category=845638
- 활성화함수는 꼭 비선형 함수이어야 한다. 선형 함수를 사용하면 신경망의 층을 깊게 쌓는 것에 의미가 없어지기 때문이다. 그 이유는 예를 들어, 활성화 함수를 h(x) = cx 라는 선형함수라 해보자. 층으로 구성된 네트워크라 할 때, y(x) = h(h((x))) = c\*c\*c\*x = c^3\*x이다. 이는 곧 y = ax에서 a=c^3과 같다. 즉, 기껏 3층이나 쌓았지만 1층만 쌓은 네트워크와 같아진다. 이것이 바로 활성함수의 역할이다.
- Source: 
http://blog.naver.com/PostView.nhn?blogId=wideeyed&logNo=221017173808
- 2개를 분류하는 문제일 때는 Vanishing Gradient Problem때문에 sigmoid는 잘 사용하지 않고 ReLU와 그 변형된 활성화함수를 주로 이용한다. 3개 이상을 분류할 때 주로 Softmax와 그 변형된 활성화함수를 주로 이용한다.
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
- Source: http://blog.naver.com/PostView.nhn?blogId=wideeyed&logNo=221021710286&parentCategoryNo=&categoryNo=&viewDate=&isShowPopularPosts=false&from=postView
## Categorical Cross-Entropy Loss
- Source : https://wordbe.tistory.com/entry/ML-Cross-entropyCategorical-Binary의-이해
- Softmax activation 뒤에 Cross-Entropy loss를 붙인 형태로 주로 사용하기 때문에 Softmax loss 라고도 불립니다. → Multi-class classification에 사용됩니다.
우리가 분류문제에서 주로 사용하는 활성화함수와 로스입니다. 분류 문제에서는 MSE(mean square error) loss 보다 CE loss가 더 빨리 수렴한 다는 사실이 알려져있습니다. 따라서 multi class에서 하나의 클래스를 구분할 때 softmax와 CE loss의 조합을 많이 사용합니다.
## `tf.keras.activations.linear()`
## `tf.keras.activations.relu()`(= `"relu"`)

# Back Propogation
- Source: https://sacko.tistory.com/19
- 지난 시간까지는 Input에서 Output으로 가중치를 업데이트하면서 활성화 함수를 통해서 결과값을 가져오는 것까지 배웠다. 이렇게 쭉 오는 것을 순전파(foward)라고 하며 말 그대로 앞쪽으로 input 값을 전파, 보내는 것이라고 보면 된다. 하지만 우리가 임의로 한 번 순전파 했다고 출력 값이 정확하지는 않을 것이다. 우리가 임의로 설정한 가중치 값이 input에 의해서 한 번 업데이트 되긴 했지만 많은 문제가 있을 수 있다.
역전파 방법은 결과 값을 통해서 다시 역으로 input 방향으로 오차를 다시 보내며 가중치를 재업데이트 하는 것이다. 물론 결과에 영향을 많이 미친 노드(뉴런)에 더 많은 오차를 돌려줄 것이다.
- 결과값은 오차(error)를 가지게 되는데 역전파는 이 오차(error)를 다시 역방향으로 hidden layer와 input layer로 오차를 다시 보내면서 가중치를 계산하면서 output에서 발생했던 오차를 적용시키는 것이다.
한 번 돌리는 것을 1 epoch 주기라고 하며 epoch를 늘릴 수록 가중치가 계속 업데이트(학습)되면서 점점 오차가 줄어나가는 방법이다. 
- 위의 그림을 보면 Output layer에서 나온 결과 값이 가진 오차가 0.6이라고 되어 있다. 이전 노드(뉴런에서) Output layer에 각각 3, 2라는 값을 전달하였기 때문에 Ouput의 Error에 위 노드는 60%, 아래 노드는 40% 영향을 주었다고 볼 수 있다. 균등하게 가중치를 나눠줄 수 있지만 영향을 미친 만큼 오차를 다시 역전파하는게 맞는 것 같다.
error 0.6을 0.6, 0.4를 곱하니  위 노드에는 error가 0.36이, 아래 노드에는 0.24가 전달된다. 
오차 역전파는 말 그대로 이렇게 오차를 점점 거슬러 올라가면서 다시 전파하는 것을 의미한다.
- 앞에서 오차가 역전파되는 것을 알아보았는데 오차를 역전파하여 계속 업데이트 하는 이유는 신경망을 통해 더 나은 결과 값을 내기 위해서 weight를 조정하는데 오차가 영향을 주기 때문이다. 위의 예처럼 간단한 신경망이라면 오차를 계산하는 식은 굉장히 간단할 것이지만 효과적인 신경망은 절대 저렇게 간단하지 않다. 수십, 수백개의 노드(뉴런)이 연결되어서 수많은 가중치의 조합으로 특정 노드의 weight를 계산해야 한다면... 효율적인 방법이 필요할 것이다.
경사하강법은 너무나 많은 신경망 안의 가중치 조합을 모두 계산하면 시간이 오래걸리기 때문에 효율적으로 이를 하기위해 고안된 방법입이다.
- Source: https://sacko.tistory.com/39?category=632408
- 지난 오차역전파 관련 포스팅에서는 오차역전파법이 순전파(foward propagation)로 가중치 학습이 되고 이를 갱신하기 위해서 오차를 반영하여 반대 방향에서 다시 가중치를 업데이트 한다는 식으로만 설명을 했다. 역전파를 사용하는 또 다른 중요한 이유는 역전파를 통해서 '미분'을 효율적으로 계산할 수 있다는 것이다.

# Gradient Descent
- Source: https://en.wikipedia.org/wiki/Gradient_descent
- ***Gradient descent is a first-order iterative optimization algorithm for finding a local minimum of a differentiable function. The idea is to take repeated steps in the opposite direction of the gradient (or approximate gradient) of the function at the current point, because this is the direction of steepest descent.*** Conversely, stepping in the direction of the gradient will lead to a local maximum of that function; the procedure is then known as gradient ascent.
## Optimizers
### Stochastic Gradient Descent (SGD)
```python
from tensorflow.keras.optimizers import SGD
```
### Adam
```python
from tensorflow.keras.optimizers import Adam
```
### Adagrad
```python
from tensorflow.keras.optimizers import Adagrad
```
- Source: https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adagrad
- *Adagrad tends to benefit from higher initial learning rate values compared to other optimizers.*

# Categorical Variables
- Sources: https://homeproject.tistory.com/4, http://blog.naver.com/PostView.nhn?blogId=choco_9966&logNo=221374544814&parentCategoryNo=&categoryNo=77&viewDate=&isShowPopularPosts=false&from=postView, https://dailyheumsi.tistory.com/120, https://towardsdatascience.com/all-about-categorical-variable-encoding-305f3361fd02
## One-hot Encoding
- One-Hot encoding should not be performed if the number of categories are high. This would result in a sparse data.
- Decision trees does not require doing one-hot encoding. Since xgboost, AFAIK, is a boosting of decision trees, I assume the encoding is not required.
- 피처내 값들이 서로 분리 되어있기 때문에, 우리가 모를 수 있는 어떤 관계나 영향을 주지 않는다.
- features 내 값의 종류가 많을 경우(High Cardinaliry), 매우 많은 Feature 들을 만들어 낸다. 이는, 모델 훈련의 속도를 낮추고 훈련에 더 많은 데이터를 필요로 하게 한다.(차원의 저주 문제)
- 단순히 0과 1로만 결과를 내어 큰 정보이득 없이 Tree 의 depth 만 깊게 만든다. 중요한건, Tree Depth 를 증가시키는 것에 비해, 2가지 경우로만 트리를 만들어 나간다는 것이다.
- Random Forest 와 같이, 일부 Feature 만 Sampling 하여 트리를 만들어나가는 경우, One-hot Feature 로 생성된 Feature 의 수가 많기 때문에 이 features가 다른 features보다 더 많이 쓰인다.
```python
import tensorflow as tf

tf.keras.utils.to_categorical([2, 5, 1, 6, 3, 7])
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
- If we consider in the temperature scale as the order, then the ordinal value should from cold to “Very Hot. “ Ordinal encoding will assign values as ( Cold(1) <Warm(2)<Hot(3)<”Very Hot(4)). Usually, we Ordinal Encoding is done starting from 1.
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

# Splitting Dataset
- Source: https://developers.google.com/machine-learning/crash-course/training-and-test-sets/splitting-data
## Random Sampling
- The benefit of this approach is that we can see how the model reacts to previously unseen data.
However, what if one subset of our data only have people of a certain age or income levels? This is typically referred to as a sampling bias:
Sampling bias is systematic error due to a non-random sample of a population, causing some members of the population to be less likely to be included than others, resulting in a biased sample.
- If only use a train/test split, then I would advise comparing the distributions of your train and test sets. If they differ significantly, then you might run into problems with generalization. Use Facets to easily compare their distributions.
- Using `sklearn.model_selection.train_test_split()`
	```python
	from sklearn.model_selection import train_test_split

	tr_X, te_X, tr_y, te_y = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=3)
	```
	- `stratif`: If not None, data is split in a stratified fashion, using this as the class labels.
	
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
## Regression Problem
### MSE(Mean Squared Error)
### RMSE(Root Mean Squared Error)
### MAE(Mean Absolute Error)
### MPE(Mean Percentage Error)
### MAPE(Mean Absolute Percentage Error)
### R-Squared
- Source: https://statisticsbyjim.com/regression/r-squared-invalid-nonlinear-regression/
- Explained variance + Error variance = Total variance.
- However, this math works out correctly only for linear regression models. In nonlinear regression, these underlying assumptions are incorrect. Explained variance + Error variance DO NOT add up to the total variance! The result is that R-squared isn’t necessarily between 0 and 100%.
- If you use R-squared for nonlinear models, their study indicates you will experience the following problems:
R-squared is consistently high for both excellent and appalling models.
R-squared will not rise for better models all of the time.
If you use R-squared to pick the best model, it leads to the proper model only 28-43% of the time.
### Adjusted R-Squared
### RMSLE(Root Mean Squared Logarithmic Error)
- Source: https://shryu8902.github.io/machine%20learning/error/
## Classification Problem
### Confusion Matrix
- Source: https://datascienceschool.net/view-notebook/731e0d2ef52c41c686ba53dcaf346f32/
- 정답 클래스와 예측 클래스의 일치 여부를 센 결과. 정답 클래스는 행(row)으로 예측한 클래스는 열(column)로 나타낸다.
### Binary Classification Problem
- A, B 클래스 중 B 클래스를 맞히는 문제라고 가정했을 때
### Accuracy(정확도)
- 전체 샘플 중 A 또는 B라고 맞게 예측한 샘플 수의 비율
### Precision(정밀도)
- B 클래스에 속한다고 출력한 샘플 중 실제로 B 클래스에 속하는 샘플 수의 비율
### Recall(재현율)
- 실제 B 클래스에 속한 표본 중에 B 클래스에 속한다고 출력한 표본의 수의 비율
### F1 Ccore(정밀도와 재현율의 조화 평균)
### Fall-out(위양성률)
- 실제 B 클래스에 속하지 않는 표본 중에 B 클래스에 속한다고 출력한 표본의 비율
- Source: https://datascienceschool.net/view-notebook/731e0d2ef52c41c686ba53dcaf346f32/
### ROC(Receiver Operator Characteristic) Curve
- Source: https://datascienceschool.net/view-notebook/731e0d2ef52c41c686ba53dcaf346f32/
- 위에서 설명한 각종 평가 점수 중 재현율(recall)과 위양성률(fall-out)은 일반적으로 양의 상관 관계가 있다.
재현율을 높이기 위해서는 양성으로 판단하는 기준(threshold)을 낮추어 약간의 증거만 있어도 양성으로 판단하도록 하면 된다. 그러나 이렇게 되면 음성임에도 양성으로 판단되는 표본 데이터가 같이 증가하게 되어 위양성율이 동시에 증가한다. 반대로 위양성율을 낮추기 위해 양성을 판단하는 기준을 엄격하게 두게 되면 증거 부족으로 음성 판단을 받는 표본 데이터의 수가 같이 증가하므로 재현율이 떨어진다.
- 클래스 판별 기준값의 변화에 따른 위양성률(fall-out)과 재현율(recall)의 변화를 시각화한 것이다.

# Data Augmentation
## DAR
- Source: http://faculty.bscb.cornell.edu/~hooker/darpaper.pdf
- Our motivation for DAR arose from the problem of extrapolation when using machine learning methods for prediction. Methods such as trees and
neural networks are guaranteed to give predictions in a bounded interval. The variance of these predictions may, nonetheless, be severe in regions far from observed data. 
## DARE
- Source: http://faculty.bscb.cornell.edu/~hooker/DARE.pdf
- The requirement that a function return to a base model away from training data is not easy to implement as part of the learning procedure in a universal approximator. However, we can achieve this behavior by stochastically generating uniform data with response given by the base model and adding it to the training data. The resulting procedure, which we termed Data-Augmented Regression for Extrapolation (DARE) can supplement any regression method.
- In this paper we have considered the problem of making predictions at points of extrapolation. Few learning procedures are designed to produce stable results at points of extrapolation and we show that even constant extrapolators can exhibit high variance away from training data. In order to stabilize these predictions and recognize their semi-arbitrary nature, we propose that predictions should be shrunk toward a base model in proportion to the density of training points near them. This follows the heuristic argument that as new examples get further away from known examples, our model predictions becomes less informed about the response. In order to carry out this shrinkage, we propose a very simple procedure of generating new uniformly distributed data, giving it the response associated with the base model and augmenting the training set with this data. Unless strong prior knowledge is available that pertains to the whole space, we recommend that an appropriate base model should be constant. This idea has the advantage that it can be applied to any learning method. Viewed from a Bayesian perspective, our method amounts to placing a random field prior on prediction values, basing predictions on a null model unless empirical data – in the form of nearby training data – provides evidence to the contrary. We also show that when linear regression is employed, our method is a stochastic form of ridge regression. The extent to which DARE regularizes will depend on the flexibility of the learner that is employed and the concentration of the training examples in predictor space. These are the factors that also influence the extent of our concern about extrapolation. Regularization
often also has a positive effect on predictive accuracy. We have demonstrated on simulated and real examples that it is possible to simultaneously improve predictive accuracy on the data distribution and stability at points of extrapolation.
## SMOGN(Synthetic Minority Over-Sampling Technique for Regression with Gaussian Noise)
- Source: https://github.com/nickkunz/smogn/blob/master/examples/smogn_example_3_adv.ipynb
- Here we cover the focus of this example. We call the smoter function from this package (smogn.smoter) and satisfy all of the available arguments for manual operation: data, y, samp_method, drop_na_col, drop_na_row, replace, k, rel_thres, rel_method, rel_ctrl_pts_rg
- The data argument takes a Pandas DataFrame, which contains the training set split. In this example, we input the previously loaded housing training set with follow input: data = housing
- The y argument takes a string, which specifies a continuous reponse variable by header name. In this example, we input 'SalePrice' in the interest of predicting the sale price of homes in Ames, Iowa with the following input: y = 'SalePrice'
- The k argument takes a positive integer less than 𝑛, where 𝑛 is the sample size. k specifies the number of neighbors to consider for interpolation used in over-sampling. In this example, we input 7 to consider 2 additional neighbors (default is 5) with the following input: k = 7
- The pert argument takes a real number between 0 and 1. It represents the amount of perturbation to apply to the introduction of Gaussian Noise. In this example, we input 0.04 to increase the noise generated by synthetic examples where applicable (default is 0.02). We utilize the following input: pert = 0.04
- The samp_method argument takes a string, either 'balance' or 'extreme'. If 'balance' is specified, less over/under-sampling is conducted. If 'extreme' is specified, more over/under-sampling is conducted. In this case, we input 'balance' (default is 'balance') with the following input: samp_method = 'balance'
- The drop_na_col and drop_na_row arguments take a boolean. They specify whether or not to automatically remove features (columns) and observations (rows) that contain missing values (default is True for both). In this example, we make the argument explicit with the following inputs: drop_na_col = True and drop_na_row = True
- The replace argument takes a boolean. It specifies whether or not to utilize replacement in under-sampling (default is False). In this example, we make the argument explicit with the following input: replace = False
- The rel_thres argument takes a real number between 0 and 1. It specifies the threshold of rarity. The higher the threshold, the higher the over/under-sampling boundary. The inverse is also true, where the lower the threshold, the lower the over/under-sampling boundary. In this example, we dramatically reduce the boundary to 0.10 (default is 0.50) with the following input: rel_thres = 0.10
- The rel_method argument takes a string, either 'auto' or 'manual'. It specifies how relevant or rare "minority" values in y are determined. If 'auto' is specified, "minority" values are automatically determined by box plot extremes. If 'manual' is specified, "minority" values are determined by the user. In this example, we input 'manual' with the following input: rel_method = 'manual'
- The rel_ctrl_pts_rg argument takes a 2d array (matrix). It is used to manually specify the regions of interest or rare "minority" values in y. The first column indicates the y values of interest, the second column indicates a mapped value of relevance, either 0 or 1, where 0 is the least relevant and 1 is the most relevant, and the third column is indicative. It will be adjusted afterwards, use 0 in most cases.
The specified relevance values mapped to 1 are considered "minorty" values and are over-sampled. The specified relevance values mapped to 0 are considered "majority" values and are under-sampled.

# Decision Tree Algorithms
- Source: https://dailyheumsi.tistory.com/113?category=815369
## Feature Importance
- Source: https://stats.stackexchange.com/questions/162162/relative-variable-importance-for-boosting
- sum up the feature importances of the individual trees, then divide by the total number of trees.
- return the fmap, which has the counts of each time a  variable was split on
- How many times was this variable split on?
## Data Skewness
- Parametric methods are mainly based on the assumptions on the distribution of the data. They estimate a parameter (usually mean , sd ) from the sample data and is used in the modelling framework.
Point to ponder - Mean for a normal distribution will be different than mean for a right skewed distribution hence affecting how your model performs.
In Non Parametric methods no such feature of distribution is used for modelling. Primarily in Decision trees (say CART) it takes into account which variable/split brings in maximum difference in the two branches(eg - Gini) . In such a case , the distribution does not really matter.
- A positive aspect of using the error ratio instead of the error difference is that the feature importance measurements are comparable across different problems.
## Ensemble
- Source: https://lsjsj92.tistory.com/543?category=853217
## XGBoost
- Source: https://bcho.tistory.com/1354](https://bcho.tistory.com/1354, https://www.datacamp.com/community/tutorials/xgboost-in-python, https://xgboost.readthedocs.io/en/latest/parameter.html, https://brunch.co.kr/@snobberys/137, https://www.datacamp.com/community/tutorials/xgboost-in-python
### early_stopping
- Source: https://xgboost.readthedocs.io/en/latest/python/python_intro.html
- If there’s more than one metric in the eval_metric parameter given in params, the last metric will be used for early stopping.
### Customized Objective Function, Customized Metric Function
- Source: https://xgboost.readthedocs.io/en/latest/tutorials/custom_metric_obj.html
### Hyperparameters Tunning
- Source: https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/
#### `booster` [default=gbtree]
- Select the type of model to run at each iteration. It has 2 options:
gbtree: tree-based models
gblinear: linear models
#### `silent` [default=0]:
- Silent mode is activated is set to 1, i.e. no running messages will be printed. It’s generally good to keep it 0 as the messages might help in understanding the model.
#### `eta` [default=0.3] (= `learning_rate`)
- Analogous to learning rate in GBM
- Makes the model more robust by shrinking the weights on each step
- Typical final values to be used: 0.01-0.2
#### `min_child_weight` [default=1]
- Used to control over-fitting. **Higher values prevent a model from learning relations which might be highly specific to the particular sample selected for a tree.**
- Too high values can lead to under-fitting hence, **it should be tuned using CV.**
#### `max_depth` [default=6]
- Used to control over-fitting as higher depth will allow model to learn relations very specific to a particular sample.
- ***Should be tuned using CV.***
#### `max_leaf_nodes`
- Can be defined in place of max\_depth. Since binary trees are created, a depth of ‘n’ would produce a maximum of 2^n leaves. If this is defined, GBM will ignore max\_depth.
#### `gamma` [default=0]
- A node is split only when the resulting split gives a positive reduction in the loss function. Gamma specifies the minimum loss reduction required to make a split.
- Makes the algorithm conservative. The values can vary depending on the loss function and **should be tuned**.
#### `max_delta_step` [default=0]
- In maximum delta step we allow each tree’s weight estimation to be. If the value is set to 0, it means there is no constraint. If it is set to a positive value, it can help making the update step more conservative.
- Usually this parameter is not needed, but it might help in logistic regression when class is extremely imbalanced.
This is generally not used but you can explore further if you wish.
#### `subsample` [default=1]
- Same as the subsample of GBM. Denotes the fraction of observations to be randomly sampled for each tree.
- **Lower values make the algorithm more conservative and prevents overfitting but too small values might lead to under-fitting.**
- Typical values: 0.5-1
#### `colsample_bytree` [default=1]
- Similar to max_features in GBM. Denotes the fraction of columns to be randomly sampled for each tree.
Typical values: 0.5-1
#### `colsample_bylevel` [default=1]
- Denotes the subsample ratio of columns for each split, in each level. I don’t use this often because subsample and colsample_bytree will do the job for you. but you can explore further if you feel so.
#### `lambda` [default=1] (=`reg_lambda`)
- L2 regularization term on weights (analogous to Ridge regression) .This is used to handle the regularization part of XGBoost. Though many data scientists don’t use it often, **it should be explored to reduce overfitting.**
#### `alpha` [default=0]==reg_alpha
- L1 regularization term on weight (analogous to Lasso regression). **Can be used in case of very high dimensionality so that the algorithm runs faster when implemented.**
#### `scale_pos_weight` [default=1]
- **A value greater than 0 should be used in case of high class imbalance as it helps in faster convergence.**
#### `n_jobs`
- int Number of parallel threads used to run xgboost.
#### General Approach for Parameter Tuning
1. Choose a relatively **high learning rate**. Generally a learning rate of 0.1 works but somewhere between 0.05 to 0.3 should work for different problems. Determine the optimum **number of trees for this learning rat**e. XGBoost has a very useful function called as “cv” which performs cross-validation at each boosting iteration and thus returns the optimum number of trees required.
2. Tune tree-specific parameters (**max\_depth, min\_child\_weight, gamma, subsample, colsample\_bytree**) for decided learning rate and number of trees. Note that we can choose different parameters to define a tree and I’ll take up an example here.
3. Tune **regularization parameters(lambda, alpha)** for xgboost which can help reduce model complexity and enhance performance.
4. **Lower the learning rate and decide the optimal parameters**.
Let us look at a more detailed step by step approach.
## Random Forest
- Source: https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance.html?fbclid=IwAR1bPn1xldrpum8FWZEte0M7wPi9kE3BuFjSsyG9B4jSS5Th4oBkujEenNc
- the impurity-based feature importance of random forests suffers from being computed on statistics derived from the training dataset: the importances can be high even for features that are not predictive of the target variable, as long as the model has the capacity to use them to overfit.


# Feature Importance
## Permutation Feature Importance
- Source: https://scikit-learn.org/stable/modules/permutation_importance.html
- Permutation feature importance is a model inspection technique that can be used for any fitted estimator when the data is tabular.
- Features that are important on the training set but not on the held-out set might cause the model to overfit.
- Tree-based models provide an alternative measure of feature importances based on the mean decrease in impurity (MDI). Impurity is quantified by the splitting criterion of the decision trees (Gini, Entropy or Mean Squared Error). However, this method can give high importance to features that may not be predictive on unseen data when the model is overfitting. Permutation-based feature importance, on the other hand, avoids this issue, since it can be computed on unseen data.
- Furthermore, impurity-based feature importance for trees are strongly biased and favor high cardinality features (typically numerical features) over low cardinality features such as binary features or categorical variables with a small number of possible categories.
- Permutation-based feature importances do not exhibit such a bias. Additionally, the permutation feature importance may be computed performance metric on the model predictions and can be used to analyze any model class (not just tree-based models).
- When two features are correlated and one of the features is permuted, the model will still have access to the feature through its correlated feature. This will result in a lower importance value for both features, where they might actually be important.
- One way to handle this is to cluster features that are correlated and only keep one feature from each cluster.
- Source: https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance_multicollinear.html#sphx-glr-auto-examples-inspection-plot-permutation-importance-multicollinear-py
- Because this dataset contains multicollinear features, the permutation importance will show that none of the features are important. One approach to handling multicollinearity is by performing hierarchical clustering on the features’ Spearman rank-order correlations, picking a threshold, and keeping a single feature from each cluster.
- The permutation importance plot shows that permuting a feature drops the accuracy by at most 0.012, which would suggest that none of the features are important.
- When features are collinear, permutating one feature will have little effect on the models performance because it can get the same information from a correlated feature. One way to handle multicollinear features is by performing hierarchical clustering on the Spearman rank-order correlations, picking a threshold, and keeping a single feature from each cluster.
- Source: https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance.html#sphx-glr-auto-examples-inspection-plot-permutation-importance-py
- We will show that the impurity-based feature importance can inflate the importance of numerical features.
- Furthermore, the impurity-based feature importance of random forests suffers from being computed on statistics derived from the training dataset: the importances can be high even for features that are not predictive of the target variable, as long as the model has the capacity to use them to overfit.
- random_num is a high cardinality numerical variable (as many unique values as records).
random_cat is a low cardinality categorical variable (3 possible values).
The impurity-based feature importance ranks the numerical features to be the most important features. As a result, the non-predictive random_num variable is ranked the most important!
This problem stems from two limitations of impurity-based feature importances:
impurity-based importances are biased towards high cardinality features;
impurity-based importances are computed on training set statistics and therefore do not reflect the ability of feature to be useful to make predictions that generalize to the test set (when the model has enough capacity).
- It is also possible to compute the permutation importances on the training set. This reveals that random_num gets a significantly higher importance ranking than when computed on the test set. The difference between those two plots is a confirmation that the RF model has enough capacity to use that random numerical feature to overfit. You can further confirm this by re-running this example with constrained RF with min_samples_leaf=10.
- Source: https://christophm.github.io/interpretable-ml-book/feature-importance.html#feature-importance-data
- We measure the importance of a feature by calculating the increase in the model's prediction error after permuting the feature. A feature is "important" if shuffling its values increases the model error, because in this case the model relied on the feature for the prediction. A feature is "unimportant" if shuffling its values leaves the model error unchanged, because in this case the model ignored the feature for the prediction.
- If you want a more accurate estimate, you can estimate the error of permuting feature j by pairing each instance with the value of feature j of each other instance (except with itself). This gives you a dataset of size n(n-1) to estimate the permutation error, and it takes a large amount of computation time. I can only recommend using the n(n-1) method if you are serious about getting extremely accurate estimates.
- **The feature importance based on training data makes us mistakenly believe that features are important for the predictions, when in reality the model was just overfitting and the features were not important at all.**
- Feature importance based on the training data tells us which features are important for the model in the sense that it depends on them for making predictions.
- If you would use (nested) cross-validation for the feature importance estimation, you would have the problem that the feature importance is not calculated on the final model with all the data, but on models with subsets of the data that might behave differently.
-  **You need to decide whether you want to know how much the model relies on each feature for making predictions (-> training data) or how much the feature contributes to the performance of the model on unseen data (-> test data).**
## Drop-out Feature Importance
- Source: https://towardsdatascience.com/explaining-feature-importance-by-example-of-a-random-forest-d9166011959e
## On Train Set or Test Set?
- Source: https://christophm.github.io/interpretable-ml-book/feature-importance.html
### On Test Set
- Really, it is one of the first things you learn in machine learning: If you measure the model error (or performance) on the same data on which the model was trained, the measurement is usually too optimistic, which means that the model seems to work much better than it does in reality. And since the permutation feature importance relies on measurements of the model error, we should use unseen test data. The feature importance based on training data makes us mistakenly believe that features are important for the predictions, when in reality the model was just overfitting and the features were not important at all.

# Feature Engineering
- Source: https://adataanalyst.com/machine-learning/comprehensive-guide-feature-engineering/
## Time-Related Variables
- Time-stamp attributes are usually denoted by the EPOCH time or split up into multiple dimensions such as (Year, Month, Date, Hours, Minutes, Seconds). But in many applications, a lot of that information is unnecessary. Consider for example a supervised system that tries to predict traffic levels in a city as a function of Location+Time. In this case, trying to learn trends that vary by seconds would mostly be misleading. The year wouldn’t add much value to the model as well. Hours, day and month are probably the only dimensions you need. So when representing the time, try to ensure that your model does require all the numbers you are providing it.
- Here is an example hypothesis: An applicant who takes days to fill in an application form is likely to be less interested / motivated in the product compared to some one who fills in the same application with in 30 minutes. Similarly, for a bank, time elapsed between dispatch of login details for Online portal and customer logging in might show customers’ willingness to use Online portal. Another example is that a customer living closer to a bank branch is more likely to have a higher engagement than a customer living far off.
## Creating New Ratios and Proportions
- For example, in order to predict future performance of credit card sales of a branch, ratios like credit card sales / Sales person or Credit card Sales / Marketing spend would be more powerful than just using absolute number of card sold in the branch
## Creating Weights
- There may be domain knowledge that items with a weight above 4 incur a higher taxation rate. That magic domain number could be used to create a new binary feature Item_Above_4kg with a value of “1” for our example of 6289 grams.
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
- Source: https://www.tandfonline.com/doi/full/10.1080/10095020.2018.1503775
- The Euclidean function is unrealistic for some (notably urban) settings which contain complex physical restrictions and social structures for example road and path networks, large restricted areas of private land and legal road restrictions such as speed limits and one-way systems.
## House prices in space
- Most contemporary analysis mimics this trend, for example predicting property value by using (1) the average sales price of other properties in the local comparable markets, (2) a spatial clustering of properties and demographics (Malczewski 2004) and (3) a local demographic “trade area” (Daniel 1994).
- In the case of spatially dependent data, cross-validation is optimistic due to its inherent IID assumption.
- Euclidean distances are exclusively considered in all of the above work. This paper hypothesizes that house prices are related to a more complex structural network relating to (restricted) road distance and travel time; hence, we introduce an approximate (restricted) road distance and travel time metric using the Minkowski distance function for a valid house price Kriging predictor (Matheron 1963; Cressie 1990).
## Data Description
- Figure 1. A comparison of an Euclidean distance matrix versus a drive time distance matrix and a road distance matrix around the center point of Coventry. (a) Euclidean distance buffer from 0 to 4 miles around the centre of Coventry; (b) Travel time distance buffer from 0 to 10 minutes drive time around the centre of Coventry; (c) Road distance buffer from 0 to 4 miles around the centre of Coventry.
## Collapsing Time
- The price paid data for 2016 are addressed only (herewithin named ). This accounts for 3669 sales in Coventry. Stage 1 predicts each property’s sale price based on its value on the 1 January 2017 (for time singularity). This process involves each property being assigned some percentage price change based on the date that it was sold and the lower super output area that the property is contained within to produce a value for all 3669 properties at the date 1 January 2017 (). The error for the purposes of this experiment is minimal or nonexistent due to the small temporal and spatial aggregate areas being considered.
- Figure 2 shows an exact example where the distance between houses  to  is 0.24 mi along the red dotted line which takes a route along “Brownshill Green Road” and is marked as a one-way system, this means that the route  to  must be different, which, in this case, is further; hence, the distance matrix is not symmetric. The same reasoning applies for a travel time matrix

# Time Feature
- Source https://medium.com/@andrejusb/machine-learning-date-feature-transformation-explained-4feb774c9dbe
## Splitting Features
- One of the ways is to split date value into multiple columns with numbers describing the original date (year, quarter, month, week, day of year, day of month, day of week).
## Date Feature Transformation into a Difference Between Dates
- We can use date difference as such:
— Day difference between Payment Due Date and Invoice Date
— Day difference between Payment Date and Invoice Date
This should bring clear pattern when there is payment delay — difference between payment date/invoice date will be bigger than between payment due date/invoice date.

# Explicit Data, Implicit Data
- source: https://orill.tistory.com/entry/Explicit-vs-Implicit-Feedback-Datasets
## Explicit Datasets
- 영화 추천시스템을 만드는 상황을 생각해보겠습니다. 어떤 데이터를 사용할 수 있을까요?
- 가장 먼저 떠오르는 데이터는 사용자의 평점 기록 데이터입니다. 이미지 분류에 MNIST가 있다면 추천시스템에는 Movielens Data가 있습니다. ratings.csv 파일은 (userId,movieId,rating,timestamp) 형태의 데이터를 저장하고 있습니다. 유저가 본 영화에 대해서 5.0점을 만점으로 0.5점 단위로 평가한 데이터가 시간과 함께 저장되어 있습니다. Watcha는 이런 종류의 데이터를 수집하여 추천 앱을 서비스하고 있고 Neflix는 평점대신 좋아요, 싫어요 데이터를 수집하고 있습니다. 평점 기록처럼 유저가 자신의 선호도를 직접(Explicit) 표현한 Data를 Explicit Data라고 합니다. 다른 종류의 Explicit Data로는 영화 리뷰, 구독, 차단 데이터 등이 있습니다.
- Explicit Data를 통해서 얻을 수 있는 정보는 강력합니다. 유저의 호불호를 명백히 알 수 있기 때문입니다. 유용성이 좋은 반면 데이터를 얻기 힘들다는 단점이 있습니다. 데이터 분석을 해보면 영화를 보고 평점을 매기는 유저의 숫자는 전체 유저에 비해 적고 리뷰까지 남기는 데이터는 훨씬 적습니다. 유저가 적극적인 Action을 취해야 하는데 단순히 귀찮거나 심리적인 거부감이 있기 때문이죠. "좋아요와 구독 부탁드려요"라는 말은 유저들이 웬만해서는 '좋아요'를 눌러주지 않는다는 점을 방증하고 있는 듯합니다. 
## Implicit Datasets
- Collaborative Filtering For Implicit Feedback Datasets 논문에 따르면 2010년 이전까지는 추천시스템 분야에서 Explicit Data를 활용하는 방법에 대한 연구가 주로 이루어졌던 모양입니다. 논문에서는 Implicit라는 다른 종류의 데이터를 활용하여 추천시스템을 만드는 방법을 제시하고 있습니다. 아래부터는 이 논문에서 소개한 Implicit Datasets의 개념과 특징을 정리한 내용입니다. Model 부분에 추후에 포스팅할 예정입니다.
- Implicit Data는 유저가 간접적(Implicit)으로 선호, 취향을 나타내는 데이터를 의미합니다. 예시로는 검색 기록, 방문 페이지, 구매 내역 심지어 마우스 움직임 기록이 있습니다. 이런 데이터는 일단 유저가 개인정보제공에 동의만 한다면 자동적으로 수집되기 때문에 수집의 난이도 낮고 활용할 수 있는 데이터량이 많습니다.
- Implicit Datasets을 다룰 때 염두해두어야 할 몇 가지 특징이 있습니다.
	- 부정적인 피드백이 없다(No Negative Feedback) : 유저가 어떤 영화를 본 기록이 없을 때 유저가 그 영화를 싫어해서 보지 않았는지 혹은 그저 알지 못했기 때문에 보지 않았는지 알 수 없습니다. 이런 차이점 때문에 Explicit Feedback Dataset을 다룰 때는 수집된 데이터에만 집중하고 Unobserved Data는 Missing Value 취급하여 모델을 만들어도 괜찮습니다. 유저의 불호 정보(낮은 평점, 싫어요)가 포함되어 있기 때문입니다. 하지만 Implicit Data를 모델링할 때는 수집되지 않은 데이터도 같이 모델링해야 합니다. 수집되지 않은 데이터에 (확실하지는 않아도) 불호 정보, 부정적인 정보가 담겨 있을 가능성이 크기 때문입니다.
	- 애초에 잡음이 많다(Inherently Noisy) :  반대로 어떤 영화를 봤다고 해서 유저가 그 영화를 좋아한다고 말하기 힘듭니다. 과제 때문에 영화를 본 것일 수도 있고 영화가 마음에 안들지만 구매한 게 아쉬워서 끝까지 본 것일 수도 있기 때문입니다. 유튜브의 경우 시청시간이 중요하다는 말이 있는데 유저가 영상을 틀어놓고 잠들었을 수도 있습니다.
	- 수치는 신뢰도를 의미한다.(The numerical value of implicit feedback indicates confidence) : Explicit Data의 경우 높은 수치는 높은 선호도를 의미합니다. 2번에서 봤듯이 Implicit Data에서는 높은 수치가 꼭 높은 선호도를 의미하는 것은 아닙니다. 인생 영화를 봤어도 시청시간은 2시간 즈음인 반면 그저 그런 드라마를 보는 경우에 10시간을 넘게 볼 수도 있습니다. 그럼에도 높은 값은 신뢰할만한 데이터임을 의미한다고 해석할 수 있습니다. 한 번 보고만 영상보다는 자주, 오래 본 영상이 유저의 선호도, 의견을 표현했을 확률이 높기 때문입니다.
	- Implicit-feedback Recommender System의 평가는 적절한 방법을 고민해봐야 한다 : 평점 데이터를 이용하는 경우 예측값과 실제값이 얼마나 다른지를 평가하는 Mean Squared Error 방법을 사용하면 편리합니다. 하지만 시청시간, 클릭수, 조회 기록을 이용하는 경우 정답값을 주기가 어렵습니다. 따라서 implicit model의 경우 item의 availability나 반복되는 feeback 등을 고려해야 합니다. availability란 동시간에 방영되는 두 TV Show의 경우 한쪽만 볼 수 있어서 다른 프로그램을 좋아한다고 해도 Implicit Data가 쌓이지 않는 상황을 말합니다. 반복되는 Feedback은 유저가 한 번 이상 프로그램을 봤을 때 한 번 본 경우와 어떻게 다르게 평가할 것인가에 대한 고려입니다.
## Collaborative Filtering For Implicit Feedback Datasets
- 다음 포스팅에서는 앞서 언급한 Implicit Datasets의 특징을 고려해서 어떻게 추천 모델을 만들고 평가할지에 대한 내용을 정리해보겠습니다. 짧게 정리하자면 Unobserved Data와 Observed Data를 구분하고 높은 Confidence Data에 높은 Loss를 의미하는 Loss Function을 정의하여 Matrix Factorization을 수행합니다. Evaluation은 Recall 지표를 사용하는 것이 타당하다고 생각하여 Expected Percentile Ranking을 사용합니다.

# Model Linearity
## Difference between Linear and Nonlinear
- Source: https://statisticsbyjim.com/regression/difference-between-linear-nonlinear-regression-models/
- The form is linear in the parameters because all terms are either the constant or a parameter multiplied by an independent variable (IV). A linear regression equation simply sums the terms. While the model must be linear in the parameters, you can raise an independent variable by an exponent to fit a curve. For instance, you can include a squared or cubed term.
Nonlinear regression models are anything that doesn’t follow this one form.
While both types of models can fit curvature, nonlinear regression is much more flexible in the shapes of the curves that it can fit. After all, the sky is the limit when it comes to the possible forms of nonlinear models.
- While the independent variable is squared, the model is still linear in the parameters. Linear models can also contain log terms and inverse terms to follow different kinds of curves and yet continue to be linear in the parameters.
- If a regression equation doesn’t follow the rules for a linear model, then it must be a nonlinear model
- Source: https://brunch.co.kr/@gimmesilver/18
- 비선형 모델은 데이터를 어떻게 변형하더라도 파라미터를 선형 결합식으로 표현할 수 없는 모델을 말합니다. 이런 비선형 모델 중 단순한 예로는 아래와 같은 것이 있습니다. 이 식은 아무리 x, y 변수를 변환하더라도 파라미터를 선형식으로 표현할 수 없습니다.
    선형 회귀 모델은 파라미터 계수에 대한 해석이 단순하지만 비선형 모델은 모델의 형태가 복잡할 경우 해석이 매우 어렵습니다. 그래서 보통 모델의 해석을 중시하는 통계 모델링에서는 비선형 회귀 모델을  잘 사용하지 않습니다. 
    그런데 만약 회귀 모델의 목적이 해석이 아니라 예측에 있다면 비선형 모델은 대단히 유연하기 때문에 복잡한 패턴을 갖는 데이터에 대해서도 모델링이 가능합니다. 그래서 충분히 많은 데이터를 갖고 있어서 variance error를 충분히 줄일 수 있고 예측 자체가 목적인 경우라면 비선형 모델은 사용할만한 도구입니다. 기계 학습 분야에서는 실제 이런 비선형 모델을 대단히 많이 사용하고 있는데 가장 대표적인 것이 소위 딥 러닝이라고 부르는 뉴럴 네트워크입니다.
- 정리하자면, 선형 회귀 모델은 파라미터가 선형식으로 표현되는 회귀 모델을 의미합니다. 그리고 이런 선형 회귀 모델은 파라미터를 추정하거나 모델을 해석하기가 비선형 모델에 비해 비교적 쉽기 때문에, 데이터를 적절히 변환하거나 도움이 되는 feature들을 추가하여 선형 모델을 만들 수 있다면 이렇게 하는 것이 적은 개수의 feature로 복잡한 비선형 모델을 만드는 것보다 여러 면에서 유리합니다. 반면 선형 모델은 표현 가능한 모델의 가짓수(파라미터의 개수가 아니라 파라미터의 결합 형태)가 한정되어 있기 때문에 유연성이 떨어집니다. 따라서 복잡한 패턴을 갖고 있는 데이터에 대해서는 정확한 모델링이 불가능한 경우가 있습니다. 그래서 최근에는 모델의 해석보다는 정교한 예측이 중요한 분야의 경우 뉴럴 네트워크와 같은 비선형 모델이 널리 사용되고 있습니다.

# Missing Value
## Treating Missing Values
- Source: https://adataanalyst.com/machine-learning/comprehensive-guide-feature-engineering/
### Imputation(Mean, Mode, Median)
- Generalized Imputation: In this case, we calculate the mean or median for all non missing values of that variable then replace missing value with mean or median. Like in above table, variable “Manpower” is missing so we take average of all non missing values of “Manpower” (28.33) and then replace missing value with it.
Similar case Imputation: In this case, we calculate average for gender “Male” (29.75) and “Female” (25) individually of non missing values then replace the missing value based on gender. For “Male“, we will replace missing values of manpower with 29.75 and for “Female” with 25.
### Prediction
- In this case, we divide our data set into two sets: One set with no missing values for the variable and another one with missing values. First data set become training data set of the model while second data set with missing values is test data set and variable with missing values is treated as target variable.
### Interpolation
### K-Nearest Neighbors Imputation
- KNN algorithm is very time-consuming in analyzing large database. It searches through all the dataset looking for the most similar instances.
- Choice of k-value is very critical. Higher value of k would include attributes which are significantly different from what we need whereas lower value of k implies missing out of significant attributes. 

# Outliers
## Types of Outliers
- Source: https://adataanalyst.com/machine-learning/comprehensive-guide-feature-engineering/
- Univariate outliers can be found when we look at distribution of a single variable.
Multi-variate outliers are outliers in an n-dimensional space. In order to find them, you have to look at distributions in multi-dimensions.
### Data Entry Errors
- Human errors such as errors caused during data collection, recording, or entry can cause outliers in data.
### Data Processing Errors
- It is possible that some manipulation or extraction errors may lead to outliers in the dataset.
### Measurement Errors
- It is the most common source of outliers. This is caused when the measurement instrument used turns out to be faulty. 
E## Experimental Errors
- For example: In a 100m sprint of 7 runners, one runner missed out on concentrating on the ‘Go’ call which caused him to start late. Hence, this caused the runner’s run time to be more than other runners.
### Intentional Outliers
- For example: Teens would typically under report the amount of alcohol that they consume. Only a fraction of them would report actual value. Here actual values might look like outliers because rest of the teens are under reporting the consumption.
### Sampling Errors
- For instance, we have to measure the height of athletes. By mistake, we include a few basketball players in the sample.
### Natural Outliers
## Outliers Detection
- Source: https://adataanalyst.com/machine-learning/comprehensive-guide-feature-engineering/
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
- Source: https://cyan91.tistory.com/40, https://lsjsj92.tistory.com/556?category=853217
### Z-score
- Source: https://cyan91.tistory.com/40, https://soo-jjeong.tistory.com/121

# Sparse Respresentation, Dense Representation
- Source: https://dreamgonfly.github.io/blog/word2vec-explained/

# Anomaly

- 단순한 문제는 단순한 모델로 풀어야 함
- 시간의 흐름에 따라 관측치가 변하는 데이터

# Autoencoder
- Source: https://en.wikipedia.org/wiki/Autoencoder

# `graphviz`
```python
import graphviz
```
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

## `sklearn.feature_extraction.text`
### `CountVectorizer()`
```python
from sklearn.feature_extraction.text import CountVectorizer
```
```python
vect = CountVectorizer(max_df=500, min_df=5, max_features=500)
```
- Ignore if frequency of the token is greater than `max_df` or lower than `min_df`.
#### `vect.fit()`
#### `vect.transform()`
- Build document term maxtrix.
##### `vect.transform().toarray()`
#### `vect.fit_transform()`
- `vect.fit()` + `vect.transform()`
##### `vect.fit_transform().toarray()`
#### `vect.vocabulary_`
##### `vect.vocabulary_.get()`
- Return the index of the argument.
### `TfidfVectorizer()`
```python
from sklearn.feature_extraction.text import TfidfVectorizer
```
## `sklearn.preprocessing`
### `LabelEncoder()`
#### `le.fit()`, `le.transform()`, `le.fit_transform()`, `le.inverse_transform()`
#### `le.classes_`
```python
label2idx = dict(zip(le.classes_, set(label_train)))
```
### `StandardScaler()`, `MinMaxScaler()`, `RobustScaler()`, `Normalizer()`
```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer
```
```python
sc = StandardScaler()
```
#### `sc.fit()`, `sc.transform()`, `sc.fit_transform()`
## `sklearn.decomposition`
### `PCA()`
```python
from sklearn.decomposition import PCA
```
```python
pca = PCA(n_components=2)
```
```python
pca_mat = pca.fit_transform(user_emb_df)
```
## `sklearn.pipeline`
### `Pipeline()`
```python
from sklearn.pipeline import Pipeline
```
```python
model = Pipeline([("vect", CountVectorizer()), ("model", SVC(kernel="poly", degree=8))])
```
- 파이프라인으로 결합된 모형은 원래의 모형이 가지는 fit, predict 메서드를 가지며 각 메서드가 호출되면 그에 따른 적절한 메서드를 파이프라인의 각 객체에 대해서 호출한다. 예를 들어 파이프라인에 대해 fit 메서드를 호출하면 전처리 객체에는 fit_transform이 내부적으로 호출되고 분류 모형에서는 fit 메서드가 호출된다. 파이프라인에 대해 predict 메서드를 호출하면 전처리 객체에는 transform이 내부적으로 호출되고 분류 모형에서는 predict 메서드가 호출된다.
## `sklearn.svm`
### `SVC()`, `SVR()`
```python
from sklearn.svm import SVC
```
```python
SVC(kernel="linear")
```
- `kernel="linear"`
- `kernel="poly"`: gamma, coef0, degree
- `kernel="rbf"`: gamma
- `kernel="sigmoid"`: gomma, coef0
## `sklearn.naive_bayes`
```python
from sklearn.naive_bayes import MultinomialNB
```
## `sklearn.linear_model`
```python
from sklearn.linear_model import SGDClassifier
```
### `Ridge()`, `Lasso()`, `ElasticNet()`
```python
fit = Ridge(alpha=alpha, fit_intercept=True, normalize=True, random_state=123).fit(x, y)
```
#### `fit.intercept_`, `fit.coef_`
### `SGDClassifier`
```python
model = SGDClassifier(loss="perceptron", penalty="l2", alpha=1e-4, random_state=42, max_iter=100)
...
model.fit(train_x, train_y)
train_pred = model.pred(train_x)
train_acc = np.mean(train_pred == train_y)
```
- `loss`: The loss function to be used.
    - `loss="hinge"`: Give a linear SVM.
    - `loss="log"`: Give logistic regression.
    - `loss="perceptron"`: The linear loss used by the perceptron algorithm.
- `penalty`: Regularization term.
    - `penalty="l1"`
    - `penalty="l2"`: The standard regularizer for linear SVM models.
- `alpha`: Constant that multiplies the regularization term. The higher the value, the stronger the regularization. Also used to compute the learning rate when `learning_rate` is set to `"optimal"`.
- max_iter`: The maximum number of passes over the training data (aka epochs).
## `sklearn.ensemble`
### `RandomForestRegressor()`, `GradientBoostingRegressor()`, `AdaBoostRegressor()`
```python
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
```
## `sklearn.tree`
### `DecisionTreeRegressor()`
```python
from sklearn.tree import DecisionTreeRegressor
```
## `sklearn.datasets`
### `sklearn.datasets.fetch_20newsgroups()`
```python
newsdata = sklearn.datasets.fetch_20newsgroups(subset="train")
```
- `subset`: (`"all"`, `"train"`, `"test"`)
### `sklearn.datasets.sample_generator`
#### `make_blobs()`
```python
 from sklearn.datasets.sample_generator improt make_blobs
```
## `sklearn.metrcis`
### `sklearn.metrics.pairwise`
#### `sklearn.metrics.pairwise.cosine_similarity`
### `sklearn.metrics.classification_report()`
```python
print(sklearn.metrics.classification_report(y_pred, y_test))
```

# `tensorflow`
```python
import tensorflow as tf
from tensorflow.keras import Input, Model, Sequential
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from tensorflow.keras.layers import Layer, Dense, Flatten, Dropout, Concatenate, Add, Dot, Multiply, Reshape, Activation, BatchNormalization, SimpleRNNCell, RNN, SimpleRNN, LSTM, Embedding, Bidirectional, TimeDistributed, Conv1D, Conv2D, MaxPool1D, MaxPool2D, GlobalMaxPool1D, GlobalMaxPool2D, AveragePooling1D, AveragePooling2D, GlobalAveragePooling1D, GlobalAveragePooling2D, ZeroPadding2D
from tensorflow.keras.optimizers import SGD, Adam, Adagrad
from tensorflow.keras.metrics import RootMeanSquaredError, BinaryCrossentropy, SparseCategoricalAccuracy
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.activations import linear, sigmoid, relu
```
## Create Tensors
### `tf.Variable(initial_value, [shape=None], [trainable=True], [validate_shape=True], [dtype], [name])`
- Source: https://www.tensorflow.org/api_docs/python/tf/Variable
- `initial_value`: This initial value defines the type and shape of the variable. After construction, the type and shape of the variable are fixed.
- [`shape`]: The shape of this variable. If `None`, the shape of `initial_value` will be used.
- `validate_shape`: If `False`, allows the variable to be initialized with a value of unknown shape. If `True`, the default, the shape of `initial_value` must be known.
- [`dtype`]: If set, `initial_value` will be converted to the given type. If `None`, either the datatype will be kept (if `initial_value` is a Tensor), or `convert_to_tensor()` will decide.
### `tf.zeros()`
```python
W = tf.Variable(tf.zeros([2, 1], dtype=tf.float32), name="weight")
```
## Layers
## Computation between Tensors
### `tf.stack(values, axis, [name])`
- Source: https://www.tensorflow.org/api_docs/python/tf/stack
- Stacks a list of tensors of rank R into one tensor of rank (R + 1).
- `axis`: The axis to stack along.
- Same syntax as `np.stack()`
### `Add()`
```python
logits = Add()([logits_mlr, logits_fm, logits_dfm])
```
- It takes as input a list of tensors, all of the same shape, and returns a single tensor (also of the same shape).
### `Dot(axes)`
```python
pos_score = Dot(axes=(1, 1))([z1, z2])
```
- `axes` : (integer, tuple of integers) Axis or axes along which to take the dot product. If a tuple, should be two integers corresponding to the desired axis from the first input and the desired axis from the second input, respectively. Note that the size of the two selected axes must match.
### `Multiply()`
```python
def se_block(x, c, r):
	z = GlobalAveragePooling2D()(x)
	z = Dense(units=c//r, activation="relu")(z)
	z = Dense(units=c, activation="sigmoid")(z)
	z = Reshape(target_shape=(1, 1, c))(z)
	z = Multiply()([x, z])
	return z
```
## `Reshape()`
```python
z = Reshape(target_shape=(1, 1, ch))(z)
```
### `Concatenate()`
```python
Concatenate(axis=1)(embs_fm)
```
- tf.concat()와 동일합니다.
## `Activation()`
```python
x = Activation("relu")(x)
```
## Transforms Tensor Shape
### `Flatten()`
- 입력되는 tensor의 row를 펼쳐서 일렬로 만듭니다.
- 학습되는 weights는 없고 데이터를 변환하기만 합니다.
```python
model.add(Flatten(input_shape=(28, 28)))
```
## `Input(shape, [name], [dtype], ...)`
- `shape`
	- ***A shape tuple (integers), not including the batch size***. For instance, shape=(32,) indicates that the expected input will be batches of 32-dimensional vectors.
	- ***Elements of this tuple can be None; "None" elements represent dimensions where the shape is not known.***
## `Embedding(input_dim, output_dim, [mask_zero], [input_length], [name], ...)`
- Source: https://www.tensorflow.org/api_docs/python/tf/keras/layers/Embedding
- `input_dim`: Size of the vocabulary.
- `output_dim`: Dimension of the dense embedding.
- `input_length`: Length of input sequences, when it is constant. This argument is required if you are going to connect Flatten then Dense layers upstream.
- `mask_zero=True`: Whether or not the input value 0 is a special "padding" value that should be masked out. This is useful when using recurrent layers which may take variable length input. If `mask_zero` is set to `True`, as a consequence, index 0 cannot be used in the vocabulary (`input_dim` should equal to (Size of vocabulary + 1)).
- Input shape: `(batch_size, input_length)`
- Output shape: `(batch_size, input_length, output_dim)`
## `Dense()`
```python
Dense(units=52, input_shape=(13,), activation="relu")
```
- units: 해당 은닉층에서 활동하는 뉴런의 수(출력 값의 크기)
- activation: 활성화함수, 해당 은닉층의 가중치와 편향의 연산 결과를 어느 함수에 적합하여 출력할 것인가?
- input_shape : 입력 벡터의 크기. 여기서 13은 해당 데이터 프레임의 열의 수를 나타낸다. 데이터의 구조(이미지, 영상)에 따라 달라질 수 있다. 첫 번째 은닉층에서만 정의해준다.
## `Dropout()`
- rate : dropout을 적용할 perceptron의 비율
## `BatchNormalization()`
- usually used before activation function layers.
## `Conv1D()`
```python
Conv1D(filters=n_kernels, kernel_size=kernel_size, padding="same", activation="relu", strides=1)
```
- `strides` : basically equals to 1
## `Conv2D()`
```python
conv2d = Conv2D(filters=n_filters, kernel_size=kernel_size, strides=(1, 1), padding="same")(image)
```
- `image`: (batch, height of image, width of image, number of channels)
- `kernel`: (height of filter, width of filter, number of channels, number of kernels)
- `convolution`: (batch, height of convolution, width of convolution, number of kernels)
- number of channels와 number of kernels는 서로 동일합니다.
- `kernal_size`: window_size
- `padding="valid"`: No padding. There can be a loss of information. The size of the output image is smaller than the size of the input image.
- `padding="same"`: Normally, padding is set to same while training the model.
- `data_format`: (`"channels_last"`)
- `input_shape`: 처음에만 설정해 주면 됩니다.
- `activation`: (`"tanh"`)
## Pooling Layers
### `MaxPool1D()`, `MaxPooling1D()`
- `strides` : basically equals to 2
### `MaxPool2D()`, `MaxPooling2D()`
```python
pool = MaxPool2D(pool_size=(2, 2), strides=1, padding="valid", data_format="channels_last")(image)
```
### `GlobalMaxPool1D()`, `GlobalMaxPooling1D()`
- Shape changes from (a, b, c, d) to (a, d).
### `GlobalMaxPool2D()`, `GlobalMaxPooling2D()`
- Downsamples the input representation by taking the maximum value over the time dimension.
- Shape changes from (a, b, c) to (b, c).
### `AveragePooling1D()`
### `AveragePooling2D([pool_size], [strides], [padding])`
### `GlobalAveragePooling1D()`
### `GlobalAveragePooling2D()`
### `ZeroPadding2D`
```python
z = ZeroPadding2D(padding=((1, 0), (1, 0)))(x)
```
- `padding`:
	- Int: the same symmetric padding is applied to height and width.
	- Tuple of 2 ints: interpreted as two different symmetric padding values for height and width: `(symmetric_height_pad, symmetric_width_pad)`.
	- Tuple of 2 tuples of 2 ints: interpreted as `((top_pad, bottom_pad), (left_pad, right_pad))`.
## `SimpleRNNCell()`
## `RNN()`
## `SimpleRNN()`, `GRU()`
```python
outputs, hidden_states = SimpleRNN(units=hidden_size)(x_data), input_shape=(timesteps, input_dim), return_sequences=True, return_state=True)(x_date)
```
- `SimpleRNN()` = `SimpleRNNCell()` + `RNN()`
- `batch_input_shape=(batch_size, timesteps, input_dim)`
- `return_sequences=False` : (default)time step의 마지막에서만 아웃풋을 출력합니다.(shape of output : (batch_size, hidden_size))
- `return_sequences=True` : 모든 time step에서 아웃풋을 출력합니다. many to many 문제를 풀거나 LSTM 레이어를 여러개로 쌓아올릴 때는 이 옵션을 사용합니다.(shape of output : (batch_size, timesteps, hidden_size))
- `return_state=True` : hidden state를 출력합니다.(shape of hidden state : (batch_size, hidden_size))
```python
GRU(units=hidden_size, input_shape=(timesteps, input_dim))
```
## `LSTM()`
```python
_, hidden_state, cell_state = LSTM(units=256, return_state=True)(inputs_enc)
```
- `tf.keras.layers.SimpleRNN()`과 문법이 동일합니다.
- `return_state=True` : hidden state와 cell state를 출력합니다.
## `Bidirectional()`
```python
Bidirectional(tf.keras.layers.SimpleRNN(hidden_size, return_sequences=True), input_shape=(timesteps, input_dim))
```
## `TimeDistributed()`
```python
model.add(TimeDistributed(tf.keras.layers.Dropout(rate=0.2)))
```
- TimeDistributed를 이용하면 각 time에서 출력된 아웃풋을 내부에 선언해준 레이어와 연결시켜주는 역할을 합니다.
- In keras - while building a sequential model - usually the second dimension (one after sample dimension) - is related to a time dimension. This means that if for example, your data is 5-dim with (sample, time, width, length, channel) you could apply a convolutional layer using TimeDistributed (which is applicable to 4-dim with (sample, width, length, channel)) along a time dimension (applying the same layer to each time slice) in order to obtain 5-d output.
#### `tf.keras.layers.Layer`
- custom layer를 만들려면 `tf.keras.layers.Layer` 클래스를 상속하고 다음 메서드를 구현합니다
    - __init__: 이 층에서 사용되는 하위 층을 정의할 수 있습니다. instance 생성 시에 호출됩니다.
    - build: 층의 가중치를 만듭니다. add_weight 메서드를 사용해 가중치를 추가합니다.
    - call: forward feeding 단계에서 호출됩니다. 입력 값을 이용해서 결과를 계산한 후 반환하면 됩니다.

## `tf.identity()`
## `tf.constant()`
```python
image = tf.constant([[[[1], [2], [3]], [[4], [5], [6]], [[7], [8], [9]]]], dtype=tf.float32)
```
## `tf.convert_to_tensor()`
```python
img = tf.convert_to_tensor(img)
```
## `tf.transpose()`
## `tf.cast()`
```python
pred = tf.cast(h > 0.5, dtype=tf.float32)
```
- 조건이 True면 1, False면 0 반환.
- 혹은 단순히 Tensor의 자료형 변환.
## `tf.concat()`(= `tf.keras.layers.Concatenate()`)
```python
layer3 = tf.concat([layer1, layer2], axis=1)
```
- 지정한 axis의 dimension이 유지됩니다.
## `tf.shape()`
```python
batch_size = tf.shape(conv_output)[0]
```
## `tf.reshape()`
```python
conv_output = tf.reshape(conv_output, shape=(batch_size, output_size, output_size, 3,
                                                 5 + n_clss))
```
## `tf.range()`
```python
tf.range(3, 18, 3)
```
## `tf.tile()`
```python
y = tf.tile(y, multiples=[1, output_size])
```
## `tf.constant_initializer()`
```
weight_init = tf.constant_initializer(weight)
```
## `tf.GradientTape()`
```python
with tf.GradientTape() as tape:
    hyp = W * X + b
    loss = tf.reduce_mean(tf.square(hyp - y))

dW, db = tape.gradient(loss, [W, b])
```
## `tf.math`
### `tf.math.add()`, `tf.math.subtract()`, `tf.math.multiply()`, `tf.math.divide()`
- Adds, substract, multiply or divide two input tensors element-wise.
### `tf.math.add_n(inputs)`
- Adds all input tensors element-wise.
- `inputs`: A list of Tensors, each with the same shape and type.
### `tf.math.square()`
- Compute square of x element-wise.
### `tf.math.argmax()`
```python
y_pred = tf.math.argmax(model.predict(X_test), axis=1)
```
### `tf.math.sign`
```python
tf.math.sign(tf.math.reduce_sum(self.w * x) + self.b)
```
### `tf.math.exp()`
### `tf.math.log()`
### `tf.math.equal()`
```python
acc = tf.math.reduce_mean(tf.cast(tf.math.equal(pred, labels), dtype=tf.float32))
```
### `tf.math.reduce_sum()`, `tf.math.reduce_mean()`
- Source: https://www.tensorflow.org/api_docs/python/tf/math/reduce_sum#returns_1
- `axis=None`: 모든 elements에 대해 연산합니다.
- `axis=0`: reduces along the 1st dimension. dimension이 1만큼 감소합니다.
- `axis=1`: reduces along the 2nd dimension. dimension이 1만큼 감소합니다.
- `keepdims=True`: dimension이 감소하지 않습니다.
## `tf.nn`
### `tf.nn.softmax()`
```python
h = tf.nn.softmax(tf.matmul(train_X, W) + b)
```
### `tf.nn.relu`
## `tf.data`
### `tf.data.Dataset`
#### `tf.data.Dataset.from_tensor_slices()`
```python
train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y)).shuffle(len(train_x)).batch(batch_size, drop_remainder=True).prefetch(batch_size)
test_dataset = tf.data.Dataset.from_tensor_slices((test_x, test_y)).shuffle(len(test_x)).batch(len(test_x)).prefetch(len(test_x))
```
##### `tf.data.Dataset.from_tensor_slices().shuffle()`
- 지정한 개수의 데이터를 무작위로 섞어서 출력합니다.
##### `tf.data.Dataset.from_tensor_slices().batch()`
- 지정한 개수의 데이터를 묶어서 출력합니다.
##### `tf.data.Dataset.from_tensor_slices().prefetch()`
- This allows later elements to be prepared while the current element is being processed. This often improves latency and throughput, at the cost of using additional memory to store prefetched elements.
## `tf.train`
### `tf.train.Checkpoint()`
## `tf.keras`
### `Sequential()`
```python
model = Sequential()
```

## Build Model
```python
model = Model(inputs, ouputs, [name])
```
## Compile
```python
# `optimizer`: (`"sgd"`, `"adam"`, `"rmsprop"`)
# `loss`: (`"mse"`, `"binary_crossentropy"`, `"categorical_crossentropy"`, `"sparse_categorical_crossentropy"`)
# `metrics`: (`["mse"]`, `["binary_accuracy"]`, `["categorical_accuracy"]`, `["sparse_categorical_crossentropy"]`, `["acc"]`)
# `loss_weights`
model.compile(optimizer, loss, metrics, [loss_weights])

model.summary()
```
## Fit
- Source: https://keras.io/api/models/model_training_apis/
```python
# `mode`: (`"auto"`, `"min"`, `"max"`).
	# `"min"`: Training will stop when the quantity monitored has stopped decreasing;
	# `"max"`: It will stop when the quantity monitored has stopped increasing;
	# `"auto"`: The direction is automatically inferred from the name of the monitored quantity.
# `patience`: Number of epochs with no improvement after which training will be stopped.
es = EarlyStopping(monitor="val_loss", mode="auto", verbose, patience)
model_path = "model_path.h5"
# `verbose=1`: 모델이 저장 될 때 '저장되었습니다' 라고 화면에 표시됩니다.
# `save_best_only=True`: `monitor` 기준으로 가장 좋은 값으로 모델이 저장됩니다.
# `save_best_only=False`: 매 epoch마다 모델이 filepath{epoch}으로 저장됩니다.
# `save_weights_only=True`: 모델의 weights만 저장됩니다.
# `save_weights_only=False`: 모델 레이어 및 weights 모두 저장됩니다.
mc = ModelCheckpoint(filepath=model_path, monitor="val_acc", mode, verbose=1, [save_best_only])
# `x`
# `y`
# `validation_split`
# `verbose=2`: One line per epoch. recommended.
hist = model.fit(x=X, y=y, validation_split=0.2, batch_size, epochs, verbose=2, shuffle=True, callbacks=[es, mc])
```
```python
hist = model.fit(train_ds, validation_data=val_ds, epochs=epochs)
```
```python
hist = model.fit_generator(generator=train_set.shuffle(len(x_train)).batch(batch_size), epochs=n_epochs, validation_data=val_set.batch(batch_size))
```
### `hist.history`
```python
hist.history["accuracy"]
```
- (`"accuracy"`, `"val_accuracy"`, `"loss"`, `"val_loss"`)
## Evaluate Model
```python
score = model.evaluate(x_test, y_test, batch_size=128, verbose=0)
```
## Predict
```python
preds = model.predict(x.values)
```
## 가중치 확인
```python
for layer in model.layers:
	...
```
```python
layer = model.get_layer("layer_name")

name = layer.name
output = layer.output
input_shape = layer.input_shape
output_shape = layer.output_shape
weight = layer.get_weights()[0]
bias = layer.get_weights()[1]
```
#### `model.trainable_variables`
#### `model.save()`
#### `model.input`

### `tf.keras.utils`
#### `tf.keras.utils.get_file()`
```python
base_url = "https://pai-datasets.s3.ap-northeast-2.amazonaws.com/recommender_systems/movielens/datasets/"
movies_path = tf.keras.utils.get_file(fname="movies.csv", origin=os.path.join(base_url, "movies.csv"))

movie_df = pd.read_csv(movies_path)
```
- 인터넷의 파일을 로컬 컴퓨터의 홈 디렉토리 아래 `.keras/datasets` 디렉토리로 다운로드합니다.
- `untar=True`
### `tf.keras.backend`
#### `tf.keras.backend.clear_session()`
- Resets all state generated by Keras.
#### `optimizer.apply_gradients()`
```python
opt.apply_gradients(zip([dW, db], [W, b]))
```
```python
opt.apply_gradients(zip(grads, model.trainable_variables))
```
### `tf.keras.losses`
#### `tf.keras.losses.MeanSquaredError()`(= `"mse"`)
#### `tf.keras.losses.BinaryCrossentropy()`(= `"binary_crossentropy"`)
#### `tf.keras.losses.categorical_crossentropy()`
- Source: [https://hwiyong.tistory.com/335](https://hwiyong.tistory.com/335)
- 딥러닝에서 쓰이는 logit은 매우 간단합니다. 모델의 출력값이 문제에 맞게 normalize 되었느냐의 여부입니다. 예를 들어, 10개의 이미지를 분류하는 문제에서는 주로 softmax 함수를 사용하는데요. 이때, 모델이 출력값으로 해당 클래스의 범위에서의 확률을 출력한다면, 이를 logit=False라고 표현할 수 있습니다(이건 저만의 표현인 점을 참고해서 읽어주세요). 반대로 모델의 출력값이 sigmoid 또는 linear를 거쳐서 만들어지게 된다면, logit=True라고 표현할 수 있습니다.
- 클래스 분류 문제에서 softmax 함수를 거치면 `from_logits=False`(default값), 그렇지 않으면 `from_logits=True`(numerically stable)
- 정답 레이블이 one-hot encoding 형태일 경우 사용합니다.
#### `tf.keras.losses.sparse_categorical_crossentropy()`
```python
def loss_fn(model, x, y):
    return tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y_true=y, y_pred=model(x), from_logits=True))
```
- 정답 레이블이 one-hot vector가 아닐 경우 사용합니다.
##### `tf.keras.layers.experimental.preprocessing`
###### `Rescaling`
```python
model.add(Rescaling(1/255, input_shape=(img_height, img_width, 3)))
```
```python
from tensorflow.keras.layers.experimental.preprocessing import RandomFlip, RandomRotation, RandomZoom
```
###### `RandomFlip`
```python
data_aug.add(RandomFlip("horizontal", input_shape=(img_height, img_width, 3)))
```
###### `RandomRotation`
```python
data_aug.add(RandomRotation(0.1))
```
###### `RandomZoom()`
```python
data_aug.add(RandomZoom(0.1))
```
```python
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
```
### `tf.keras.initializers`
#### `tf.keras.initializers.RandomNormal()`
#### `tf.keras.initializers.glorot_uniform()`
#### `tf.keras.initializers.he_uniform()`
#### `tf.keras.initializers.Constant()`
### `tf.keras.preprocessing`
#### `tf.keras.preprocessing.image`
##### `load_img()`
```python
from tensorflow.keras.preprocessing.image import load_img
```
```python
img = load_img(fpath, target_size=(img_height, img_width))
```
##### `img_to_array()`
```python
from tensorflow.keras.preprocessing.image import img_to_array
```
```python
img_array = img_to_array(img)
```
#### `image_dataset_from_directory()`
```python
from tf.keras.preprocessing import image_dataset_from_directory
```
```python
train_ds = image_dataset_from_directory(data_dir, validation_split=0.2, subset="training",
                                        image_size=(img_height, img_width), seed=1, batch_size=batch_size)
```
```python
for image_batch, labels_batch in train_ds:
    print(image_batch.shape)
    print(labels_batch.shape)
    break
```
##### `ds.class_names`
```python
train_ds.class_names
```
##### `ds.take()`
```python
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        ax.imshow(images[i].numpy().astype("uint8"))
        ax.set_title(cls_names[labels[i]])
        ax.axis("off")
```
##### `ImageDataGenerator`
```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator
```
```python
gen = ImageDataGenerator(rescale=1/255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
```
- `validation_split`
- `shear_range` : float. Shear Intensity (Shear angle in counter-clockwise direction as radians)
- `zoom_range` : Float or [lower, upper]. Range for random zoom. If a float, [lower, upper] = [1-zoom_range, 1+zoom_range]
- `horizontal_flip` : Boolean. Randomly flip inputs horizontally.
- `rescale` : rescaling factor. Defaults to None. If None or 0, no rescaling is applied, otherwise we multiply the data by the value provided (before applying any other transformation).
- `rotation_range`
- `width_shift_range`: (Float) Fraction of total width if < 1 or pixels if >= 1. (1-D Array-like) Random elements from the Array. (Int) Pixels from interval (-`width_shift_range`, `width_shift_range`)
- `height_shift_range`
- `brightness_range` : Tuple or List of two Floats. Range for picking a brightness shift value from.
- `zoom_range`
- `horizontal_flip`
- `vertical_flip`
- transformation은 이미지에 변화를 주어서 학습 데이터를 많게 해서 성능을 높이기 위해 하는 것이기 때문에 train set만 해주고, test set에는 해 줄 필요가 없다. 그러나 주의할 것은 Rescale은 train, test 모두 해 주어야 한다.
- References: https://m.blog.naver.com/PostView.nhn?blogId=isu112600&logNo=221582003889&proxyReferer=https:%2F%2Fwww.google.com%2F
###### `gen.fit()`
- Only required if `featurewise_center` or `featurewise_std_normalization` or `zca_whitening` are set to True.
###### `gen.flow()`
```python
hist = model.fit(gen.flow(x_tr, y_tr, batch_size=32), validation_data=gen.flow(x_val, y_val, batch_size=32),
                 epochs=10)
```
###### `gen.flow_from_directory()`
```python
gen = ImageDataGenerator()
datagen_tr = gen.flow_from_directory(directory="./dogsandcats", target_size=(224, 224))
```
- `batch_size=batch_size`
- `target_size`: the dimensions to which all images found will be resized.
- `class_mode`: (`"binary"`, `"categorical"`, `"sparse"`, `"input"`, `None`)
- `class_mode="binary"`: for binary classification.
- `class_mode="categorical"`: for multi-class classification(OHE).
- `class_mode="sparse"`: for multi-class classification(no OHE).
- `class_mode="input"`
- `class_mode=None`: Returns no label.
- `subset`: (`"training"`, `"validation"`) Subset of data if `validation_split` is set in ImageDataGenerator().
- `shuffle`
#### `tf.keras.preprocessing.sequence`
##### `pad_sequences()`
```python
train_X = pad_sequences(train_X, maxlen=max_len)
```
```python
X_char = [pad_sequences([[char2idx[char] if char in chars else 1 for char in word] for word in sent]) for sent in corpus]
```
```python
train_X = pad_sequences([tokenizer.convert_tokens_to_ids(tokens) for tokens in tokens_lists], 
                        maxlen=max_len, value=tokenizer.convert_tokens_to_ids("[PAD]"),
                        truncating="post", padding="post")
```
- `padding`: (`"pre"`, `"post"`)
- `truncating`: (`"pre"`, `"post"`)
- `value=` : padding에 사용할 value를 지정합니다.
#### `tf.keras.preprocessing.text`
##### `tf.keras.preprocessing.text.Tokenizer()`
```python
tkn = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size+2, oov_token="UNK", lower=True)
```
- `lower=False`: 대문자를 유지합니다.
##### `tkn.fit_on_texts()`
```python
tkn.fit_on_texts(["나랑 점심 먹으러 갈래 점심 메뉴는 햄버거 갈래 갈래 햄버거 최고야"])
```
##### `tkn.word_index`
```python
word2idx = tkn.word_index
```
##### `tkn.index_word`
##### `tkn.word_counts`
```python
word2cnt = dict(sorted(tkn.word_counts.items(), key=lambda x:x[1], reverse=True))

cnts = list(word2cnt.values())

for vocab_size, value in enumerate(np.cumsum(cnts)/np.sum(cnts)):
    if value >= ratio:
        break

print(f"{vocab_size:,}개의 단어로 전체 data의 {ratio:.0%}를 표현할 수 있습니다.")
print(f"{len(word2idx):,}개의 단어 중 {vocab_size/len(word2idx):.1%}에 해당합니다.")
```
##### `tkn.texts_to_sequences()`
```python
train_X = tkn.texts_to_sequences(train_X)
```
- `num_words`가 적용됩니다.
##### `tkn.sequences_to_texts()`
##### `tkn.texts_to_matrix()`
```python
tkn.texts_to_matrix(["먹고 싶은 사과", "먹고 싶은 바나나", "길고 노란 바나나 바나나", "저는 과일이 좋아요"], mode="count"))
```
- `mode`: (`"count"`, `"binary"`, `"tfidf"`, `"freq"`)
- `num_words`가 적용됩니다.
### `tf.keras.models`
#### `tf.keras.models.load_model()`
```python
model = tf.keras.models.load_model(model_path)
```
### `tf.keras.applications`
#### `tf.keras.applications.VGG16()`
```python
vgg = tf.keras.applications.VGG16(input_shape=(224, 224, 3), include_top=False, weights="imagenet")
```
##### `vgg.trainable`
```python
vgg.trainable=Flase
```

# `tensorflow_addons`
```python
import tensorflow_addons as tfa
```
## `tfa.optimizers`
### `tfa.optimizers.RectifiedAdam()`
```python
opt = tfa.optimizers.RectifiedAdam(lr=5.0e-5, total_steps = 2344*4, warmup_proportion=0.1, min_lr=1e-5, epsilon=1e-08, clipnorm=1.0)
```

# `tensorflow_hub`
```python
import tensorflow_hub as hub
```
## `hub.Module()`
```python
elmo = hub.Module("https://tfhub.dev/google/elmo/3", trainable=True)
```
### `elmo()`
```python
embeddings = elmo(["the cat is on the mat", "dogs are in the fog"], signature="default", as_dict=True)["elmo"]
```

# `tensorflow_datasets`
```python
import tensorflow_datasets as tfds
```
## `tfds.deprecated`
### `tfds.deprecated.text`
#### `tfds.deprecated.text.SubwordTextEncoder`
##### `tfds.deprecated.text.SubwordTextEncoder.build_from_corpus()`
```python
tkn = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(train_data["document"], target_vocab_size=2**13)
```
##### `tkn.subwords`
##### `tkn.encode()`
```python
tkn.encode(train_data["document"][20])
```
##### `tkn.decode()`
```python
tkn.decode(tkn.encode(sample))
```

# `torch`
```python
conda install pytorch cpuonly -c pytorch
```
```python
import torch
```
## `torch.tensor()`
## `torch.empty()`
## `torch.ones()`
```python
w = torch.ones(size=(1,), requires_grad=True)
```
- Returns a tensor filled with the scalar value `1`, with the shape defined by the variable argument `size`
- `requires_grad`: (bool)
## `torch.ones_like()`
```python
torch.ones_like(input()
```
- `input`: (Tensor)
- Returns a tensor filled with the scalar value `1`, with the same size as `input`.
### `Tensor.data`
### `Tensor.shape`
### `Tensor.size()`
### `Tensor.view()`
```python
torch.randn(4, 4).view(-1, 8)
```
- Returns a new tensor with the same data as the `self` tensor but of a different `shape`.
### `Tensor.float()`
### `Tensor.backward()`
### `Tensor.grad`
- 미분할 Tensor에 대해 `requires=False`이거나 미분될 Tensor에 대해 `Tensor.backward()`가 선언되지 않으면 `None`을 반환합니다.
## `torch.optim`
### `torch.optim.SGD()`, `torch.optim.Adam()`
```python
opt = torch.optim.SGD(params=linear.parameters(), lr=0.01)
```
#### `opt.zero_grad()`
- Sets the gradients of all optimized Tensors to zero.
#### `opt.step()`
## `torch.nn`
```python
import torch.nn as nn
```
### `nn.Linear()`
```python
linear = nn.Linear(in_features=10, out_features=1)
```
- `in_features`: size of each input sample.
- `out_features`: size of each output sample.
### `linear.parameters()`
```python
for param in linear.parameters():
    print(param)
    print(param.shape)
    print('\n')
```
### `linear.weight`, `linear.bias`
#### `linear.weight.data`, `linear.bias.data`
## `nn.MSELoss()`
```python
loss = nn.MSELoss()(ys_hat, ys)
```
### `loss.backward()`
## `nn.Module`
## `nn.ModuleList()`

# `xgboost`
```python
import xgboost as xgb
```
## `xgb.DMatrix()`
```python
dtrain = xgb.DMatrix(data=train_X, label=train_y, missing=-1, nthread=-1)
dtest = xgb.DMatrix(data=test_X, label=test_y, missing=-1, nthread=-1)
```
## `xgb.train()`
```python
params={"eta":0.02, "max_depth":6, "min_child_weight":5, "gamma":1, "subsample":0.5, "colsample_bytree":1, "reg_alpha":0.1, "n_jobs":6}
watchlist = [(dtrain, "train"), (dtest,"val")]
num=12
def objective(pred, dtrain):
    observed = dtrain.get_label()
    grad = np.power(pred - observed, num - 1)
    hess = np.power(pred - observed, num - 2)
    return grad, hess
def metric(pred, dtrain):
    observed = dtrain.get_label()
    return "error", (pred - observed)/(len(observed), 1/num)
model = xgb.train(params=params, evals=watchlist, dtrain=dtrain, num_boost_round=1000, early_stopping_rounds=10, obj=objective, feval=metric)
```
## `xgb.XGBRegressor()`
```python
model = xgb.XGBRegressor(booster="gbtree", max_delta_step=0, importance_type="gain", missing=-1, n_jobs=5, reg_lambda=1, scale_pos_weight=1, seed=None, base_score=0.5, verbosity=1, warning="ignore", silent=0)
model.eta=0.01
model.n_estimators=1000
model.max_depth=6
model.min_child_weight=5
model.gamma=1
model.subsample=0.5
model.colsample_bytree=1
model.reg_alpha=0.1
model.objective = custom_se
model.n_jobs=5
```
- `n_estimators`
### `model.fit()`
```python
model.fit(train_X, train_y, eval_set=[(train_X, train_y), (val_X, val_y)], early_stopping_rounds=50, verbose=True)
```

# `seqeval`
## `seqeval.metrics`
### `precision_score`
### `recall_score`
### `f1_score`
### `classification_report`
```python
from seqeval.metrics import classification_report
```