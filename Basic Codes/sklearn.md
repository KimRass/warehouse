# sklearn.model_selection
```python
from sklearn.model_selection import train_test_split
```
## train_test_split
```python
train_X, val_X, train_y, val_y=train_test_split(train_val_X, train_val_y, train_size=0.8, shuffle=True, random_state=3)
```
# sklearn.feature_extraction.text
## CountVectorizer()
```python
from sklearn.feature_extraction.text import CountVectorizer
```
```python
vect = CountVectorizer(max_df=500, min_df=5, max_features=500)
```
- 토큰의 빈도가 max_df보다 크거나 min_df보다 작은 경우 무시.
### vect.fit()
### vect.transform() : built DTM
### vect.fit_transform() : vect.fit() + vect.transform()
### vect.vocabulary_
### vect.vocabulary_.get() : 특정 word의 index 출력
## TfidfVectorizer()
```python
from sklearn.feature_extraction.text import TfidfVectorizer
```
# sklearn.preprocessing
```python
from sklearn.preprocessing import LabelEncoder
```
## LabelEncoder()
```python
le = LabelEncoder()
le.fit(data["name_addr"])
data["id"] = le.transform(data["name_addr"])
```
# sklearn.pipeline
```python
from sklearn.pipeline import Pipeline
```
## Pipeline
```python
model = Pipeline([("vect", CountVectorizer()), ("model", SVC(kernel="poly", degree=8))])
```
- 파이프라인으로 결합된 모형은 원래의 모형이 가지는 fit, predict 메서드를 가지며 각 메서드가 호출되면 그에 따른 적절한 메서드를 파이프라인의 각 객체에 대해서 호출한다. 예를 들어 파이프라인에 대해 fit 메서드를 호출하면 전처리 객체에는 fit_transform이 내부적으로 호출되고 분류 모형에서는 fit 메서드가 호출된다. 파이프라인에 대해 predict 메서드를 호출하면 전처리 객체에는 transform이 내부적으로 호출되고 분류 모형에서는 predict 메서드가 호출된다.
# sklearn.linear_model
```python
from sklearn.linear_model import SGDClassifier
```
# sklearn.svm
```python
from sklearn.svm import SVC
```
## SVC()
```python
SVC(kernel="linear")
```
- kernel="linear"
- kernel="poly" : gamma, coef0, degree
- kernel="rbf" : gamma
- kernel="sigmoid" : gomma, coef0
# sklearn.naive_bayes
```python
from sklearn.naive_bayes import MultinomialNB
```
# sklearn.linear_model
```python
from sklearn.linear_model import SGDClassifier
```
## SGDClassifier
```python
model = SGDClassifier(loss="perceptron", penalty="l2", alpha=1e-4, random_state=42, max_iter=100)
...
model.fit(train_x, train_y)
train_pred = model.pred(train_x)
train_acc = np.mean(train_pred == train_y)
```
- loss : The loss function to be used.
    - "hinge" : gives a linear SVM.
    - "log" : gives logistic regression.
    - "perceptron" : the linear loss used by the perceptron algorithm.
- penalty : regularization term.
    - "l1"
    - "l2" : the standard regularizer for linear SVM models.
- "alpha" : constant that multiplies the regularization term. The higher the value, the stronger the regularization. Also used to compute the learning rate when set to learning_rate is set to ‘optimal’.
- "max_iter" : The maximum number of passes over the training data (aka epochs).
# sklearn.datasets.sample_generator
## make_blobs()
```python
 from sklearn.datasets.sample_generator improt make_blobs
```
# sklearn.linear_model
## SGDClassifier()
```python
from sklearn.linear_model import SGDClassifier
```
```python
model = Pipeline([("vect", CountVectorizer()), ("model", SGDClassifier(loss="perceptron", penalty="l2", alpha=1e-4, random_state=42, max_iter=100))])
```
# sklearn.naive_bayes
## MultinomialNB()
```python
from sklearn.naive_bayes import MultinomialNB
```
# sklearn.metrics.pairwise
```python
from sklearn.metrics.pairwise import cosine_similarity
```
## cosine_similarity
