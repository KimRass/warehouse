# Python Built-in Functions
## var.data
### var.data.nbytes : 변수에 할당된 메모리 크기 리턴
## print()
```python
print(... end="") 
```
### print() + format()
```python
print("{0} and {1}".format("spam", "eggs")
```
```python
print("[{0:>4d}], [{1:>20d}]".format(100, 200))
```
```python
print("[{0:<20s}]".format("string"))
```
```python
print("[{0:<20.20f}], [{1:>10.2f}]".format(3.14, 10.925))
```
```python
print("{:>4d} | {:>7.4f} | {:>7.4f} | {:>9.6f}".format(i + 1, W.numpy(), b.numpy(), loss.numpy()))
```
## isinstance()
```python
if not isinstance(movie, frozenset):
    movie = frozenset(movie)
```
## assert
```python
assert model_name in self.model_list, "There is no such a model."
```
## list
### list[]
### list.index()
```python
names.index((17228, '아트빌'))
```
### list.append()
```python
feature_to_shuffle.append("area")
```
### list.remove()
```python
features.remove("area")
```
### list.sort()
```python
A.sort(reverse=True)
```
```python
m.sort(key=len)
```
- in-place 함수
### sorted(), reversed()
```python
A = reversed(A)
```
### str.join()
```python
" ".join(["good", "bad", "worse", "so good"])
```
- str을 사이에 두고 리스트의 모든 원소들을 하나로 합침
### map()
```python
x_data = list(map(lambda word : [char2idx.get(char) for char in word], words))
```
### split()
```python
msg_tkn = [msg.split() for msg in data["msg"]]
```
### filter()
### sum()
```python
sum(sents, [])
```
## set
### set1 & set2
### set1 | set2
### set.add()
### set.update()
- list.append()와 동일.
## frozenset()
- 구성 요소들이 순서대로 들어 있지 않아 인덱스를 사용한 연산을 할 수 없고
- 유일한 항목 1개만 들어가 있습니다.
- set 객체는 구성 항목들을 변경할 수 있지만 frozenset 객체는 변경할 수 없는 특성을 지니고 있어, set에서는 사용할 수 있는 메소드들 add, clear, discard, pop, remove, update 등은 사용할 수 없습니다. 
- 항목이 객체에 포함되어 있는지(membership)를 확인할 수 있으며,  항목이 몇개인지는 알아낼 수 있습니다.
- frozenset은 dictionary의 key가 될 수 있지만 set은 될 수 없음.
## dictionary
### dic[key]
### dic[key] = value
```python
aut2doc = {}
for user, idx in auts.groups.items():
    aut2doc[user] = list(idx)
```
### dic.items()
```python
for key, value in dic.items():
    print(key, value)
```
### dic.setdefault() : 추가
```python
dic.setdefault(key)
dic.setfefault(key, value)
```
### dic.update() : 추가 또는 수정
```python
dic.update({key1:value1, key2:value2})
```
### dic.pop() : 삭제
```python
dic.pop(key)
```
- \>\>\> value
### dic.get()
```python
dic.get(key)
```
- \>\>\> value
### dic.keys(), dic.values()
### dic.fromkeys(list or tuple, value)
### dictionary comprehension
```python
{idx:char for idx, char in enumerate(char_set)}
```
## exec()
```python
for i in range(N):
    exec(f"a{i} = int(input())")
```
## eval()
```python
A = [eval(f"A{i}") for i in range(N, 0, -1)]
```
- string 앞뒤의 " 제거
# class
## instance attribute(instance variables)
```python
class CLASS:
    def __init__(self):
        self.ATTRIBUTE = VALUE
```
* INSTANCE.ATTRIBUTE로 사용
## class attriubute(class variables)
```python
class CLASS:
    ATTRIBUTE = VALUE
```
* CLASS.ATTRIBUTE로 사용
* 모든 INSTANCE가 ATTRIBUTE 값을 공유.
* 동일한 instance attribute와 class attribute가 있으면 instance attribute -> class attribute 순으로 method를 탐색.
* INSTANCE.ATTRIBUTE로 사용 시 INSTANCE의 namespace에서 ATTRIBUTE를 찾고 없으면 CLASS의 namespace로 이동한 후 다시 ATTRIBUTE를 찾아 그 값을 반환.
## \_\_init\_\_
```python
INSTANCE = CLASS() #instance를 initiate 할 때 실행
```
## \_\_init\_\_, \_\_call\_\_
```python
class CLASS:
    def __init__(self, parameter1, parameter2, ...)
        ...
        
    def __call__(self, parameter3, parameter4, ...)
        ...
        return ...
        
    def FUNCTION(self, parameter5, parameter6, ...)
        ...
        return ...
...
INSTANCE = CLASS(parameter1, parameter2, ...) #__init__문은 instance를 initiate 할 때 실행
INSTANCE(parameter3, parameter4, ...) #__call__문은 instance를 call 할 때 실행
INSTANCE.FUNCTION(parameter5, parameter6, ...)
```
## method
- method : class 정의문 안에서 정의된 함수
- method의 첫번째 parameter는 항상 self여야 함
- method의 첫 번째 parameter는 self지만 호출할 때는 아무것도 전달하지 않는 이유는 첫 번째 parameter인 self에 대한 값은 파이썬이 자동으로 넘겨주기 때문입니다.

### class variables
- class 정의문 안에서 정의된 variables
### instance variables
- self가 붙어 있는 variables
## override
- 출처 : https://rednooby.tistory.com/55
## super()
- 출처 : https://rednooby.tistory.com/56?category=633023
```python
class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
```
# open()
```python
with open("C:/Users/5CG7092POZ/nsmc-master/ratings_train.txt", "r", encoding="utf-8") as f:
    train_docs = [line.split("\t") for line in f.read().splitlines()][1:]
```
## input()
```python
A = list(map(int, input("A를 차례대로 입력 : ").split()))
```
## ord()

# pandas
```python
import pandas as pd
```
## pd.set_option()
```python
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option("display.float_format", "{:.3f}".format)
```
## pd.DataFrame()
```python
list_for_data = [(re.match(r"(\[)(\w+)(\])", line[0]).group(2), re.findall(r"(\d\] )(.*)$", line[0])[0][1]) for line in raw_data if re.match(r"(\[)(\w+)(\])", line[0])]
data = pd.DataFrame(list_for_data, columns=["user", "content"])
```
## pd.read_csv()
```python
raw_data = pd.read_csv("C:/Users/00006363/☆데이터/실거래가_충청북도_2014.csv", thousands=",", encoding="Ansi", float_precision="round_trip", skiprows=15)
```
## pd.read_excel()
## pd.read_pickle()
## df.to_csv()
```python
data.to_csv("D:/☆디지털혁신팀/☆실거래가 분석/☆데이터/실거래가 전처리 완료_200928-3.csv", index=False)
```
## df.to_pickle()
## pd.concat()
```python
data_without = pd.concat([data_without, data_subset], axis=0)
```
## pd.melt()
```python
data = pd.melt(raw_data, id_vars=["세부직종별"], var_name="입찰년월", value_name="노임단가")
```
## pd.pivot_table()
```python
ui = pd.pivot_table(ratings_df_tr, index="user_id", columns="movie_id", values="rating")
```
```python
pd.pivot_table(df, index="요일", columns="지역", aggfunc=np.mean)
```
- pd.melt()의 반대
## pd.Categorical()
```python
results["lat"] = pd.Categorical(results["lat"], categories=order)
results_ordered=results.sort_values(by="lat")
```
- dtype을 category로 변환.
- ordered=True로 category들 간에 대소 부여.
- 출처 : [https://kanoki.org/2020/01/28/sort-pandas-dataframe-and-series/](https://kanoki.org/2020/01/28/sort-pandas-dataframe-and-series/)
## pd.get_dummies()
```python
data = pd.get_dummies(data, columns=["heating", "company1", "company2", "elementary", "built_month", "trade_month"], drop_first=False, dummy_na=True)
```
* 결측값이 있는 경우 "drop\_first", "dummy\_na" 중 하나만 True로 설정해야 함
## pd.to_datetime()
```python
ratings_df["rated_at"] = pd.to_datetime(ratings_df["rated_at"], unit="s")
```
- timestamp -> 초 단위로 변경
## pd.merge()
```python
data = pd.merge(data, start_const, on=["지역구분", "입찰년월"], how="left")
```
```python
pd.merge(df1, df2, left_on="id", right_on="movie_id")
```
```python
floor_data = pd.merge(floor_data, df_conv, left_index=True, right_index=True, how="left")
```
- df와 df 또는 df와 ser 간에 사용 가능.
## df.shape
## df.quantile()
```python
top90per = plays_df[plays_df["plays"]>plays_df["plays"].quantile(0.1)]
```
## df.groupby()
```python
df.groupby(["Pclass", "Sex"], as_index=False)
```
### df.groupby().groups
### df.groupby().mean()
### df.groupby().size()
- 형태 : ser
### df.groupby().count()
- 형태 : df
### df.groupby()[].apply(set)
```python
over4.groupby("user_id")["movie_id"].apply(set)
```
## df.pivot()
```python
df_pivoted = df.pivot("col1", "col2", "col3")
```
- 참고자료 : [https://datascienceschool.net/view-notebook/4c2d5ff1caab4b21a708cc662137bc65/](https://datascienceschool.net/view-notebook/4c2d5ff1caab4b21a708cc662137bc65/)
## df.stack()
- 열 인덱스 -> 행 인덱스로 변환
## df.unstack()
- 행 인덱스 -> 열 인덱스로 변환
- pd.pivot_table()과 동일
- level에는 index의 계층이 정수로 들어감
```python
groupby.unstack(level=-1, fill_value=None)
```
## df.apply()
```python
data["반기"]=data["입찰년월"].apply(lambda x:x[:4]+" 상반기" if int(x[4:])<7 else x[:4]+" 하반기")
data["반기"]=data.apply(lambda x:x["입찰년월"][:4]+" 상반기" if int(x["입찰년월"][4:])<7 else x["입찰년월"][:4]+" 하반기", axis=1)
```
```python
data["1001-작업반장]=data["반기"].apply(lambda x:labor.loc[x,"1001-작업반장])
data["1001-작업반장]=data.apply(lambda x:labor.loc[x["반기"],"1001-작업반장], axis=1)
```
## df.rename()
```python
data = data.rename({"단지명.1":"name", "세대수":"houses_buildings", "저/최고층":"floors"}, axis=1)
```
## df.insert()
```python
data.insert(3, "age2", data["age"]*2)
```
## df.sort_values()
```python
results_groupby_ordered = results_groupby.sort_values(by=["error"], ascending=True, na_position="first", axis=0)
```
## df.nlargest(), df.nsmallest()
```python
df.nlargest(3, ["population", "GDP"], keep="all")
```
- keep="first" | "last" | "all"
## df.index
## df.index.names
```python
df.index.name = None
```
## df.sort_index()
## df.set_index()
```python
data=data.set_index(["id", "name"])
df.set_index("Country", inplace=True)
```
## df.reset_index()
```python
cumsum.reset_index(drop=True)
```
## df.loc()
```python
data.loc[data["buildings"]==5, ["age", "ratio2"]]
data.loc[[7200, "대림가경"], ["houses", "lowest"]]
```
## df.isin()
```python
train_val = data[~data["name"].isin(names_test)]
```
- dictionary와 함께 사용 시 key만 가져옴
## df.query()
```python
data.query("houses in @list")
```
- 외부 변수 또는 함수 사용 시 앞에 @을 붙임.
## df.idxmax()
```python
data["genre"] = data.loc[:, "unknown":"Western"].idxmax(axis=1)
```
## df.drop()
```python
data = data.drop(["Unnamed: 0", "address1", "address2"], axis=1)
```
```python
data = data.drop(data.loc[:, "unknown":"Western"].columns, axis=1)
```
## df.duplicated()
```python
df.duplicated(keep="first)
```
## df.columns
```python
concat.columns = ["n_rating", "cumsum"]
```
### df.columns.droplevel
```python
df.columns=df.columns.droplevel([0, 1])
```
## df.drop_duplicates()
```python
df = df.drop_duplicates(["col1"], keep="first")
```
## df.mul()
```python
df1.mul(df2)
```
## df.dot()
```python
def cos_sim(x, y):
    return x.dot(y)/(np.linalg.norm(x, axis=1, ord=2)*np.linalg.norm(y, ord=2))
```
## df.isna()
## df.notna()
```python
retail[retail["CustomerID"].notna()]
```
## df.dropna()
```python
data = data.dropna(subset=["id"])
```
## df.dropna(axis=0)
## df.quantile()
```python
Q1 = subset["money"].quantile(0.25)
```
## df.sample()
```python
ratings_df.sample(5)
```
```python
baskets_df.sample(frac=0.05)
```
## df.mean()
```python
ui.mean(axis=1)
```
## df.mean().mean()
## df.add(), df.sub(). df.mul(), df.div(), df.pow()
```python
adj_ui = ui.sub(user_bias, axis=0).sub(item_bias, axis=1)
```
## ser.rename()
```python
plays_df.groupby(["user_id"]).size().rename("n_arts")
```
## ser.value_counts()
```python
ratings_df["movie_id"].value_counts()
```
## ser.nunique()
```python
n_item = ratings_df["movie_id"].nunique()
```
## ser.isnull()
## ser.map()
```python
target_ratings["title"] = target_ratings["movie_id"].map(target)
```
```python
all_seen = ratings_df_target_set[ratings_df_target_set.map(lambda x : len(x)==5)].index
```
## ser.astype()
```python
data.loc[:, cats] = data.loc[:, cats].astype("category")
```
- "int32", "int63", "float64", "object", "category"
- np.uint8 : 0~255, np.uint16 : 0~65,535, np.uint32 : 0~4,294,967,295
## ser.hist()
## ser.cumsum()
```python
cumsum = n_rating_item.cumsum()/len(ratings_df)
```
## ser.min(), ser.max(), ser.mean(), ser.std()
## ser.str
### ser.str.replace()
```python
data["parking"] = data["parking"].str.replace("대", "", regex=False)
```
### ser.str.split()
```python
data["buildings"] = data.apply(lambda x : str(x["houses_buildings"]).split("총")[1][:-3], axis=1)
```
- ser.str.split()는 ser.str.split(" ")과 같음
### ser.str.strip()
```python
data["fuel"] = data["heating"].apply(lambda x:x.split(",")[1]).str.strip()
```
### ser.str.contains()
```python
raw_data[raw_data["시군구"].str.contains("충청북도")]
```
## ser.cat
### ser.cat.categories
### ser.cat.set_categories()
```python
ser.cat.set_categories([2, 3, 1], ordered=True)
```
- 순서 부여
### ser.cat.codes
```python
for cat in cats:
    data[cat] = data[cat].cat.codes
```
- label encoding을 시행합니다.
## ser.items()
```python
for k, v in target.items():
    queries.append(f"{k}-{v}")
```
# sklearn
```python
!pip install -U scikit-learn
```
## sklearn.model_selection
### train_test_split
```python
from sklearn.model_selection import train_test_split
```
```python
train_X, val_X, train_y, val_y=train_test_split(train_val_X, train_val_y, train_size=0.8, shuffle=True, random_state=3)
```
## sklearn.feature_extraction.text
### CountVectorizer()
```python
from sklearn.feature_extraction.text import CountVectorizer
```
```python
vect = CountVectorizer(max_df=500, min_df=5, max_features=500)
```
- 토큰의 빈도가 max_df보다 크거나 min_df보다 작은 경우 무시.
#### vect.fit()
#### vect.transform() : built DTM
#### vect.fit_transform() : vect.fit() + vect.transform()
#### vect.vocabulary_
##### vect.vocabulary_.get() : 특정 word의 index 출력
### TfidfVectorizer()
```python
from sklearn.feature_extraction.text import TfidfVectorizer
```
## sklearn.preprocessing
```python
from sklearn.preprocessing import LabelEncoder
```
## sklearn.decomposition
### PCA()
```python
from sklearn.decomposition import PCA
```
```python
pca = PCA(n_components=2)
```
```python
pca_mat = pca.fit_transform(user_emb_df)
```
### LabelEncoder()
```python
le = LabelEncoder()
le.fit(data["name_addr"])
data["id"] = le.transform(data["name_addr"])
```
## sklearn.pipeline
```python
from sklearn.pipeline import Pipeline
```
### Pipeline
```python
model = Pipeline([("vect", CountVectorizer()), ("model", SVC(kernel="poly", degree=8))])
```
- 파이프라인으로 결합된 모형은 원래의 모형이 가지는 fit, predict 메서드를 가지며 각 메서드가 호출되면 그에 따른 적절한 메서드를 파이프라인의 각 객체에 대해서 호출한다. 예를 들어 파이프라인에 대해 fit 메서드를 호출하면 전처리 객체에는 fit_transform이 내부적으로 호출되고 분류 모형에서는 fit 메서드가 호출된다. 파이프라인에 대해 predict 메서드를 호출하면 전처리 객체에는 transform이 내부적으로 호출되고 분류 모형에서는 predict 메서드가 호출된다.
## sklearn.linear_model
```python
from sklearn.linear_model import SGDClassifier
```
## sklearn.svm
```python
from sklearn.svm import SVC
```
### SVC()
```python
SVC(kernel="linear")
```
- kernel="linear"
- kernel="poly" : gamma, coef0, degree
- kernel="rbf" : gamma
- kernel="sigmoid" : gomma, coef0
## sklearn.naive_bayes
```python
from sklearn.naive_bayes import MultinomialNB
```
## sklearn.linear_model
```python
from sklearn.linear_model import SGDClassifier
```
### SGDClassifier
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
## sklearn.datasets.sample_generator
### make_blobs()
```python
 from sklearn.datasets.sample_generator improt make_blobs
```
## sklearn.metrics.pairwise
```python
from sklearn.metrics.pairwise import cosine_similarity
```
### cosine_similarity
# tensorflow
```
conda create --name tf2.0 python=3.7
pip install tensorflow==2.0
conda install jupyter
```
```python
import tensorflow as tf
```
## tf.multiply()

## tf.square()

* 각 arguments를 제곱하여 ndarray 생성

## tf.reduce\_sum()

* axis=0 \| 1

## tf.reduce\_mean()

* axis=0 \| 1

## tf.argmax()

* axis=0 \| 1

## assign

```python
W.assign(W - tf.multiply(lr, dW))
```

## assign_sub

```python
W.assign_sub(tf.multiply(lr, dW))
```

## tf.sign

```python
tf.sign(tf.reduce_sum(self.w * x) + self.b)
```

## tf.exp()

## tf.math.log()

## tf.sigmoid()

## tf.constant()

```python
image = tf.constant([[[[1], [2], [3]], [[4], [5], [6]], [[7], [8], [9]]]], dtype=tf.float32)
```

## tf.convert\_to\_tensor()

```python
img = tf.convert_to_tensor(img)
```

## tf.Variable()

## tf.zeros()

```python
W = tf.Variable(tf.zeros([2, 1], dtype=tf.float32), name = "weight")
```

## tf.transpose()

## tf.zeros()

```python
tf.zeros([2, 1])
```

## tf.cast()

```python
pred = tf.cast(h > 0.5, dtype=tf.float32)
```

* 조건이 True면 1, False면 0 반환.
* 혹은 단순히 Tensor의 자료형 변환.

## tf.equal()

```python
acc = tf.reduce_mean(tf.cast(tf.equal(pred, labels), dtype=tf.float32))
```

## tf.concat()

```python
layer3 = tf.concat([layer1, layer2], axis=1)
```

## tf.reshape()

```python
layer3 = tf.reshape(layer3, shape=[-1, 2])
```

## tf.constant\_initializer()

```
weight_init = tf.constant_initializer(weight)
```

# tf.random

## tf.random.set\_seed()

## tf.random.normal()

```python
x = tf.Variable(tf.random.normal([784, 200], 1, 0.35))
```

## tf.keras

### tf.keras.utils
#### tf.keras.utils.get_file()
```python
base_url = "https://pai-datasets.s3.ap-northeast-2.amazonaws.com/recommender_systems/movielens/datasets/"

movies_path = tf.keras.utils.get_file("movies.csv", os.path.join(base_url, "movies.csv"))
movie_df = pd.read_csv(movies_path)
```
- 인터넷의 파일을 로컬 컴퓨터의 홈 디렉토리 아래 .keras/datasets 디렉토리로 다운로드.
#### tf.keras.utils.to_categorical()
```python
train_labels = tf.keras.utils.to_categorical(train_labels, 10)
```
* one-hot encoding

### @tf.function

* 자동 그래프 생성
* 함수 정의문 직전에 사용

### tf.GradientTape()

```python
with tf.GradientTape() as tape:
    hyp = W * X + b
    loss = tf.reduce_mean(tf.square(hyp - y))

dW, db = tape.gradient(loss, [W, b])
```

### tf.keras.datasets

#### tf.keras.datasets.mnist

```python
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
```

### tf.keras.losses

#### tf.keras.losses.categorical_crossentropy()
```python
def loss_fn(model, images, labels):
    logits = model(images, training=True)
    loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_true=labels, y_pred=logits, from_logits=True))
    return loss
```
* 출처 : [https://hwiyong.tistory.com/335](https://hwiyong.tistory.com/335)
* 딥러닝에서 쓰이는 logit은 매우 간단합니다. 모델의 출력값이 문제에 맞게 normalize 되었느냐의 여부입니다. 예를 들어, 10개의 이미지를 분류하는 문제에서는 주로 softmax 함수를 사용하는데요. 이때, 모델이 출력값으로 해당 클래스의 범위에서의 확률을 출력한다면, 이를 logit=False라고 표현할 수 있습니다(이건 저만의 표현인 점을 참고해서 읽어주세요). 반대로 모델의 출력값이 sigmoid 또는 linear를 거쳐서 만들어지게 된다면, logit=True라고 표현할 수 있습니다.
* 결론: 클래스 분류 문제에서 softmax 함수를 거치면 from_logits = False(default값), 그렇지 않으면 from_logits = True.
* 텐서플로우에서는 softmax 함수를 거치지 않고, from_logits = True를 사용하는게 numerical stable하다고 설명하고 있다.
* training=True : tf.keras.layers.Dropout() 적용
- 정답 레이블이 one-hot encoding 형태일 경우 사용.
#### tf.keras.losses.sparse_categorical_crossentropy()
```python
def loss_fn(model, x, y):
    return tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y_true=y, y_pred=model(x), from_logits=True))
```
- 정답 레이블이 one-hot encoding 형태가 아닐 경우 사용.
### tf.keras.optimizers

#### tf.keras.optimizers.Adam()

```python
opt = tf.keras.optimizers.Adam(learning_rate=lr)
```

#### tf.keras.optimizers.SGD()

```python
opt = tf.keras.optimizers.SGD(lr=0.01)
```

#### optimizer.apply_gradients()

```python
opt.apply_gradients(zip([dW, db], [W, b]))
```

```python
opt.apply_gradients(zip(grads, model.trainable_variables))
```

### tf.keras.metrics

```python
tf.keras.metrics.Mean(name="test_loss")
```

```python
tf.keras.metrics.SparseCategoricalAccuracy(name="test_accuracy")
```

### tf.keras.layers
#### tf.keras.layers.Dot()
```python
pos_score = Dot(axes=(1,1))([user_embedding, pos_item_embedding])
```
- axes : Integer or tuple of integers, axis or axes along which to take the dot product. If a tuple, should be two integers corresponding to the desired axis from the first input and the desired axis from the second input, respectively. Note that the size of the two selected axes must match.
#### tf.keras.layers.Flatten()

```python
def flatten() :
    return tf.keras.layers.Flatten()
```

#### tf.keras.layers.Dense()

```python
def dense(label_dim, weight_init) :
    return tf.keras.layers.Dense(units=label_dim, use_bias=True, kernel_initializer=weight_init)
```

* units : 출력 값 차원

#### tf.keras.layers.Activation()

* tf\.keras\.activations\.sigmoid \| tf\.keras\.activations\.relu

#### tf.keras.layers.Dropout()

```python
def dropout(rate):
    return tf.keras.layers.Dropout(rate)
```

* rate : dropout을 적용할 perceptron의 비율

#### tf.keras.layers.BatchNormalization()

```python
def batch_norm()
    return tf.keras.layers.BatchNormalization()
```

#### tf,keras.layers.Conv2D()

```python
conv2d = keras.layers.Conv2D(filters=3, kernel_size=2, padding="same", data_format="channels_last", kernel_initializer=weight_init)(image)
```

* image : (batch, height, width, number of channels)
* filter : (height, width, number of channels, number of filters)
* convolution : (batch, height, width, number of filters)
* image의 number of channels와 filter의 number of filters의 값은 동일
* fliters : filter 개수.
* kernal\_size : filter의 사이즈(int, tuple, list 가능).
* strides : stride(int, tuple, list 가능).
* padding : "valid" \| "same"

#### tf.keras.layers.MaxPool2D()
```python
pool = keras.layers.MaxPool2D(pool_size=(2, 2), strides=1, padding="valid", data_format="channels_last")(image)
```
#### tf.keras.layers.SimpleRNNCell()
```python
cell = tf.keras.layers.SimpleRNNCell(units=hidden_size)
```
#### tf.keras.layers.RNN()
```python
rnn = tf.keras.layers.RNN(cell, return_sequences=True, return_state=True)

outputs, states = rnn(x_data)
```
- return_sequences : default는 False이며 time step의 마지막에서만 아웃풋을 출력. True인 경우 모든 time step에서 아웃풋을 출력. return_sequences 인자에 따라 마지막 시퀀스에서 한 번만 출력할 수 있고 각 시퀀스에서 출력을 할 수 있습니다. many to many 문제를 풀거나 LSTM 레이어를 여러개로 쌓아올릴 때는 return_sequence=True 옵션을 사용합니다.
#### tf.keras.layers.SimpleRNN()
```python
rnn = tf.keras.layers.SimpleRNN(units=hidden_size, return_sequences=True, return_state=True)

outputs, states = rnn(x_data)
```
- tf.keras.layers.SimpleRNN() = tf.keras.layers.SimpleRNNCell() + tf.keras.layers.RNN()
- 입력 값의 차원 : (batch_size, sequence length, input dimension)
- outputs의 차원 : (batch_size, sequence length, hidden_size)
- states의 차원 : (batch size, hidden_size)
#### tk.keras.layers.Embedding()
```python
one_hot = np.eye(len(char2idx))

model.add(tf.keras.layers.Embedding(input_dim=input_dim, output_dim=output_dim, trainable=False, mask_zero=True, input_length=max_sequence, embeddings_initializer=tf.keras.initializers.Constant(one_hot)))
```
- input dim : 입력되는 단어의 개수.
- output_dim : 출력되는 embedding vector의 크기
- input_length : 입력 sequence의 길이
- trainable : one-hot vector의 training을 건너뛸지 여부.
- mask_zero : 0인 값 무시 여부. If mask_zero is set to True, as a consequence, index 0 cannot be used in the vocabulary. so input_dim should equal to size of vocabulary + 1.
- embeddings_initializer : Initializer for the embeddings matrix
#### tf.keras.layers.TimeDistributed()
```python
model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dropout(rate=0.2)))
```
- TimeDistributed를 이용하면 각 time에서 출력된 아웃풋을 내부에 선언해준 레이어와 연결시켜주는 역할을 합니다. 아래 예제에서는 Dense(unit=1)로 연결을 했고, 이는 RNN Cell의 가중치와 마찬가지로 모든 step step에서 가중치를 공유합니다.
#### tf.keras.layers.Layer
- custom layer를 만들려면 `tf.keras.layers.Layer` 클래스를 상속하고 다음 메서드를 구현합니다
    - __init__: 이 층에서 사용되는 하위 층을 정의할 수 있습니다.
    - build: 층의 가중치를 만듭니다. add_weight 메서드를 사용해 가중치를 추가합니다.
    - call: 정방향 패스를 구현합니다.
### tf.keras.initializers

#### tf.keras.initializers.RandomNormal()

#### tf.keras.initializers.glorot_uniform()

#### tf.keras.initializers.he_uniform()
#### tf.keras.initializers.Constant()
* tf.keras.layers.Activation(tf.keras.activations.relu) 사용 시 선택
### tf.keras.Model, tf.keras.Sequential()

```python
class CreateModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.flatten = tf.keras.layers.Flatten(input_shape=(28, 28))
        self.dense1 = tf.keras.layers.Dense(32, activation = 'relu')
        self.dense2 = tf.keras.layers.Dense(64, activation = 'relu')
        self.dense3 = tf.keras.layers.Dense(128, activation = 'relu')
        self.dense4 = tf.keras.layers.Dense(256, activation = 'relu')
        self.dense5 = tf.keras.layers.Dense(10, activation = 'softmax')

    def __call__(self, x, training=None, mask=None):
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dense4(x)
        return self.dense5(x)
```

```python
class CreateModel(tf.keras.Model):
    def __init__(self, label_dim):
        super(CreateModel, self).__init__()

        self.model = tf.keras.Sequential()
        self.model.add(flatten()) 

        rand_norm = tf.keras.initializers.RandomNormal()
        for i in range(2):
            self.model.add(dense(units=256, rand_norm))
            self.model.add(sigmoid())

        self.model.add(dense(label_dim, rand_norm))

    def call(self, x, training=None, mask=None):
        y = self.model(x)
        return y
```
#### model.summary()
#### model.layers
##### layer.name
##### layer.output_shape
##### layer.get_weights()
###### weight.shape
```python
for layer in model.layers[1:]:
    weight = layer.get_weights()[0]
    print(f"{layer.name}의 weight shape : {weight.shape}")
```
```python
for layer in model.layers[1:]:
    bias = layer.get_weights()[1]
    print(f"{layer.name}의 bias shape : {bias.shape}")
```
### tf.keras.Input()

```python
inputs = tf.keras.Input(shape=(28, 28, 1))
```
### tf.keras.preprocessing.sequence.pad_sequences()
```python
x_data = tf.keras.preprocessing.sequence.pad_sequences(sequences=x_data, maxlen=max_sequence, padding="post", truncating="post", value=0)
```
- padding="pre" | "post"
- truncating="pre" | "post"
## tf.nn

### tf.nn.softmax()

```python
h = tf.nn.softmax(tf.matmul(train_X, W) + b)
```

### tf.nn.relu

```python
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, padding="same", activation=tf.nn.relu, input_shape=(28, 28, 1)))
```

## tf.data.Dataset

### tf.data.Dataset.from_tensor_slices()

```python
train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y)).shuffle(len(train_x)).batch(batch_size, drop_remainder=True).prefetch(batch_size)

test_dataset = tf.data.Dataset.from_tensor_slices((test_x, test_y)).shuffle(len(test_x)).batch(len(test_x)).prefetch(len(test_x))
```

* shuffle() : 지정한 개수의 데이터를 무작위로 섞어서 출력.
* batch() : 지정한 개수의 데이터를 묶어서 출력.
* prefetch() : This allows later elements to be prepared while the current element is being processed. This often improves latency and throughput, at the cost of using additional memory to store prefetched elements.

## tf.train

### tf.train.Checkpoint()

```python
ckpt = tf.train.Checkpoint(cnn=model)
...
ckpt.save(file_prefix=ckpt_prefix)
```
# tensorflow_hub
```python
import tensorflow_hub as hub
```
## hub.Module()
```python
elmo = hub.Module("https://tfhub.dev/google/elmo/3", trainable=True)
```
### elmo()
```python
embeddings = elmo(["the cat is on the mat", "dogs are in the fog"], signature="default", as_dict=True)["elmo"]
```

# bs4
## BeautifulSoup
```python
from bs4 import BeautifulSoup as bs
```
### bs()
```python
soup = bs(xml,"lxml")
```
#### soup.find_all()
##### soup.find_all().find()
###### soup.find_all().find().get_text()
```python
features = ["bjdcode", "codeaptnm", "codehallnm", "codemgrnm", "codesalenm", "dorojuso", "hocnt", "kaptacompany", "kaptaddr", "kaptbcompany",  "kaptcode", "kaptdongcnt", "kaptfax", "kaptmarea", "kaptmarea",  "kaptmparea_136", "kaptmparea_135", "kaptmparea_85", "kaptmparea_60",  "kapttarea", "kapttel", "kapturl", "kaptusedate", "kaptdacnt", "privarea"]
for item in soup.find_all("item"):
    for feature in features:
        try:
            kapt_data.loc[index, feature] = item.find(feature).get_text()
        except:
            continue
```
# selenium
## webdriver
```python
from selenium import webdriver
```
```python
driver = webdriver.Chrome("chromedriver.exe")
```
### driver.get()
```python
driver.get("https://www.google.co.kr/maps/")
```
### driver.find_element_by_css_selector(), driver.find_element_by_tag_name(), driver.find_element_by_class_name(), driver.find_element_by_id(), driver.find_element_by_xpath(),
#### driver.find_element_by_*().text
```python
df.loc[index, "배정초"]=driver.find_element_by_xpath("//\*[@id='detailContents5']/div/div[1]/div[1]/h5").text
```
#### driver.find_element_by_*().get_attribute()
```python
driver.find_element_by_xpath("//*[@id='detailTab" +str(j) + "']").get_attribute("text")
```
#### driver.find_element_by_*().click()
#### driver.find_element_by_\*().clear()
```python
driver.find_element_by_xpath('//*[@id="searchboxinput"]').clear()
```
#### driver.find_element_by_\*().send_keys()
```python
driver.find_element_by_xpath('//*[@id="searchboxinput"]').send_keys(qeury)
```
```python
driver.find_element_by_name('username').send_keys(id)
driver.find_element_by_name('password').send_keys(pw)
```
```python
driver.find_element_by_xpath('//*[@id="wpPassword1"]').send_keys(Keys.ENTER)
```
### driver.execute_script()
```python
for j in [4,3,2]:
    button = driver.find_element_by_xpath("//\*[@id='detailTab"+str(j)+"']")
    driver.execute_script("arguments[0].click();", button)
```
### driver.implicitly_wait()
```python
driver.implicitly_wait(1)
```
### driver.current_url
### driver.save_screenshot()
```python
driver.save_screenshot(screenshot_title)
```
## WebDriverWait(), By, EC
```python
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
```
```python
WebDriverWait(driver, wait_sec).until(EC.presence_of_element_located((By.XPATH,"//\*[@id='detailContents5']/div/div[1]/div[1]/h5")))
```
## ActionChains()
```python
from selenium.webdriver import ActionChains
```
```python
module=["MDM","사업비","공사","외주","자재","노무","경비"]

for j in module:
    module_click=driver.find_element_by_xpath("//div[text()='"+str(j)+"']")
    actions=ActionChains(driver)
    actions.click(module_click)
    actions.perform()
```
### actions.click(), actions.double_click()
# urllib
```python
import urllib
```
## urllib.requests.urlopen().read().decode()
```python
xml = urllib.request.urlopen(full_url).read().decode("utf8")
```

------------------------------------------------------------------------------------
```python
!pip install --upgrade category_encoders
```
# category_encoders
```python
import category_encoders as ce
```
## ce.target_encoder
### ce.target_encoder.TargetEncoder()
```python
encoder = ce.target_encoder.TargetEncoder(cols=["company1"])
encoder.fit(data["company1"], data["money"]);
data["company1_label"] = encoder.transform(data["company1"]).round(0)
```

# collections
## deque
```python
from collections import deque
```

# cv2
```python
!pip install opencv-python
```
```python
import cv2
```
## cv2.waitKey()
```python
k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
```
## cv2.VideoCapture()
```python
cap = cv2.VideoCapture(0)
```
## cv2.destroyAllWindows()
```python
cv2.destroyAllWindows()
```
## cv2.rectangle()
```python
for i, rect in enumerate(rects_selected):
    cv2.rectangle(img, (rect[0], rect[1]), (rect[0]+rect[2], rect[1]+rect[3]), (0, 0, 255), 2)
```
## cv2.circle()
```python
for i, rect in enumerate(rects_selected):
    cv2.circle(img, (rect[0]+1, rect[1]-12), 12, (0, 0, 255), 2))
```
## cv2.puttext()
```python
for i, rect in enumerate(rects_selected):
    cv2.putText(img, str(i+1), (rect[0]-5, rect[1]-5), fontFace=0, fontScale=0.6, color=(0, 0, 255), thickness=2)
```
## cv2.resize()
```python
img_resized = cv2.resize(img, dsize=(640, 480), interpolation=cv2.INTER_AREA)
```
## cv2.cvtColor()
```python
img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
```
## cv2.imread()
```python
img = cv2.imread("300.jpg")
```
## cv2.imshow()
```python
cv2.imshow("img_resized", img_resized)
```
## cv2.findContours()
```python
mask = cv2.inRange(hsv,lower_blue,upper_blue)
contours, hierachy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
```
## cv2.TERM_CRITERIA_EPS, cv2.TERM_CRITERIA_MAX_ITER
```python
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
```
-  Define criteria = (type, max_iter = 10, epsilon = 1.0)
## CV2.KMEANS_RANDOM_CENTERS
```python
flags = cv2.KMEANS_RANDOM_CENTERS
```
- 초기 중심점을 랜덤으로 설정.
```python
compactness, labels, centers = cv2.kmeans(z, 3, None, criteria, 10, flags)
```
- KMeans를 적용. k=2,  10번 반복한다.

# datetime
```python
import datetime
```
## datetime.datetime
```python
datetime.datetime(2018, 5, 19)
```
- \>\>\> datetime.datetime(2018, 5, 19, 0, 0)
### datetime.datetime.now()
```python
datetime.datetime.now()
```
- \>\>\> datetime.datetime(2020, 8, 18, 21, 44, 20, 835233)
## timestamp()
```python
datetime.datetime.now().timestamp()
```
- 1970년 1월 1일 0시 0분 0초로부터 몇 초가 지났는지 출력.


# gensim
```python
import gensim
```
## gensim.corpora
### gensim.corpora.Dictionary()
```python
id2word = gensim.corpora.Dictionary(docs_tkn)
```
#### id2word.id2token
- dict(id2word)와 dict(id2word.id2token)은 서로 동일.
#### id2word.token2id
- dict(id2word.token2id)는 key와 value가 서로 반대.
#### id2word.doc2bow()
```python
dtm = [id2word.doc2bow(doc) for doc in docs_tkn]
```
#### gensim.corpora.Dictionary.load()
```python
id2word = gensim.corpora.Dictionary.load("kakaotalk id2word")
```
### gensim.corpora.BleiCorpus
#### gensim.corpora.BleiCorpus.serizalize()
```python
gensim.corpora.BleiCorpus.serialize("kakotalk dtm", dtm)
```
### gensim.corpora.bleicorpus
#### gensim.corpora.bleicorpus.BleiCorpus()
```python
dtm = gensim.corpora.bleicorpus.BleiCorpus("kakaotalk dtm")
```
## gensim.models
### gensim.models.TfidfModel()
```python
tfidf = gensim.models.TfidfModel(dtm)[dtm]
```
### gensim.models.AuthorTopicModel()
```python
model = gensim.models.AuthorTopicModel(corpus=dtm, id2word=id2word, num_topics=n_topics, author2doc=aut2doc, passes=1000)
```
### gensim.models.Word2Vec()
```python
model = gensim.models.Word2Vec(sentences, min_count=5, size=300, sg=1, iter=10, workers=4, ns_exponent=0.75, window=7)
```
### gensim.models.FastText()
```python
model = gensim.models.FastText(sentences, min_count=5, sg=1, size=300, workers=4, min_n=2, max_n=7, alpha=0.05, iter=10, window=7)
```
#### model.save()
```python
model.save("kakaotalk model")
```
#### model.show_topic()
```python
model.show_topic(1, topn=20)
```
#### model.wv.most_similar()
```python
model.wv.most_similar("good)
```
#### gensim.models.AuthorTopicModel.load()
```python
model = gensim.models.AuthorTopicModel.load("kakaotalk model")
```
## gensim.models.ldamodel.Ldamodel()
```python
model = gensim.models.ldamodel.LdaModel(dtm, num_topics=n_topics, id2word=id2word, alpha="auto", eta="auto")
```
### model.show_topic()
```python
model.show_topic(2, 10)
```
- arguments : topic의 index, 출력할 word 개수

# graphviz
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
# gym
```python
import gym
```
## gym.make()
```python
env = gym.make("Taxi-v1")
```
### env.reset()
```python
observ = env.reset()
```
### env.render()
### env.action_space.sample()
```python
action = env.action_space.sample()
```
### env.step()
```python
observ, reward, done, info = env.step(action)
```
```python
env = gym.make("Taxi-v1")
observ = env.reset()
for _ in range(1000):
    env.render()
    action = env.action_space.sample()
    observ, reward, done, info = env.step(action)
```
# io
## BytesIO
```python
from io import BytesIO
```
```python
url = "https://pai-datasets.s3.ap-northeast-2.amazonaws.com/recommender_systems/movielens/img/POSTER_20M_FULL/{}.jpg".format(movie_id)
req = requests.get(url)
b = BytesIO(req.content)
img = np.asarray(Image.open(b))
```
# IPython
## IPython.display
### set_matplotlib_formats
```python
from IPython.display import set_matplotlib_formats
```
```python
set_matplotlib_formats("retina")
```
- font를 선명하게 표시
# itertools
## combinations()
```python
from itertools import combinations
```
```python
movies = {a | b for a, b in combinations(movie2sup.keys(), 2)}
```
# konlpy
## Okt()
```python
from konlpy.tag import Okt
```
```python
okt = Okt()
```
### okt.add_dictionary
```python
okt.add_dictionary(["대금", "지급", "근재", "사배책", "건설", "기계"], "Noun")
```
### okt.morphs()
```python
codes_tokenized = [" ".join(okt.morphs(code)) for code in codes]
items_tokenized = [" ".join(okt.morphs(item)) for item in items]
```
# mapboxgl
## mapboxgl.viz
```python
from mapboxgl.viz import *
```
### CircleViz()
```python
viz = CircleViz(data=geo_data, access_token=token, center=[127.46,36.65], zoom=11, radius=2, stroke_color="black", stroke_width=0.5)
```
### GraduatedCircleViz()
```python
viz = GraduatedCircleViz(data=geo_data, access_token=token, height="600px", width="600px", center=(127.45, 36.62), zoom=11, scale=True, legend_gradient=True, add_snapshot_links=True, radius_default=4, color_default="black", stroke_color="black", stroke_width=1, opacity=0.7)
```
### viz.style
```python
viz.style = "mapbox://styles/mapbox/outdoors-v11"
```
- "mapbox://styles/mapbox/streets-v11"
- "mapbox://styles/mapbox/outdoors-v11"
- "mapbox://styles/mapbox/light-v10"
- "mapbox://styles/mapbox/dark-v10"
- "mapbox://styles/mapbox/satellite-v9"
- "mapbox://styles/mapbox/satellite-streets-v11"
- "mapbox://styles/mapbox/navigation-preview-day-v4"
- "mapbox://styles/mapbox/navigation-preview-night-v4"
- "mapbox://styles/mapbox/navigation-guidance-day-v4"
- "mapbox://styles/mapbox/navigation-guidance-night-v4"
### viz.show()
```python
viz.show()
```
### viz.create_html()
```python
with open("D:/☆디지털혁신팀/☆실거래가 분석/☆그래프/1km_store.html", "w") as f:
    f.write(viz.create_html())
```
## mapboxgl.utils
```python
from mapboxgl.utils import df_to_geojson, create_color_stops, create_radius_stops
```
### df.to_geojson()
```python
geo_data = df_to_geojson(df=df, lat="lat", lon="lon")
```
### viz.create_color_stops()
```python
viz.color_property = "error"
viz.color_stops = create_color_stops([0, 10, 20, 30, 40, 50], colors="RdYlBu")
```
### viz.create_radius_stops()
```python
viz.radius_property = "errorl"
viz.radius_stops = create_radius_stops([0, 1, 2], 4, 7)
```
# matplotlib
```python
import matplotlib as mpl
```
## mpl.font_manager.FontProperties().get_name()
```python
path = "C:/Windows/Fonts/malgun.ttf"
font_name = mpl.font_manager.FontProperties(fname=path).get_name()
```
## mpl.rc()
```python
mpl.rc("font", family=font_name)
```
```python
mpl.rc("axes", unicode_minus=False)
```
## matplotlib.pyplot
```python
import matplotlib.pyplot as plt
```
### plt.subplots()
```python
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12)
```
### plt.setp()
```python
plt.setp(obj=ax1, yticks=ml_mean_gr_ax1["le"], yticklabels=ml_mean_gr_ax1.index)
```
### fig.colorbar()
```python
cbar = fig.colorbar(ax=ax, mappable=scatter)
```
### plt.style.use()
```python
plt.style.use("dark_background")
```
### plt.imshow()
```python
plt.imshow(image.numpy().reshape(3,3), cmap="Greys")
```
### fig.savefig()
```python
fig.savefig("means_plot_200803.png", bbox_inches="tight")
```
### fig.tight_layout()

### cbar.set_label()
```python
cbar.set_label(label="전용면적(m²)", size=15)
```
### ax.set()
```python
ax.set(title="Example", xlabel="xAxis", ylabel="yAxis", xlim=[0, 1], ylim=[-0.5, 2.5], xticks=data.index, yticks=[1, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3])
```
### ax.set_title()
```python
ax.set_title("Example", size=20)
```
### ax.set_xlabel(), ax.set_ylabel()
```python
ax.set_xlabel("xAxis", size=15)
```
### ax.set_xlim(), ax.set_ylim()
```python
ax.set_xlim([1, 4])
```
### ax.axis()
```python
ax.axis([2, 3, 4, 10])
```
### ax.xaxis, ax.yaxis
#### ax.xaxis.set_ticks_position(), ax.yaxis.set_ticks_position()
```python
ax1.yaxis.set_ticks_position("right")
```
### ax.invert_xaxis(), ax.invert_yaxis()
#### ax.xaxis.set_tick_position(), ax.yaxis.set_tick_position()
```python
ax2.yaxis.set_ticks_position("right")
```
#### ax.xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter(), ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter()
```python
ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter("{x:,.0f}"))
```
### ax.set_xticks(), ax.set_yticks()
```python
ax.set_xticks([1, 2])
```
### ax.tick_params()
```python
ax.tick_params(axis="x", labelsize=20, labelcolor="red", labelrotation=45, grid_linewidth=3)
```
### ax.legend()
```python
ax.legend(fontsize=14, loc="best")
```
### ax.grid()
```python
ax.grid(axis="x", color="White", alpha=0.3, linestyle="--", linewidth=2)
```
### ax.plot()
### ax.scatter()
```python
ax.scatter(x=gby["0.5km 내 교육기관 개수"], y=gby["실거래가"], s=70, c=gby["전용면적(m²)"], cmap="RdYlBu", alpha=0.7, edgecolors="black", linewidth=0.5)
```
### ax.bar()
```python
ax.bar(x=nby_genre.index, height=nby_genre["movie_id"])
```
### ax.barh()
```python
ax.barh(y=ipark["index"], width=ipark["가경4단지 84.8743A"], height=0.2, alpha=0.5, color="red", label="가경4단지 84.8743A", edgecolor="black", linewidth=1)
```
### ax.hist()
```python
ax.hist(cnt_genre["genre"], bins=30)
```
### ax.axhline()
```python
ax.axhline(y=mean, color="r", linestyle=":", linewidth=2)
```
### ax.text()
```python
for _, row in ml_gby_ax1.iterrows():
    ax1.text(y=row["le"]-0.18, x=row["abs_error"], s=round(row["abs_error"], 1), va="center", ha="left", fontsize=10)
```
### plot(kind="pie")
```python
cnt_genre.sort_values("movie_id", ascending=False)["movie_id"].plot(ax=ax, kind="pie", startangle=90, legend=True)
```

# seaborn
```python
import seaborn as sb
```
### sb.scatterplot()
```python
sb.scatterplot(ax=ax, data=df, x="ppa", y="error", hue="id", hue_norm=(20000, 20040), palette="RdYlGn", s=70, alpha=0.5)
```
### sb.lineplot()
```python
ax = sb.lineplot(x=data.index, y=data["ppa_ratio"], linewidth=3, color="red", label="흥덕구+서원구 아파트 평균")
ax = sb.lineplot(x=data.index, y=data["84A"], linewidth=2, color="green", label="가경아이파크 4단지 84A")
ax = sb.lineplot(x=data.index, y=data["84B"], linewidth=2, color="blue", label="가경아이파크 4단지 84B")
```
### sb.barplot()
```python
sb.barplot(ax=ax, x=area_df["ft_cut"], y=area_df[0], color="brown", edgecolor="black", orient="v")
```
### sb.replot()
```python
ax = sbrelplot(x="total_bill", y="tip", col="time", hue="day", style="day", kind="scatter", data=tips)
```
### sb.kedplot()
```python
sb.kdeplot(ax=ax, data=data["ppa_root"])
```
### sb.stripplot()
```python
ax = sb.stripplot(x=xx, y=yy, data=results_order, jitter=0.4, edgecolor="gray", size=4)
```
### sb.pairtplot()
```python
ax = sb.pairplot(data_for_corr)
```
### sb.heatmap()
* http://seaborn.pydata.org/generated/seaborn.heatmap.html
```python
sb.heatmap(ax=ax, data=gby_occup_genre, annot=True, annot_kws={"size": 10}, fmt=".2f", linewidths=0.2, center=3, cmap="RdBu")
```
# MeCab
```python
import MeCab
```
```python
def pos(text):
    p = re.compile(".+\t[A-Z]+")
    return [tuple(p.match(line).group().split("\t")) for line in MeCab.Tagger().parse(text).splitlines()[:-1]]

def morphs(text):
    p = re.compile(".+\t[A-Z]+")
    return [p.match(line).group().split("\t")[0] for line in MeCab.Tagger().parse(text).splitlines()[:-1]]

def nouns(text):
    p = re.compile(".+\t[A-Z]+")
    temp = [tuple(p.match(line).group().split("\t")) for line in MeCab.Tagger().parse(text).splitlines()[:-1]]
    nouns=[]
    for word in temp:
        if word[1] in ["NNG", "NNP", "NNB", "NNBC", "NP", "NR"]:
            nouns.append(word[0])
    return nouns

def cln(text):
    return re.sub("[^ㄱ-ㅣ가-힣 ]", "", text)

def def_sw(path):
    sw = set()
    for i in string.punctuation:
        sw.add(i)
    with open(path, encoding="utf-8") as f:
        for word in f:
            sw.add(word.split("\n")[0])
    return sw
```
```python
train_data = []
for line in tqdm(train_docs):
    review = line[1]
    label = line[2]
    review_tkn = nouns(cln(review))
    review_tkn = [word for word in review_tkn if (word not in sw)]
    train_data.append((review_tkn, label))
```
# wordcloud
## WordCloud
```python
from wordcloud import WordCloud
```
```python
wc = WordCloud(font_path="C:/Windows/Fonts/HMKMRHD.TTF", relative_scaling=0.2, background_color="white", width=1600, height=1600, max_words=30000, mask=mask, max_font_size=80, background_color="white")
```
### wc.generate_from_frequencies()
```python
wc.generate_from_frequencies(words)
```
### wc.generate_from_text
### wc.recolor()
```python
wc.recolor(color_func=img_colors)
```
### wc.to_file()
```python
wc.to_file("test2.png")
```
## ImageColorGenerator
```python
from wordcloud import ImageColorGenerator
```
```python
img_arr = np.array(Image.open(pic))
img_colors = ImageColorGenerator(img_arr)
img_colors.default_color=[0.6, 0.6, 0.6]
```
## STOPWORDS
```python
from wordcloud import STOPWORDS
```

# mlxtend
```python
import mlxtend
```
## mlxtend.preprocessing
### TransactionEncoder
```python
from mlxtend.preprocessing import TransactionEncoder
```
```python
te = TransactionEncoder()
```
#### te.fit_transform()
```python
baskets_te = te.fit_transform(baskets)
```
#### te.columns_
```python
baskets_df = pd.DataFrame(baskets_te, index=baskets.index, columns=te.columns_)
```
## mlxtend.frequent_patterns
### apriori
```python
from mlxtend.frequent_patterns import apriori
```
```python
freq_sets_df = apriori(baskets_df_over5000.sample(frac=0.05), min_support=0.01, max_len=2, use_colnames=True, verbose=1)
```
### association_rules
```python
from mlxtend.frequent_patterns import association_rules
```
```python
asso_rules = association_rules(sups, metric="support", min_threshold=0.01)
```
# nltk
```python
import nltk
```
## nltk.Text()
```python
text = nltk.Text(total_tokens, name="NMSC")
```
### text.tokens
### text.vocab() : returns frequency distribution
#### text.vocab().most_common()
```python
text.vocab().most_common(10)
```
### text.plot()
```python
text.plot(50)
```
## nltk.download()
```python
nltk.download("movie_reviews")
```
```python
nltk.download("punkt")
```
## nltk.corpus
```python
from nltk.corpus import movie_reviews
```
### movie_reviews
#### movie_reviews.sents()
```python
sentences = [sent for sent in movie_reviews.sents()]
```
# numpy
```python
import numpy as np
```
## np.set_printoptions()
```python
np.set_printoptions(precision=3)
```
## np.arange
```python
np.arange(5, 101, 5)
```
## np.ones()
```python
np.ones((2, 3, 4))
```
## np.zeros()
## np.empty()
## np.full()
```python
np.full((2, 3, 4), 7))
```
## np.eye()
```python
np.eye(4)
```
## np.ones_like(), np.zeros_like()
```python
np.ones_like(arr)
```
## np.linspace()
```python
np.linspace(-5, 5, 100)
```
## np.any()
```python
np.any(arr>0)
```
## np.all()
## np.where()
```python
np.where(arr>0, arr, 0)
```
## np.isin()
```python
data[np.isin(data["houses"], list)]
```
## np.transpose()
## np.swapaxes()
```python
feature_maps = np.transpose(conv2d, (3, 1, 2, 0))
```
```python
feature_maps = np.swapaxes(conv2d, 0, 3)
```
## np.maximum(), np.minimum()
- Element-wise minimum of array elements.
## np.concatenate()
```python
intersected_movie_ids = np.concatenate([json.loads(row) for row in rd.mget(queries)], axis=None)
```
## np.random
### np.random.seed()
```python
np.random.seed(23)
```
### np.random.rand()
```python
np.random.rand(2, 3, 4)
```
### np.random.randint()
```python
np.random.randint(1, 100, size=(2, 3, 4))
```
### np.random.choice()
```python
np.random.choice(arr(1d), size=(2, 3, 4), replace=False)
```
### np.random.normal()
```python
np.random.normal(mean, std, size=(3, 4))
```
## np.digitize()
```python
bins=range(0, 55000, 5000)
data["price_range"]=np.digitize(data["money"], bins)
```
## np.isnan()
## np.nanmean()
## np.sort()
## np.reshape()
```python
np.reshape(mh_df.values, (1000, 1, 128))
```
## np.expand_dims()
```python
np.expand_dims(mh_df.values, axis=1)
```
## np.newaxis
```python
mh_df.values[:, np.newaxis, :]
```
```python
mh_df.values[:, None, :]
```
## np.unique()
```python
items, counts = np.unique(intersected_movie_ids, return_counts=True)   
```
## np.linalg
### np.linalg.norm()
```python
np.linalg.norm(x, axis=1, ord=2)
```
- ord=1 : L1 normalization.
- ord=2 : L2 normalization.
## np.sqrt()
## np.power()
## np.exp()
```python
def sig(x):
    return 1 / (1 + np.exp(-x))
```
## np.add.outer(), np.multiply.outer()
```python
euc_sim_item = 1 / (1 + np.sqrt(np.add.outer(square, square) - 2*dot))
```
## np.fill_diagonal()
```python
np.fill_diagonal(cos_sim_item, 0)
```
# arr
## arr.ravel()
```python
arr.ravel(order="F")
```
- order="C" : row 기준
- order="F" : column 기준
##  arr.flatten()
- 복사본 반환
## arr.T
## arr.shape
# os
```python
import os
```
## os.getcwd()
```python
cur_dir = os.getcwd()
```
## os.makedirs()
```python
os.makedirs(ckpt_dir, exist_ok=True)
```
## os.path
### os.path.join()
```python
checkpoint_dir = os.path.join(cur_dir, ckpt_dir_name, model_dir_name)
```
### os.path.exists()
```python
if os.path.exists("C:/Users/5CG7092POZ/train_data.json"):
```
## os.environ[], os.pathsep
```python
os.environ["PATH"] += os.pathsep + "C:\Program Files (x86)/Graphviz2.38/bin/"
```
# pickle
```python
import pickle as pk
```
## pk.dump()
```python
with open("filename.pk", "wb") as f:
    pk.dump(list, f)
```
## pk.load()
```python
with open("filename.pk", "rb") as f:
    data = pk.load(f)
```
- 한 줄씩 load
# json
```python
import json
```
## json.dump()
```python
with open(path, "w", encoding="utf-8") as f:
    json.dump(train_data, f, ensure_ascii=False, indent="\t")
```
## json.load()
```python
with open(path, "r", encoding="utf-8") as f:
    train_data = json.load(f)
```
# Image
```python
from PIL import Image
```
## Image.open()
```python
img = Image.open("20180312000053_0640 (2).jpg")
```
### img.size
### img.save()
### img.thumbnail()
```python
img.thumbnail((64, 64))  
```
### img.crop()
```python
img_crop = img.crop((100, 100, 150, 150))
```
### img.resize()
```python
img = img.resize((600, 600))
```
### img.convert()
```python
img.convert("L")
```
- "RGB" | "RGBA" | "CMYK" | "L" | "1"
### img.paste()
```python
img1.paste(img2, (20,20,220,220))
```
- img2.size와 동일하게 두 번째 parameter 설정.
## Image.new()
```python
mask = Image.new("RGB", icon.size, (255, 255, 255))
```
# platform
```python
import platform
```
## platform.system()
```python
path = "C:/Windows/Fonts/malgun.ttf"
if platform.system() == "Darwin":
    mpl.rc("font", family="AppleGothic")
elif platform.system() == "Windows":
    font_name = mpl.font_manager.FontProperties(fname=path).get_name()
    mpl.rc('font', family=font_name)
```
- "Darwin", "Windows" 등 OS의 이름 반환.
# pprint
## pprint()
```python
from pprint import pprint
```
# pptx
## Presentation()
```python
from pptx import Presentation
```
```python
prs = Presentation("sample.pptx")
```
### prs.slides[].shapes[].text_frame.paragraphs[].text
```python
prs.slides[a].shapes[b].text_frame.paragraphs[c].text
```
- a : 슬라이드 번호
- b : 텍스트 상자의 인덱스
- c : 텍스트 상자 안에서 텍스트의 인덱스
### prs.slides[].shapes[].text_frame.paragraphs[].font
#### prs.slides[].shapes[].text_frame.paragraphs[].text.name
```python
prs.slides[a].shapes[b].text_frame.paragraphs[c].font.name = "Arial"
```
#### prs.slides[].shapes[].text_frame.paragraphs[].text.size
```python
prs.slides[a].shapes[b].text_frame.paragraphs[c].font.size = Pt(16)
```
### prs.save()
```python
prs.save("파일 이름")
```
# pygame
```python
import pygame
```
## gamepad
### gamepad.blit()
```python
def bg(bg, x, y):
    global gamepad, background
    gamepad.blit(bg, (x, y))
```
```python
def plane(x, y):
    global gamepad, aircraft
    gamepad.blit(aircraft, (x, y)) #"aircraft"를 "(x, y)"에 배치
```
## pygame
### pygame.event.get(), event.type, event.key
```python
while not crashed:
    for event in pygame.event.get():
        if event.type==pygame.QUIT: #마우스로 창을 닫으면
            crashed=True #게임 종료

        if event.type==pygame.KEYDOWN: #키보드를 누를 때
            if event.key==pygame.K_RIGHT:
                x_change=15
```
### pygame.KEYUP, pygame.KEYDOWN
```python
if event.type==pygame.KEYUP
```
- pygame.KEYUP : 키보드를 누른 후 뗄 때
- pygame.KEYDOWN : 키보드를 누를 때
### pygame.init()
### pygame.quit()
### pygame.display.set_model()
```python
gamepad = pygame.display.set_mode((pad_width, pad_height))
```
```python
import pygame

WHITE=(255, 255, 255)
pad_width=1536
pad_height=960
bg_width=2560

def bg(bg, x, y):
    global gamepad, background
    gamepad.blit(bg, (x, y))

def plane(x, y):
    global gamepad, aircraft
    gamepad.blit(aircraft, (x, y)) #"aircraft"를 "(x, y)"에 배치

def runGame():
    global gamepad, aircraft, clock, bg1, bg2
    
    x=pad_width*0.01
    y=pad_height*0.5
    x_change=0
    y_change=0
    
    bg1_x=0
    bg2_x=bg_width
    
    crashed=False #"True" : 게임 종료,  "False" : 안 종료
    while not crashed:
        for event in pygame.event.get():
            if event.type==pygame.QUIT: #마우스로 창을 닫으면
                crashed=True #게임 종료
                
            if event.type==pygame.KEYDOWN: #키보드를 누를 때
                if event.key==pygame.K_RIGHT:
                    x_change=15
                if event.key==pygame.K_LEFT:
                    x_change=-15
                elif event.key==pygame.K_UP:
                    y_change=-15
                elif event.key==pygame.K_DOWN:
                    y_change=15
            if event.type==pygame.KEYUP: #키보드를 누른 후 뗄 때
                if event.key==pygame.K_RIGHT or event.key==pygame.K_LEFT:
                    x_change=0
                if event.key==pygame.K_UP or event.key==pygame.K_DOWN:
                    y_change=0
        x+=x_change
        y+=y_change
                    
        gamepad.fill(WHITE) #"gamepad"를 "WHITE"로 채우고
        bg1_x-=5
        bg2_x-=5
        if bg1_x==-bg_width:
            bg1_x=bg_width
        if bg2_x==-bg_width:
            bg2_x=bg_width
            
        bg(bg1, bg1_x, 0)
        bg(bg2, bg2_x, 0)
        
        plane(x, y) #"plane(x, y)" 함수를 실행한 뒤
        pygame.display.update() #화면 갱신
        clock.tick(60) #"tick=60"으로 FPS=60 지정
        
    pygame.quit() #종료
    
def initGame():
    global gamepad, aircraft, clock, bg1, bg2
    
    pygame.init()
    gamepad=pygame.display.set_mode((pad_width, pad_height)) #화면 크기 지정
    pygame.display.set_caption("PyFlying") #타이틀 지정
    aircraft=pygame.image.load("pngguru.com (4).png")
    bg1=pygame.image.load("background.jpg")
    bg2=bg1.copy()
    
    clock=pygame.time.Clock() #FPS를 지정하기 위한 변수 "clock" 선언
    runGame()
    
initGame()
```
# pyLDAvis
```python
import pyLDAvis
```
## pyLDAvis.enable_notebook()
- pyLDAvis를 jupyter notebook에서 실행할 수 있게 활성화.
## pyLDAvis.gensim
```python
import pyLDAvis.gensim
```
### pyLDAvis.gensim.prepare()
```python
pyldavis = pyLDAvis.gensim.prepare(model, dtm, id2word)
```
# pymysql
```python
import pymysql
```
## pymysql.connect()
```python
connect = pymysql.connect(host="localhost", user="root", password="1453", db="masterdata")
```
### pd.read_sql()
```python
pd.read_sql("show tables", connect)
```
# random
```python
import random
```
## random.seed()
## random.sample()
```python
names = random.sample(list(set(data.index)), 20)
```
## random.shuffle()
- inplace function
# re
```python
 import re
```
## re.search()
## re.match()
- re.search()와 유사하나 주어진 문자열의 맨 처음과 대응할 때만 object를 반환.
## re.findall()
- re.search()와 유사하나 대응하는 모든 문자열을 list로 반환.
### re.search().group(), re.match().group()
```python
re.search(r"(\w+)@(.+)", "test@gmail.com").group(0) #test@gmail.com
re.search(r"(\w+)@(.+)", "test@gmail.com").group(1) #test
re.search(r"(\w+)@(.+)", "test@gmail.com").group(2) #gmail.com
```
## re.sub()
```python
re.sub(r"\w+@\w+.\w+", "email address", "test@gmail.com and test2@gmail.com", count=1)
```
- count=0 : 전체 치환
## re.compile()
```python
p = re.compile(".+\t[A-Z]+")
```
- 이후 p.search(), p.match() 등의 형태로 사용.

## regular expressions
### . : newline을 제외한 어떤 character
### \w,   [a-zA-Z0-9_]: 어떤 알파벳 소문자 또는 알파벳 대문자 또는 숫자 또는 _
### \W, [^a-zA-Z0-9_] : 어떤 알파벳 소문자 또는 알파벳 대문자 또는 숫자 또는 _가 아닌
### \d, [0-9] : 어떤 숫자
### \D, [^0-9] : 어떤 숫자가 아닌
### \s : 공백
### \S : 공백이 아닌 어떤 character
### \t : tab
### \n : newline
### \r : return
### \가 붙으면 문자 그 자체를 의미.
### [] : [] 안의 문자를 1개 이상 포함하는
- [] 내부의 문자는 해당 문자 자체를 나타냄.
#### [abc]
- "a"는 정규식과 일치하는 문자인 "a"가 있으므로 매치
- "before"는 정규식과 일치하는 문자인 "b"가 있으므로 매치
- "dude"는 정규식과 일치하는 문자인 a, b, c 중 어느 하나도 포함하고 있지 않으므로 매치되지 않음
### * : 0개~무한대의 바로 앞의 character
### + : 1개~무한대의 바로 앞의 character
### ? : 0개~1개의 바로 앞의 character
### {m,n} : m개~n개의 바로 앞의 character
- 생략된 m은 0과 동일, 생략된 n은 무한대와 동일
- *?, +?, {m,n}? : non-greedy way
### {n} : n개의 바로 앞의 character
### ? : 0개~1개의 바로 앞의 character
### ^ : 바로 뒤의 문자열로 시작하는
### [^] : 바로 뒤의 문자열을 제외한
### $ : 바로 앞의 문자열로 끝나는
# requests
```python
import requests
```
## requests.get()
```python
req = requests.get(url)
```
# statsmodels
## variance_inflation_factor
```python
from statsmodels.stats.outliers_influence import variance_inflation_factor
```
```python
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(data_for_corr.values, i) for i in range(data_for_corr.shape[1])]
vif["features"] = data_for_corr.columns
```
# string
```python
import string
```
## string.punctuation
- \>\>\> '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
# sys
```python
import sys
```
## sys.maxsize()
# time
```python
import time
```
## time.time()
```python
time_before=round(time.time())
...
print("{}초 경과".format(round(time.time())-time_before))
```
# tqdm
```python
from tqdm.notebook import tqdm
```
## tqdm.pandas()
# warnings
```python
import warnings
```
## warnings.filterwarnings()
```python
warnings.filterwarnings("ignore", category=DeprecationWarning)
```
# scipy
```python
import scipy
```
## scipy.sparse
### csr_matrix
```python
from scipy.sparse import csr_matrix
```
```python
vals = [2, 4, 3, 4, 1, 1, 2]
rows = [0, 1, 2, 2, 3, 4, 4]
cols = [0, 2, 5, 6, 14, 0, 1]

sparse_matrix = csr_matrix((vals,  (rows,  cols)))
```
#### sparse_mat.todense()
## stats
```python
from scipy import stats
```
### stats.beta
#### stats.beta.pdf()
```python
ys = stats.beta.pdf(xs, a, b)
```
# implicit
```python
conda install -c conda-forge implicit
```
## implicit.bpr
### BayesianPersonalizedRanking
```python
from implicit.bpr import BayesianPersonalizedRanking as BPR
```
```python
model = BPR(factors=60)

model.fit(inputs)
```
#### model.user_factors, model.item_factors
```python
user_embs = model.user_factors
item_embs = model.item_factors
```
# annoy
- https://github.com/spotify/annoy
```python
!pip install "C:\Users\5CG7092POZ\annoy-1.16.3-cp37-cp37m-win_amd64.whl"
```
- source : https://www.lfd.uci.edu/~gohlke/pythonlibs/#annoy
## AnnoyIndex
```python
from annoy import AnnoyIndex
```
```python
n_facts = 61
tree = AnnoyIndex(n_facts, "dot")
```
### tree.add_item()
```python
for idx, value in enumerate(art_embs_df.values):
    tree.add_item(idx, value)
```
### tree.build()
```python
tree.build(20)
```
### tree.get_nns_by_vector()
```python
print([art_id2name[art] for art in tree.get_nns_by_vector(user_embs_df.loc[user_id], 10)])
```
# openpyxl
## df.to_excel()
# google_drive_downloader
```python
!pip install googledrivedownloader
```
## GoogleDriveDownloader
```python
from google_drive_downloader import GoogleDriveDownloader as gdd
```
### gdd.download_file_from_google_drive()
```python
gdd.download_file_from_google_drive(file_id="1uPjBuhv96mJP9oFi-KNyVzNkSlS7U2xY", dest_path="./movies.csv")
```
# datasketch
## MinHash
```python
from datasketch import MinHash
```
```python
mh = MinHash(num_perm=128)
```
- MinHash는 각 원소 별로 signature를 구한 후, 각 Signature 중 가장 작은 값을 저장하는 방식입니다. 가장 작은 값을 저장한다 해서 MinHash라고 불립니다.
### mh.update()
```python
for value in set_A:
    mh.update(value.encode("utf-8"))
```
### mh.hashvalues
# redis
## Redis
```python
from redis import Redis
```
```python
rd = Redis(host="localhost", port=6379, db=0)
```
### rd.set()
```python
rd.set("A", 1)
```
### rd.delete()
```python
rd.delete("A")
```
### rd.get()
```python
rd.get("A")
```
