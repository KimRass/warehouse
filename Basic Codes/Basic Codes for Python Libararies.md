# Python Built-in Functions
## TypeError
## StopIteration
## hash()
## input()
```python
input("숫자 입력  : ")
```
```python
list(map(int, input("숫자 입력  : ").split()))
```
## print()
```python
print(..., end="")
```
- `sep="\n"`
## isinstance()
```python
if not isinstance(movie, frozenset):
    movie = frozenset(movie)
```
## type()
```python
type(test_X[0][0])
```
## sum()
```python
sum(sentences, [])
```
- 두번째 층의 대괄호 제거
## assert
```python
assert model_name in self.model_list, "There is no such a model."
```
## var
### var.data
#### var.data.nbytes : 변수에 할당된 메모리 크기 리턴
## lst
### lst.index()
```python
names.index((17228, "아트빌"))
```
### lst.append()
```python
feature_to_shuffle.append("area")
```
### lst.insert()
- idx, value 순으로 arg를 입력합니다.
### lst.remove()
```python
features.remove("area")
```
### lst.count()
```python
[2, 4012, 3394, 3, 1, 1].count(1)
```
### sorted()
```python
lens = sorted([len(doc) for doc in train_X])
```
- `reverse=True`로 순서를 뒤집을 수 있습니다.
- `key=`로 정렬 기준이 될 함수를 지정할 수 있습니다.
### map()
```python
list(map(len, train_tkn))
```
```python
x_data = list(map(lambda word : [char2idx.get(char) for char in word], words))
```
- Iterator.
### filter()
### sum()
```python
sum(sents, [])
```
### list comprehension
```python
chars = set([char for word in words for char in word])
```
## set
### set1 & set2
### set1 | set2
### set.add()
- 1개의 element만 추가할 수 있습니다.
### set.update()
- lst.append()와 동일합니다.
- 여러 개의 element를 추가할 수 있습니다.
## frozenset()
- 구성 요소들이 순서대로 들어 있지 않아 인덱스를 사용한 연산을 할 수 없고
- 유일한 항목 1개만 들어가 있습니다.
- set 객체는 구성 항목들을 변경할 수 있지만 frozenset 객체는 변경할 수 없는 특성을 지니고 있어, set에서는 사용할 수 있는 메소드들 add, clear, discard, pop, remove, update 등은 사용할 수 없습니다. 
- 항목이 객체에 포함되어 있는지(membership)를 확인할 수 있으며,  항목이 몇개인지는 알아낼 수 있습니다.
- frozenset은 dictionary의 key가 될 수 있지만 set은 될 수 없음.
## dictionary
### dic\[]
- key를 입력받아 value를 반환합니다.
### dic.get()
- `dic[]`과 동일합니다.
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
### dic.keys(), dic.values()
### dic.fromkeys(list or tuple, value)
### sorted()
```python
word2cnt = dict(sorted(tkn.word_counts.items(), key=lambda x:x[1], reverse=True))
```
### dictionary comprehension
```python
{idx:char for idx, char in enumerate(char_set)}
```
## str
### str.isalpha()
### str.format()
```python
print("[{0:>4d}], [{1:>20d}]".format(100, 200))
```
```python
print("[{0:<20s}]".format("string"))
```
```python
print("[{0:<20.20f}], [{1:>10.2f}]".format(3.14, 10.925))
```
### str.zfill()
```python
hr = pd.read_excel(f"./FINAL/HR/사원명단_{target_year}{str(target_month).zfill(2)}.xlsx")
```
### str.join()
```python
" ".join(["good", "bad", "worse", "so good"])
```
- str을 사이에 두고 리스트의 모든 원소들을 하나로 합침
### str.split()
```python
msg_tkn = [msg.split(" ") for msg in data["msg"]]
```
- makes a list by splitting the string using the argument as a criteria.
### str.strip()
- Removes the begining or the end whitespaces(`" "`|`"\n"`|`"\t"`)
### str.upper(), str.lower()
- argument를 기준으로 str 분할
### str.count()
```python
"저는 과일이 좋아요".count("과일이")
```
### str.find()
- returns the first index of the argument.
### str.startswith()
- returns whether `str` starts with the argument.
## eval()
```python
A = [eval(f"A{i}") for i in range(N, 0, -1)]
```
- "를 제거하는 효과입니다. 단일 변수에 사용합니다.
## exec()
```python
for data in ["tasks", "comments", "projects", "prj_members", "members"]:
    exec(f"{data} = pd.read_csv('D:/디지털혁신팀/협업플랫폼 분석/{data}.csv')")
```
```python
exec(f"{table} = pd.DataFrame(result)")
```
- "를 제거하는 효과입니다. 수식에 사용합니다.
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
```
- `INSTANCE = CLASS(parameter1, parameter2, ...)` : \_\_init\_\_문은 instance를 initiate할 때 실행
- `INSTANCE(parameter3, parameter4, ...)` : \_\_call\_\_문은 instance를 call할 때 실행
- `INSTANCE.FUNCTION(parameter5, parameter6, ...)`
- method : class 정의문 안에서 정의된 함수
- method의 첫번째 parameter는 항상 self여야 함
- method의 첫 번째 parameter는 self지만 호출할 때는 아무것도 전달하지 않는 이유는 첫 번째 parameter인 self에 대한 값은 파이썬이 자동으로 넘겨주기 때문입니다.
## \_\_iter\_\_()
```python
def __iter__(self):
	return self
```
## \_\_next\_\_()
## StopIteration
### class variables
- class 정의문 안에서 정의된 variables
### instance variables
- self가 붙어 있는 variables
## override
- 출처 : https://rednooby.tistory.com/55
## super()
- 출처 : https://rednooby.tistory.com/56?category=633023
```python
class Bahdanau(tf.keras.Model):
    def __init__(self, units):
        super().__init__()
        self.W1 = Dense(units, use_bias=False)
        self.W2 = Dense(units, use_bias=False)
        self.W3 = Dense(1)
        
        # key와 value는 같습니다.
    def call(self, query, keys):
        # query.shape : (batch_size, h_size) --> (batch_size, 1, h_size)
        query = tf.expand_dims(query, axis=1)

        # att_scores.shape : (batch_size, max_len, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.W3 : (batch_size, max_len, units)
        att_scores = self.W3(tf.nn.tanh(self.W1(query) + self.W2(keys)))

        # att_weights.shape : (batch_size, max_len, 1)
        att_weights = tf.nn.softmax(att_scores, axis=1)

        # context_vector.shape : (batch_size, h_size)
        context_vector = att_weights*keys
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, att_weights
```
## open()
```python
with open("C:/Users/5CG7092POZ/nsmc-master/ratings_train.txt", "r", encoding="utf-8") as f:
    train_docs = [line.split("\t") for line in f.read().splitlines()][1:]
```
## input()
```python
A = list(map(int, in "A를 차례대로 입력 : ").split()))
```
## ord()



# pandas
```python
import pandas as pd
```
## pd.set_option()
```python
pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 500)
```
- 최대 rows 또는 columns의 수 설정
```python
pd.set_option("display.width", 1000)
```
```python
pd.set_option("display.float_format", "{:.3f}".format)
```
```python
pd.set_option("mode.chained_assignment",  None)
```
- SettingWithCopyWarning 무시
## pd.DataFrame()
```python
gby_genre_occup = pd.DataFrame(rows, columns=["genre", "occup", "mean"])
```
- list로부터 생성합니다.
```python
gby_genre = pd.DataFrame({"mean":genre2rating, "n_rating":genre2n_rating}).sort_values(by=["mean"])
```
- ser로부터 생성합니다.
## pd.Series()
```python
idf_ser = pd.Series(idf, index=vocab)
```
## pd.read_csv()
```python
data = pd.read_csv("fra.txt", names=["src", "tar", "CC"], sep="\t")
```
- `thousands=","`
- `float_precision="round_trip"`
- `skiprows`
- `error_bad_lines=False`
## pd.read_excel()
```python
data = pd.read_excel("계약일보_속초2차.xlsx", names=["코드", "현장명", "평형", "동", "호수", "성명", "분양구분", "분양가", "계약금1차 일자", "계약금1차", "계약금2차 일자", "계약금2차", "1회 일자", "1회", "2회 일자", "2회", "3회 일자", "3회", "4회 일자", "4회", "5회 일자", "5회", "6회 일자", "6회", "7회 일자", "7회", "8회 일자", "8회", "잔금 일자", "잔금"])
```
## pd.read_table()
```python
ratings_train = pd.read_table("ratings_train.txt", usecols=["document", "label"])
```
```python
raw_data = pd.read_table("steam.txt", names=["label", "review"])
```
## df.to_csv()
```python
data.to_csv("D:/☆디지털혁신팀/☆실거래가 분석/☆데이터/실거래가 전처리 완료_200928-3.csv")
```
- '
- `index=False`
## pd.concat()
```python
data_without = pd.concat([data_without, data_subset], axis=0)
```
```python
neg_sampling = pd.concat([uses_df]*3)
```
```python
dfs = []
for filename in filenames:
    dfs.append(pd.read_csv(filename))
data = pd.concat(dfs, ignore_index=True)
```
## pd.pivot_table()
```python
pivot = pd.pivot_table(uses_df[["daytime", "weekday", "cnt"]], index="daytime", columns="weekday", values="cnt", aggfunc=np.sum)
```
## pd.melt()


```python

n_post = pd.melt(n_post, id_vars="Date", var_name="Emp", value_name="NPost", ignore_index=False)

```
```python
data = pd.melt(data, id_vars=basic_cols + money_cols, value_name="date")
data = pd.melt(data, id_vars=basic_cols + ["date"], var_name="classification", value_name="money")
```
- pd.pivot_table()의 반대 과정입니다.

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
## pd.Grouper()
```python
n_tasks_month = tasks.groupby(pd.Grouper(key="task_date", freq="M")).size()
```
## pd.MultiIndex
### pd.MultiIndex.from_tuples()
```python
order = pd.MultiIndex.from_tuples((hq, dep) for dep, hq in dep2hq.items())
```
## df.shape
## df.quantile()
```python
top90per = plays_df[plays_df["plays"]>plays_df["plays"].quantile(0.1)]
```
## df.groupby()
```python
df.groupby(["Pclass", "Sex"], as_index=False)
```
- iterable한 tuple((group, 해당 group의 DataFrame))을 반환합니다.
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
## df.append()
```python
df = df.append({"addr1":addr1, "addr2":addr2, "dist":dist}, ignore_index=True)
```
## df.apply()
```python
data["반기"]=data["입찰년월"].apply(lambda x:x[:4]+" 상반기" if int(x[4:])<7 else x[:4]+" 하반기")
data["반기"]=data.apply(lambda x:x["입찰년월"][:4]+" 상반기" if int(x["입찰년월"][4:])<7 else x["입찰년월"][:4]+" 하반기", axis=1)
```
```python
data["1001-작업반장]=data["반기"].apply(lambda x:labor.loc[x, "1001-작업반장])
data["1001-작업반장]=data.apply(lambda x:labor.loc[x["반기"], "1001-작업반장], axis=1)
```
## df.progress_apply()
```python
data["morphs"] = data["review"].progress_apply(mcb.morphs)
```
- `tqdm.pandas()`가 선행되어야합니다.
## df.rename()
```python
data = data.rename({"단지명.1":"name", "세대수":"houses_buildings", "저/최고층":"floors"}, axis=1)
```
## df.reindex()
```python
pivot = pivot.reindex(dep_order)
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
- `name=`
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
### df.columns.drop()
```python
uses_df.columns.drop("cnt")
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
```python
df.loc[~df.index.isin(df.dropna().index)]
```
## df.fillna()
```python
data = data.fillna(method="ffill")
```
- `method="ffill"` : propagate last valid observation forward to next valid backfill
## df.sample(), ser.sample()
```python
baskets_df.sample(frac=0.05)
```
```phthon
set(n_per_movie_unseen.sample(n=100, replace=False, weights=n_per_movie).index)
```
- `random_state=777`
## df.iterrows()
- index of row, row를 차례대로 return합니다.
## df.iteritems()
```python
{k:v for k, v in x_train.iteritems()}
```
- column name, column을 차례대로 return합니다.
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
train[train["token"].str.contains(".", regex=False)]
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



# numpy
```python
import numpy as np
```
## arr.astype()
```python
x_train = x_train.astype("float32")
```
## arr.ravel()
```python
arr.ravel(order="F")	
```
- `order="C"` : row 기준
- `order="F"` : column 기준
##  arr.flatten()
- 복사본 반환
## arr.T
## arr.shape
## arr.transpose(), np.transpose()
```python
conv_weights = np.fromfile(f, dtype=np.float32, count=np.prod(conv_shape)).reshape(conv_shape).transpose((2, 3, 1, 0))
```
## np.inf
## np.set_printoptions()
```python
np.set_printoptions(precision=3)
```
```python
np.set_printoptions(edgeitems=3, infstr="inf", linewidth=75, nanstr="nan", precision=8, suppress=False, threshold=1000, formatter=None)
```
- go back to the default options		
## np.load()
```python
intent_train = np.load("train_text.npy").tolist()
```
## np.logical_and(), np.logical_or()
```python
mask = np.logical_or((pred_bbox[:, 0] > pred_bbox[:, 2]), (pred_bbox[:, 1] > pred_bbox[:, 3]))
```
## np.array_equal()
```python
np.array_equal(arr1, arr2)
```
## np.arange()
```python	
np.arange(5, 101, 5)
```
## np.ones()
```python
np.ones(shape=(2, 3, 4))
```
## np.zeros()
## np.empty()
## np.full()
```python
img_paded = np.full(shape=(tar_height, tar_width, 3), fill_value=128.)
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
## np.meshgrid()
```python
xs = np.linspace(0, output_size-1, output_size)
ys = np.linspace(0, output_size-1, output_size)
x, y = np.meshgrid(xs, ys)
```
## np.any()
```python
np.any(arr>0)
```
## np.all()
## np.where()
```python
np.min(np.where(cumsum >= np.cumsum(cnts)[-1]*ratio))
```
## np.tanh()
```python
temp = np.tanh(np.dot(Wh, h_t) + np.dot(Wx, x_t) + b)
```
## np.shape()
```python
np.shape(hidden_states)
```	
## np.isin()
```python
data[np.isin(data["houses"], list)]
```
## np.prod()
```python
conv_shape = (filters, in_dim, kernel_size, kernel_size)
conv_weights = np.fromfile(f, dtype=np.float32, count=np.product(conv_shape))
```
- `axis`
- Return the product of array elements over a given axis.
## np.argmax()
## np.swapaxes()
```python
feature_maps = np.transpose(conv2d, (3, 1, 2, 0))
```
```python
feature_maps = np.swapaxes(conv2d, 0, 3)
```
## np.max(), np.min()
```python
np.max(axis=-1)
```
## np.maximum(), np.minimum()
- Element-wise minimum(maximum) of array elements.
## np.cumsum()
```python
np.cumsum(cnt)
```
## np.quantile()
```python
lens = sorted([len(doc) for doc in train_X])
ratio = 0.99
max_len = int(np.quantile(lens, ratio))
print(f"가장 긴 문장의 길이는 {np.max(lens)}입니다.")
print(f"길이가 {max_len} 이하인 문장이 전체의 {ratio:.0%}를 차지합니다.")
```
## np.concatenate()
```python
intersected_movie_ids = np.concatenate([json.loads(row) for row in rd.mget(queries)], axis=None)
```
## np.stack()
## np.delete()
```python
idx_drop = [idx for idx, doc in enumerate(X_train) if len(doc) == 0]
X_train = np.delete(X_train, idx_drop, axis=0)
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
- generates samples from the uniform distribution on \[0, 1)
### np.random.random()
```python
np.random.random((2, 3, 4))
```
- generates samples from the uniform distribution on \[0, 1)
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
## np.reshape(), arr.reshape()
```python
bn_weights = np.reshape(bn_weights, newshape=(4, filters))[[1, 0, 2, 3]]
```
```python
bn_weights = bn_weights.reshape((4, filters))[[1, 0, 2, 3]]
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
iou = calculate_iou(best_bbox[None, :4], bboxes_cls[:, :4])
```
- same as `None`
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
np.fill_diagonal(cos_sim_item, 0	
```
## np.fromfile()
- `count`: Number of items to read. `-1` means all items (i.e., the complete file).



# sklearn
```python
from sklearn import *
```
## sklearn.model_selection
```python
from sklearn.model_selection import train_test_split
```
### train_test_split
```python
train_X, val_X, train_y, val_y = train_test_split(train_val_X, train_val_y, train_size=0.8, shuffle=True, random_state=3)
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
#### vect.transform()
- builds DTM
##### vect.transform().toarray()
#### vect.fit_transform()
- `vect.fit()` + `vect.transform()`
##### vect.fit_transform().toarray()
#### vect.vocabulary_
##### vect.vocabulary_.get()
- arg의 idx를 출력합니다.
### TfidfVectorizer()
```python
from sklearn.feature_extraction.text import TfidfVectorizer
```
## sklearn.preprocessing
### LabelEncoder()
```python
from sklearn.preprocessing import LabelEncoder
```
```python
le = LabelEncoder()
```
### StandardScaler()
```python
from sklearn.preprocessing import StandardScaler
```
```python
sc = StandardScaler()
```
#### sc.fit_trainsform()
```python
x = sc.fit_transform(data.drop(["money"], axis=1))
x = pd.DataFrame(x, columns=data.drop(["money"], axis=1).columns)
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
## sklearn.preprocessing
### sklearn.preprocessing.LabelEncoder()
```python
le = LabelEncoder()
```
#### le.fit()
```python
le.fit(data["name_addr"])
```
#### le.trainsform()
```ptyhon
data["id"] = le.transform(data["name_addr"])
```
#### le.fit_trainsform()
#### le.inverse_trainsform()
#### le.classes_
```python
label2idx = dict(zip(le.classes_, set(label_train)))
```
## sklearn.pipeline
### Pipeline()
```python
from sklearn.pipeline import Pipeline
```
```python
model = Pipeline([("vect", CountVectorizer()), ("model", SVC(kernel="poly", degree=8))])
```
- 파이프라인으로 결합된 모형은 원래의 모형이 가지는 fit, predict 메서드를 가지며 각 메서드가 호출되면 그에 따른 적절한 메서드를 파이프라인의 각 객체에 대해서 호출한다. 예를 들어 파이프라인에 대해 fit 메서드를 호출하면 전처리 객체에는 fit_transform이 내부적으로 호출되고 분류 모형에서는 fit 메서드가 호출된다. 파이프라인에 대해 predict 메서드를 호출하면 전처리 객체에는 transform이 내부적으로 호출되고 분류 모형에서는 predict 메서드가 호출된다.
## sklearn.linear_model
```python
from sklearn.linear_model import SGDClassifier
```
## sklearn.svm
### SVC()
```python
from sklearn.svm import SVC
```
```python
SVC(kernel="linear")
```
- `kernel="linear"`
- `kernel="poly"` : gamma, coef0, degree
- `kernel="rbf"` : gamma
- `kernel="sigmoid"` : gomma, coef0
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
- `loss` : The loss function to be used.
    - `loss="hinge" : gives a linear SVM.
    - `loss="log"` : gives logistic regression.
    - `loss="perceptron"` : the linear loss used by the perceptron algorithm.
- `penalty` : regularization term.
    - `penalty="l1"`
    - `penalty="l2"` : the standard regularizer for linear SVM models.
- `"alpha"` : constant that multiplies the regularization term. The higher the value, the stronger the regularization. Also used to compute the learning rate when set to learning_rate is set to ‘optimal’.
- "max_iter" : The maximum number of passes over the training data (aka epochs).
## sklearn.datasets
### sklearn.datasets.fetch_20newsgroups()
```python
newsdata = sklearn.datasets.fetch_20newsgroups(subset="train")
```
- `subset="all"` | `"train"` | `"test"`
### sklearn.datasets.sample_generator
#### make_blobs()
```python
 from sklearn.datasets.sample_generator improt make_blobs
```
## sklearn.metrcis
### sklearn.metrics.pairwise
#### sklearn.metrics.pairwise.cosine_similarity
### sklearn.metrics.classification_report()
```python
print(sklearn.metrics.classification_report(y_pred, y_test))
```



# tensorflow
```python
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout, Concatenate, Add, Dot, Multiply, Reshape, Activation, BatchNormalization, SimpleRNNCell, RNN, SimpleRNN, LSTM, Embedding, Bidirectional, TimeDistributed, Conv1D, Conv2D, MaxPool1D, MaxPool2D, GlobalMaxPool1D, GlobalMaxPool2D, AveragePooling1D, AveragePooling2D, GlobalAveragePooling1D, GlobalAveragePooling2D, ZeroPadding2D
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import Input, Model, Sequential
```
## @tf.function
- 자동 그래프 생성
- 함수 정의문 직전에 사용
## tensor.shape
```python
lstm, for_h_state, for_c_state, back_h_state, back_c_state = Bidirectional(LSTM(units=64, dropout=0.5, return_sequences=True, return_state=True))(z)

print(lstm.shape, for_h_state.shape, back_h_state.shape)
```
## tf.identity()
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
W = tf.Variable(tf.zeros([2, 1], dtype=tf.float32), name="weight")
```
## tf.transpose()
## tf.cast()
```python
pred = tf.cast(h > 0.5, dtype=tf.float32)
```
- 조건이 True면 1, False면 0 반환.
- 혹은 단순히 Tensor의 자료형 변환.
## tf.concat()
```python
layer3 = tf.concat([layer1, layer2], axis=1)
```
- 지정한 axis의 dimension이 유지됩니다.
- Same as `tf.keras.layers.Concatenate()`
- `np.stack()`와 동일한 문법입니다.
## tf.stack()
```python
x = tf.stack(x, axis=0)
```
- 지정한 axis의 dimension이 +1됩니다.
- 동일한 shape을 가진 tensors에만 적용할 수 있습니다.
## tf.shape()
```python
batch_size = tf.shape(conv_output)[0]
```
## tf.reshape()
```python
conv_output = tf.reshape(conv_output, shape=(batch_size, output_size, output_size, 3,
                                                 5 + n_clss))
```
## tf.range()
```python
tf.range(3, 18, 3)
```
# tf.tile()
```python
y = tf.tile(y, multiples=[1, output_size])
```
## tf.constant\_initializer()
```
weight_init = tf.constant_initializer(weight)
```
## tf.GradientTape()
```python
with tf.GradientTape() as tape:
    hyp = W * X + b
    loss = tf.reduce_mean(tf.square(hyp - y))

dW, db = tape.gradient(loss, [W, b])
```
## tf.math
### tf.math.add(), tf.math.subtract(), tf.math.multiply(), tf.math.divide()
### tf.math.add_n()
```python
logits = tf.math.add_n(x) + self.w0
```
- Adds all input tensors element-wise.
- inputs : A list of tf.Tensor, each with the same shape and type.
### tf.math.square()
* 각 arguments를 제곱하여 ndarray 생성
### tf.math.argmax()
```python
y_pred = tf.math.argmax(model.predict(X_test), axis=1)
```
### tf.math.sign
```python
tf.math.sign(tf.math.reduce_sum(self.w * x) + self.b)
```
### tf.math.exp()
### tf.math.log()
### tf.math.equal()
```python
acc = tf.math.reduce_mean(tf.cast(tf.math.equal(pred, labels), dtype=tf.float32))
```
### tf.math.sigmoid()
### tf.math.reduce_sum(), tf.math.reduce_mean()
- source : https://www.tensorflow.org/api_docs/python/tf/math/reduce_sum#returns_1
- `axis=None` : 모든 elements에 대해 연산합니다.
- `axis=0` : reduces along the 1st dimension. dimension이 1만큼 감소합니다.
- `axis=1` : reduces along the 2nd dimension. dimension이 1만큼 감소합니다.
- `keepdims=True` : dimension이 감소하지 않습니다.
## tf.random
### tf.random.set\_seed()
### tf.random.normal()
```python
x = tf.Variable(tf.random.normal([784, 200], 1, 0.35))
```
## tf.nn
### tf.nn.softmax()
```python
h = tf.nn.softmax(tf.matmul(train_X, W) + b)
```
### tf.nn.relu
## tf.data
### tf.data.Dataset
#### tf.data.Dataset.from_tensor_slices()
```python
train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y)).shuffle(len(train_x)).batch(batch_size, drop_remainder=True).prefetch(batch_size)
test_dataset = tf.data.Dataset.from_tensor_slices((test_x, test_y)).shuffle(len(test_x)).batch(len(test_x)).prefetch(len(test_x))
```
##### tf.data.Dataset.from_tensor_slices().shuffle()
- 지정한 개수의 데이터를 무작위로 섞어서 출력합니다.
##### tf.data.Dataset.from_tensor_slices().batch()
- 지정한 개수의 데이터를 묶어서 출력합니다.
##### tf.data.Dataset.from_tensor_slices().prefetch()
- This allows later elements to be prepared while the current element is being processed. This often improves latency and throughput, at the cost of using additional memory to store prefetched elements.
## tf.train
### tf.train.Checkpoint()
## tf.keras
### Sequential()
```python
model = Sequential()
```
### Input()
```python
input_tokens = Input(shape=(max_len,), name="input_tokens", dtype=tf.int32)
```
### Model
```python
model = Model(inputs=inputs, outputs=logits, name="lr")
```
### tf.keras.utils
#### tf.keras.utils.get_file()
```python
base_url = "https://pai-datasets.s3.ap-northeast-2.amazonaws.com/recommender_systems/movielens/datasets/"
movies_path = tf.keras.utils.get_file(fname="movies.csv", origin=os.path.join(base_url, "movies.csv"))

movie_df = pd.read_csv(movies_path)
```
- 인터넷의 파일을 로컬 컴퓨터의 홈 디렉토리 아래 `.keras/datasets` 디렉토리로 다운로드합니다.
- `untar=True`
#### tf.keras.utils.to_categorical()
```python
tf.keras.utils.to_categorical([2, 5, 1, 6, 3, 7])
```
- Performs OHE.
### tf.keras.backend
#### tf.keras.backend.clear_session()
- Resets all state generated by Keras.
### tf.keras.datasets
#### tf.keras.datasets.mnist
##### tf.keras.datasets.mnist.load_data()
```python
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
```
#### tf.keras.datasets.reuters
##### tf.keras.datasets.reuters.load_data()
```python
(X_train, y_train), (X_test, y_test) = reuters.load_data(num_words=None, test_split=0.2)
```
#### tf.keras.datasets.cifar10
##### tf.keras.datasets.cifar10.load_data()
```python
(x_tr, y_tr), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
```
### tf.keras.optimizers
#### tf.keras.optimizers.SGD()
```python
opt = tf.keras.optimizers.SGD(lr=0.01)
```
#### tf.keras.optimizers.Adam()
- "adam"과 동일합니다.
#### tf.keras.optimizers.Adagrad()
#### optimizer.apply_gradients()
```python
opt.apply_gradients(zip([dW, db], [W, b]))
```
```python
opt.apply_gradients(zip(grads, model.trainable_variables))
```
### tf.keras.losses
#### tf.keras.losses.MeanSquaredError()
- "mse"와 동일합니다.
#### tf.keras.losses.BinaryCrossentropy()
- "binary_crossentropy"와 동일합니다.
#### tf.keras.losses.categorical_crossentropy()
```python
def loss_fn(model, images, labels):
    logits = model(images, training=True)
    loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_true=labels, y_pred=logits, from_logits=True))
    return loss
```
- 출처 : [https://hwiyong.tistory.com/335](https://hwiyong.tistory.com/335)
- 딥러닝에서 쓰이는 logit은 매우 간단합니다. 모델의 출력값이 문제에 맞게 normalize 되었느냐의 여부입니다. 예를 들어, 10개의 이미지를 분류하는 문제에서는 주로 softmax 함수를 사용하는데요. 이때, 모델이 출력값으로 해당 클래스의 범위에서의 확률을 출력한다면, 이를 logit=False라고 표현할 수 있습니다(이건 저만의 표현인 점을 참고해서 읽어주세요). 반대로 모델의 출력값이 sigmoid 또는 linear를 거쳐서 만들어지게 된다면, logit=True라고 표현할 수 있습니다.
- 클래스 분류 문제에서 softmax 함수를 거치면 `from_logits=False`(default값), 그렇지 않으면 `from_logits=True`(numerically stable)
- 정답 레이블이 one-hot encoding 형태일 경우 사용합니다.
#### tf.keras.losses.sparse_categorical_crossentropy()
```python
def loss_fn(model, x, y):
    return tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y_true=y, y_pred=model(x), from_logits=True))
```
- 정답 레이블이 one-hot vector가 아닐 경우 사용합니다.
### tf.keras.metrics
#### tf.keras.metrics.RootMeanSquaredError()
- "rmse"와 동일합니다.
#### tf.keras.metrics.BinaryCrossentropy()
- "binary_accuracy"와 동일합니다.
#### tf.keras.metrics.SparseCategoricalAccuracy()
- "sparse_categorical_accuracy"와 동일합니다.
### tf.keras.layers
#### Add()
```python
logits = Add()([logits_mlr, logits_fm, logits_dfm])
```
- It takes as input a list of tensors, all of the same shape, and returns a single tensor (also of the same shape).
#### Dot()
```python
pos_score = Dot(axes=(1, 1))([user_embedding, pos_item_embedding])
```
- `axes` : Integer or tuple of integers, axis or axes along which to take the dot product. If a tuple, should be two integers corresponding to the desired axis from the first input and the desired axis from the second input, respectively. Note that the size of the two selected axes must match.
#### Multiply()
```python
def se_block(x, c, r):
	z = GlobalAveragePooling2D()(x)
	z = Dense(units=c//r, activation="relu")(z)
	z = Dense(units=c, activation="sigmoid")(z)
	z = Reshape(target_shape=(1, 1, c))(z)
	z = Multiply()([x, z])
	return z
```
#### Reshape()
```python
z = Reshape(target_shape=(1, 1, ch))(z)
```
#### Concatenate()
```python
Concatenate(axis=1)(embs_fm)
```
- tf.concat()와 동일합니다.
#### Activation()
```python
x = Activation("relu")(x)
```
#### Flatten()
- 입력되는 tensor의 row를 펼쳐서 일렬로 만듭니다.
- 학습되는 weights는 없고 데이터를 변환하기만 합니다.
```python
model.add(Flatten(input_shape=(28, 28)))
```
#### Dense()
```python
Dense(units=52, input_shape=(13,), activation="relu")
```
- units: 해당 은닉층에서 활동하는 뉴런의 수(출력 값의 크기)
- activation: 활성화함수, 해당 은닉층의 가중치와 편향의 연산 결과를 어느 함수에 적합하여 출력할 것인가?
- input_shape : 입력 벡터의 크기. 여기서 13은 해당 데이터 프레임의 열의 수를 나타낸다. 데이터의 구조(이미지, 영상)에 따라 달라질 수 있다. 첫 번째 은닉층에서만 정의해준다.
#### Dropout()
* rate : dropout을 적용할 perceptron의 비율
#### BatchNormalization()
- usually used before activation function layers.
#### Conv1D()
```python
Conv1D(filters=n_kernels, kernel_size=kernel_size, padding="same", activation="relu", strides=1)
```
- `strides` : basically equals to 1
#### Conv2D()
```python
conv2d = Conv2D(filters=n_filters, kernel_size=kernel_size, strides=(1, 1), padding="same")(image)
```
- image : (batch, height of image, width of image, number of channels)
- kernel : (height of filter, width of filter, number of channels, number of kernels)
- convolution : (batch, height of convolution, width of convolution, number of kernels)
- number of channels와 number of kernels는 서로 동일합니다.
- `kernal_size` : window_size
- `padding="valid"` : No padding. There can be a loss of information. The size of the output image is smaller than the size of the input image.
- `padding="same"` : Normally, padding is set to "same" while training the model.
- `data_format` : `"channels_last"`
- `input_shape` : 처음에만 설정해 주면 됩니다.
- `activation` : `"tanh"`
- MaxPool1D, MaxPool2D, GlobalMaxPool1D, GlobalMaxPool2D, AveragePooling1D, AveragePooling2D, GlobalAveragePooling1D, GlobalAveragePooling2D, ZeroPadding2D
#### MaxPool1D()
- same as `MaxPooling1D()`
- `strides` : basically equals to 2
#### MaxPool2D()
```python
pool = MaxPool2D(pool_size=(2, 2), strides=1, padding="valid", data_format="channels_last")(image)
```
- same as `MaxPooling2D()`
#### GlobalMaxPool1D()
- same as `GlobalMaxPooling1D()`
- shape change : (a, b, c, d) -> (a, d).
#### GlobalMaxPool2D()
- same as `GlobalMaxPooling2D()`
- Downsamples the input representation by taking the maximum value over the time dimension.
- shape : (a, b, c) -> (b, c)
#### AveragePooling1D()
#### AveragePooling2D()
```python
model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid")
```
#### GlobalAveragePooling1D()
#### GlobalAveragePooling2D()
#### ZeroPadding2D
- z = ZeroPadding2D(padding=((1, 0), (1, 0)))(x)
- `padding`:
	- Int: the same symmetric padding is applied to height and width.
	- Tuple of 2 ints: interpreted as two different symmetric padding values for height and width: `(symmetric_height_pad, symmetric_width_pad)`.
	- Tuple of 2 tuples of 2 ints: interpreted as `((top_pad, bottom_pad), (left_pad, right_pad))`.
#### SimpleRNNCell()
#### RNN()
#### SimpleRNN()
```python
outputs, hidden_states = SimpleRNN(units=hidden_size)(x_data), input_shape=(timesteps, input_dim), return_sequences=True, return_state=True)(x_date)
```
- `SimpleRNN()` = `SimpleRNNCell()` + `RNN()`
- `batch_input_shape=(batch_size, timesteps, input_dim)`
- `return_sequences=False` : (default)time step의 마지막에서만 아웃풋을 출력합니다.(shape of output : (batch_size, hidden_size))
- `return_sequences=True` : 모든 time step에서 아웃풋을 출력합니다. many to many 문제를 풀거나 LSTM 레이어를 여러개로 쌓아올릴 때는 이 옵션을 사용합니다.(shape of output : (batch_size, timesteps, hidden_size))
- `return_state=True` : hidden state를 출력합니다.(shape of hidden state : (batch_size, hidden_size))
#### LSTM()
```python
_, hidden_state, cell_state = LSTM(units=256, return_state=True)(inputs_enc)
```
- `tf.keras.layers.SimpleRNN()`과 문법이 동일합니다.
- `return_state=True` : hidden state와 cell state를 출력합니다.
#### Bidirectional()
```python
Bidirectional(tf.keras.layers.SimpleRNN(hidden_size, return_sequences=True), input_shape=(timesteps, input_dim))
```
#### GRU()
```python
GRU(units=hidden_size, input_shape=(timesteps, input_dim))
```
- `SimpleRNN()`과 문법이 동일합니다.
#### Embedding()
```python
Embedding(input_dim=vocab_size+2, output_dim=emb_dim)
```
- `input_length=max_len` : 입력 sequence의 길이
- `mask_zero=True` : If mask_zero is set to True, as a consequence, index 0 cannot be used in the vocabulary. so input_dim should equal to size of vocabulary + 1
- `weights=[emb_mat]`
- `trainable=False` : 학습할지 아니면 초기 가중치 값을 그대로 사용할지 여부를 결정합니다.
#### TimeDistributed()
```python
model.add(TimeDistributed(tf.keras.layers.Dropout(rate=0.2)))
```
- TimeDistributed를 이용하면 각 time에서 출력된 아웃풋을 내부에 선언해준 레이어와 연결시켜주는 역할을 합니다.
- In keras - while building a sequential model - usually the second dimension (one after sample dimension) - is related to a time dimension. This means that if for example, your data is 5-dim with (sample, time, width, length, channel) you could apply a convolutional layer using TimeDistributed (which is applicable to 4-dim with (sample, width, length, channel)) along a time dimension (applying the same layer to each time slice) in order to obtain 5-d output.
#### tf.keras.layers.Layer
- custom layer를 만들려면 `tf.keras.layers.Layer` 클래스를 상속하고 다음 메서드를 구현합니다
    - __init__: 이 층에서 사용되는 하위 층을 정의할 수 있습니다. instance 생성 시에 호출됩니다.
    - build: 층의 가중치를 만듭니다. add_weight 메서드를 사용해 가중치를 추가합니다.
    - call: forward feeding 단계에서 호출됩니다. 입력 값을 이용해서 결과를 계산한 후 반환하면 됩니다.
#### tf.keras.layers.experimental
##### tf.keras.layers.experimental.preprocessing
###### Rescaling
```python
model.add(Rescaling(1/255, input_shape=(img_height, img_width, 3)))
```
```python
from tensorflow.keras.layers.experimental.preprocessing import RandomFlip, RandomRotation, RandomZoom
```
###### RandomFlip
```python
data_aug.add(RandomFlip("horizontal", input_shape=(img_height, img_width, 3)))
```
###### RandomRotationm
```python
data_aug.add(RandomRotation(0.1))
```
###### RandomZoom()
```python
data_aug.add(RandomZoom(0.1))
```
```python
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
```
### tf.keras.initializers
#### tf.keras.initializers.RandomNormal()
#### tf.keras.initializers.glorot_uniform()
#### tf.keras.initializers.he_uniform()
#### tf.keras.initializers.Constant()
### tf.keras.activations
#### tf.keras.activations.linear()
- Linear activation function(pass-through)
#### tf.keras.activations.sigmoid()
```python
outputs = tf.keras.activations.sigmoid(logits)
```
- "sigmoid"와 동일합니다.
#### tf.keras.activations.relu()
- "relu"와 동일합니다.
#### model.summary()
#### model.trainable_variables
#### model.save()
#### model.input
#### model.layers
```python
for layer in model.layers[1:]:
```
#### model.get_layer()
```python
model.get_layer("conv2d_22")
```
##### layer.name
##### layers.output
##### layer.input_shape
##### layer.output_shape
##### layer.get_weights()
```python
weight = layer.get_weights()[0]
bias = layer.get_weights()[1]
```
#### model.compile()
```python
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", loss_weights=[0.3, 0.3, 1], metrics=["accuracy"]) 
```
- `optimizer` : `"sgd"` | `"adam"` | `"rmsprop"`
- `loss` : `"mse"` | `"binary_crossentropy"` | `"categorical_crossentropy"` | `"sparse_categorical_crossentropy"`
- `metrics` : `["mse"]` | `["binary_accuracy"]` | `["categorical_accuracy"]` | `["sparse_categorical_crossentropy"]` | `["acc"]`
#### model.fit()
```python
hist = model.fit(x=X_train, y=y_train, validation_split=0.2, batch_size=64, epochs=10, verbose=1, shuffle=True, callbacks=[es, mc])
```
```python
hist = model.fit(train_ds, validation_data=val_ds, epochs=epochs)
```
- `validation_data`=(X_val, y_val)
#### model.fit_generator()
```python
hist = model.fit_generator(generator=train_set.shuffle(len(x_train)).batch(batch_size), epochs=n_epochs, validation_data=val_set.batch(batch_size))
```
##### hist.history
```python
hist.history["accuracy"]
```
- `"accuracy"`, `"val_accuracy"`, `"loss"`, `"val_loss"`
#### model.evaluate()
```python
score = model.evaluate(x_test, y_test, batch_size=128, verbose=0)
```
#### model.predict()
```python
preds = model.predict(x.values)
```
### tf.keras.callbacks
#### EarlyStopping()
```python
es = EarlyStopping(monitor="val_loss", mode="auto", verbose=1, patience=4)
```
- `mode` : One of {"auto", "min", "max"}. In min mode, training will stop when the quantity monitored has stopped decreasing; in "max" mode it will stop when the quantity monitored has stopped increasing; in "auto" mode, the direction is automatically inferred from the name of the monitored quantity.
- `patience` : Number of epochs with no improvement after which training will be stopped.
#### ModelCheckpoint()
```python
mc = ModelCheckpoint(filepath=model_path, monitor="val_binary_accuracy", mode="auto", verbose=1, save_best_only=True)
```
- `save_best_only=True` : `monitor` 기준으로 가장 좋은 값으로 모델이 저장됩니다.
- `save_best_only=False` : 매 epoch마다 모델이 filepath{epoch}으로 저장됩니다.
- `save_weights_only=True` : 모델의 weights만 저장됩니다.
- `save_weights_only=False` : 모델 레이어 및 weights 모두 저장됩니다.
- `verbose=1` : 모델이 저장 될 때 '저장되었습니다' 라고 화면에 표시됩니다.
- `verbose=0` : 화면에 표시되는 것 없이 그냥 바로 모델이 저장됩니다.
### tf.keras.preprocessing
#### tf.keras.preprocessing.image
##### load_img()
```python
from tensorflow.keras.preprocessing.image import load_img
```
```python
img = load_img(fpath, target_size=(img_height, img_width))
```
##### img_to_array()
```python
from tensorflow.keras.preprocessing.image import img_to_array
```
```python
img_array = img_to_array(img)
```
#### image_dataset_from_directory()
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
##### ds.class_names
```python
train_ds.class_names
```
##### ds.take()
```python
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        ax.imshow(images[i].numpy().astype("uint8"))
        ax.set_title(cls_names[labels[i]])
        ax.axis("off")
```
##### ImageDataGenerator
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
- `width_shift_range` : float: fraction of total width if < 1 or pixels if >= 1. 1-D array-like : random elements from the array. int : pixels from interval (-`width_shift_range`, `width_shift_range`)
- `height_shift_range`
- `brightness_range` : Tuple or list of two floats. Range for picking a brightness shift value from.
- `zoom_range`
- `horizontal_flip`
- `vertical_flip`
- transformation은 이미지에 변화를 주어서 학습 데이터를 많게 해서 성능을 높이기 위해 하는 것이기 때문에 train set만 해주고, test set에는 해 줄 필요가 없다. 그러나 주의할 것은 Rescale은 train, test 모두 해 주어야 한다.
- 참고 자료 : https://m.blog.naver.com/PostView.nhn?blogId=isu112600&logNo=221582003889&proxyReferer=https:%2F%2Fwww.google.com%2F
```python
gen.fit(x_tr)
```
- Only required if `featurewise_center` or `featurewise_std_normalization` or `zca_whitening` are set to True.
###### gen.flow()
```python
hist = model.fit(gen.flow(x_tr, y_tr, batch_size=32), validation_data=gen.flow(x_val, y_val, batch_size=32),
                 epochs=10)
```
###### gen.flow_from_directory()
```python
gen = ImageDataGenerator()
datagen_tr = gen.flow_from_directory(directory="./dogsandcats", target_size=(224, 224))
```
- `batch_size=batch_size`
- `target_size` : the dimensions to which all images found will be resized.
- `class_mode` : `"binary"`|`"categorical"`|`"sparse"`|`"input"`|`None`
- `class_mode="binary"` : for binary classification.
- `class_mode="categorical"` : for multi-class classification(OHE).
- `class_mode="sparse"` : for multi-class classification(no OHE).
- `class_mode="input"`
- `class_mode=None` : returns no label.
- `subset` : subset of data if `validation_split` is set in ImageDataGenerator(). `"training"`|`"validation"`
- `shuffle`
#### tf.keras.preprocessing.sequence
##### pad_sequences()
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
- `padding="pre" | "post"`
- `truncating="pre" | "post"`
- `value=` : padding에 사용할 value를 지정합니다.
#### tf.keras.preprocessing.text
##### tf.keras.preprocessing.text.Tokenizer()
```python
tkn = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size+2, oov_token="UNK", lower=True)
```
- `lower=False` : 대문자를 유지합니다.
##### tkn.fit_on_texts()
```python
tkn.fit_on_texts(["나랑 점심 먹으러 갈래 점심 메뉴는 햄버거 갈래 갈래 햄버거 최고야"])
```
##### tkn.word_index
```python
word2idx = tkn.word_index
```
##### tkn.index_word
##### tkn.word_counts
```python
word2cnt = dict(sorted(tkn.word_counts.items(), key=lambda x:x[1], reverse=True))

cnts = list(word2cnt.values())

for vocab_size, value in enumerate(np.cumsum(cnts)/np.sum(cnts)):
    if value >= ratio:
        break

print(f"{vocab_size:,}개의 단어로 전체 data의 {ratio:.0%}를 표현할 수 있습니다.")
print(f"{len(word2idx):,}개의 단어 중 {vocab_size/len(word2idx):.1%}에 해당합니다.")
```
##### tkn.texts_to_sequences()
```python
train_X = tkn.texts_to_sequences(train_X)
```
- `num_words`가 적용됩니다.
##### tkn.sequences_to_texts()
##### tkn.texts_to_matrix()
```python
tkn.texts_to_matrix(["먹고 싶은 사과", "먹고 싶은 바나나", "길고 노란 바나나 바나나", "저는 과일이 좋아요"], mode="count"))
```
- `mode="count"` | `"binary"` | `"tfidf"` | `"freq"`
- `num_words`가 적용됩니다.
### tf.keras.models
#### tf.keras.models.load_model()
```python
model = tf.keras.models.load_model(model_path)
```
### tf.keras.applications
#### tf.keras.applications.VGG16()
```python
vgg = tf.keras.applications.VGG16(input_shape=(224, 224, 3), include_top=False, weights="imagenet")
```
##### vgg.trainable
```python
vgg.trainable=Flase
```



# tensorflow_addons
```python
import tensorflow_addons as tfa
```
## tfa.optimizers
### tfa.optimizers.RectifiedAdam()
```python
opt = tfa.optimizers.RectifiedAdam(lr=5.0e-5, total_steps = 2344*4, warmup_proportion=0.1, min_lr=1e-5, epsilon=1e-08, clipnorm=1.0)
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



# tensorflow_datasets
```python
import tensorflow_datasets as tfds
```
## tfds.deprecated
### tfds.deprecated.text
#### tfds.deprecated.text.SubwordTextEncoder
##### tfds.deprecated.text.SubwordTextEncoder.build_from_corpus()
```python
tkn = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(train_data["document"], target_vocab_size=2**13)
```
##### tkn.subwords
##### tkn.encode()
```python
tkn.encode(train_data["document"][20])
```
##### tkn.decode()
```python
tkn.decode(tkn.encode(sample))
```



# bs4
## BeautifulSoup
```python
from bs4 import BeautifulSoup as bs
```
```python
soup = bs(xml,"lxml")
```
### 
### soup.find_all()
#### soup.find_all().find()
#### soup.find_all().find().get_text()
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
#### driver.find_element_by_\*().text
```python
df.loc[index, "배정초"]=driver.find_element_by_xpath("//\*[@id='detailContents5']/div/div[1]/div[1]/h5").text
```
#### driver.find_element_by_\*().get_attribute()
```python
driver.find_element_by_xpath("//*[@id='detailTab" +str(j) + "']").get_attribute("text")
```
#### driver.find_element_by_\*().click()
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
## WebDriverWait()
### WebDriverWait().until()
```python
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
```
```python
WebDriverWait(driver, 3).until(EC.presence_of_element_located((By.XPATH, "//\*[@id='detailContents5']/div/div[1]/div[1]/h5")))
```
- By.ID, By.XPATH
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
## urllib.request
### urllib.request.urlopen()
```python
xml = urllib.request.urlopen(full_url).read().decode("utf-8")
```
### urllib.request.urlretrieve()
```python
urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt", filename="ratings_train.txt")
```
- 해당 URL에 연결된 파일을 다운로드합니다.



# urllib3
```python
import urllib3
```
## urllib3.PoolManager()
### urllib3.PoolManager().request()
```python
urllib3.PoolManager().request("GET", url, preload_content=False)
```



# pathlib
```python
import pathlib
```
## pathlib.Path()
```python
data_dir = pathlib.Path(data_dir)
```



# requests
```python
import requests
```
## requests.get()
```python
req = requests.get("https://github.com/euphoris/datasets/raw/master/imdb.zip")
```
### req.content



# wget
```python
import wget
```
## wget.download()
```python
wget.download("https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz")
```



# category_encoders
```python
!pip install --upgrade category_encoders
```
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
    cv2.rectangle(img=img, pt1=(rect[0], rect[1]), pt2=(rect[0]+rect[2], rect[1]+rect[3]), color=(0, 0, 255), thickness=2)
```
## cv2.circle()
```python
for i, rect in enumerate(rects_selected):
    cv2.circle(img, (rect[0]+1, rect[1]-12), 12, (0, 0, 255), 2))
```
## cv2.getTextSize()
```python
(text_width, text_height), baseline = cv2.getTextSize(text=label, fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                                                  fontScale=font_scale, thickness=bbox_thick)
```
## cv2.puttext()
```python
cv2.putText(img=img, text=label, org=(x1, y1-4), fonFace=cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=font_scale, color=text_colors, thickness=bbox_thick, lineType=cv2.LINE_AA)
```
## cv2.resize()
```python
img_resized = cv2.resize(img, dsize=(640, 480), interpolation=cv2.INTER_AREA)
```
- `dsize` : (new_width, new_height)
## cv2.cvtColor()
```python
img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
```
## cv2.imread()
```python
img = cv2.imread("300.jpg")
```
## cv2.imwrite()
```python
cv2.imwrite("/content/drive/My Drive/Computer Vision/fire hydrants.png", ori_img)
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



# selectivesearch
## selectivesearch.selective_search()
```python
_, regions = selectivesearch.selective_search(img_rgb, scale=100, min_size=2000)
```
```python
img_recs = cv2.rectangle(img=img_rgb_copy, pt1=(rect[0], rect[1]),
                                 pt2=(rect[0]+rect[2], rect[1]+rect[3]),
                                 color=green_rgb, thickness=2)
```



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
- `dict(id2word)` is same as `dict(id2word.id2token)`
#### id2word.token2id
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
#### gensim.models.AuthorTopicModel.load()
```python
model = gensim.models.AuthorTopicModel.load("kakaotalk model")
```
## gensim.models.ldamodel.Ldamodel()
```python
model = gensim.models.ldamodel.LdaModel(dtm, num_topics=n_topics, id2word=id2word, alpha="auto", eta="auto")
```
## gensim.models.Word2Vec()
```python
    model = gensim.models.Word2Vec(result, size=100, window=5, min_count=5, workers=4, sg=0)
```
- `size` : 임베딩 벡터의 차원.
- `min_count` : 단어 최소 빈도 수(빈도가 적은 단어들은 학습하지 않는다)
- `workers` : 학습을 위한 프로세스 수  
- `sg=0` :cBoW
- `sg=1` : Skip-gram.  
### gensim.models.FastText()
```python
model = gensim.models.FastText(sentences, min_count=5, sg=1, size=300, workers=4, min_n=2, max_n=7, alpha=0.05, iter=10, window=7)
```
### gensim.models.KeyedVectors
### gensim.models.KeyedVectors.load_word2vect_format()
```python
model = gensim.models.KeyedVectors.load_word2vec_format("eng_w2v")
```
```python
model_google = gensim.models.KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin.gz", binary=True)  
```
- Loads a model.
#### model.vectors
#### model.save()
```python
model.save("kakaotalk model")
```
#### model.show_topic()
```python
model.show_topic(1, topn=20)
```
- Arguments : (the index of the topic, number of words to print)
#### model.wv
##### model.wv.vecotrs
##### model.wv.most_similar()
```python
model.wv.most_similar("안성기")
```
##### model.wv.save_word2vec_format()
```python
model.wv.save_word2vec_format("eng_w2v")
```



# glove
```python
!pip install glove_python
```
### glove.Corpus
### glove.Glove



# seqeval
## seqeval.metrics
### precision_score
### recall_score
### f1_score
### classification_report
```python
from seqeval.metrics import classification_report
```



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



# konlpy
## konlpy.tag
```python
from konlpy.tag import *

okt = Okt()
kkm = Kkma()
kmr = Komoran()
hnn = Hannanum()
```
#### okt.nouns(), kkm.nouns(), kmr.nouns(), hnn.nouns()
#### okt.morphs(), kkm.morphs(), kmr.morphs(), hnn.morphs()
- `stem=True`
- `norm=True`
#### okt.pos(), kkm.pos(), kmr.pos(), hnn.pos()
- `stem=True`
- `norm=True`



# ckonlpy
```python
!pip install customized_konlpy
```
```python
from ckonlpy.tag import Twitter

twt = Twitter()
```
## twt.add_dictionary()
```python
twt.add_dictionary("은경이", "Noun")
```


# sentencepiece
```python
import sentencepiece as sp
```
## sp.SentencePieceTrainer
### sp.SentencePieceTrainer.Train()
```python
sp.SentencePieceTrainer.Train("--input=naver_review.txt --model_prefix=naver --vocab_size=5000 --model_type=bpe")
```
- `input` : 학습시킬 파일
- `model_prefix` : 만들어질 모델 이름
- `vocab_size` : 단어 집합의 크기
- `model_type` : 사용할 모델 (unigram(default), bpe, char, word)
- `max_sentence_length`: 문장의 최대 길이
- `pad_id`, `pad_piece`: pad token id, 값
- `unk_id`, `unk_piece`: unknown token id, 값
- `bos_id`, `bos_piece`: begin of sentence token id, 값
- `eos_id`, `eos_piece`: end of sequence token id, 값
- `user_defined_symbols`: 사용자 정의 토큰
- `.model`, `.vocab` 파일 두개가 생성 됩니다.
## sp.SentencePieceProcessor()
```python
spp = sp.SentencePieceProcessor()
```
### spp.load()
```python
spp.load("imdb.model")
```
### spp.GetPieceSize()
- 단어 집합의 크기를 확인합니다.
### spp.encode_as_ids()
- 원래 문장 -> index
### spp.encode_as_pieces()
- 원래 문장 -> subword
### spp.IdToPiece()
```python
spp.IdToPiece(4)
```
- index -> subword
### spp.DecodeIds()
```python
sp.DecodeIds([54, 200, 821, 85])
```
- index -> 원래 문장
### spp.PieceToId()
```python
spp.PieceToId("영화")
```
- subword -> index
### spp.DecodePieces()
```python
sp.DecodePieces(["▁진짜", "▁최고의", "▁영화입니다", "▁ᄏᄏ"])
```
- subword -> 원래 문장
### spp.encode()
- `out_type=str`: `spp.encode_as_pieces()`와 동일합니다.
- `out_type=int`: `spp.encode_as_ids()`와 동일합니다.
- `enable_sampling=True`: drop-out을 적용합니다.
- `alpha=0.1`: 해당 확률로 drop-out을 적용합니다.
- `nbest_size=-1`


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
fpath = "C:/Windows/Fonts/malgun.ttf"
font_name = mpl.font_manager.FontProperties(fname=fpath).get_name()
```
## mpl.rc()
```python
mpl.rc("font", family=font_name)
```
- `family="NanumBarunGothic"`
```python
mpl.rc("axes", unicode_minus=False)
```
### plot(kind="pie")
```python
cnt_genre.sort_values("movie_id", ascending=False)["movie_id"].plot(ax=ax, kind="pie", startangle=90, legend=True)
```
### plot(kind="bar")
```python
data["label"].value_counts().plot(kind="bar")
```
### cbar.set_label()
```python
cbar.set_label(label="전용면적(m²)", size=15)
```
## matplotlib.pyplot
```python
import matplotlib.pyplot as plt
```
### plt.setp()
```python
plt.setp(obj=ax1, yticks=ml_mean_gr_ax1["le"], yticklabels=ml_mean_gr_ax1.index)
```
### plt.style.use()
```python
plt.style.use("dark_background")
```
### plst.subplot()
```python
for i in range(9):
	ax = plt.subplot(3, 3, i + 1)
```
### plt.subplots()
```python
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12)
```
#### fig.colorbar()
```python
cbar = fig.colorbar(ax=ax, mappable=scatter)
```
#### fig.savefig()
```python
fig.savefig("means_plot_200803.png", bbox_inches="tight")
```
#### fig.tight_layout()
#### ax.imshow()
```python
ax.imshow(image.numpy().reshape(3,3), cmap="Greys")
```
```python
ax.imshow(images[i].numpy().astype("uint8"))
```
#### ax.set()
```python
ax.set(title="Example", xlabel="xAxis", ylabel="yAxis", xlim=[0, 1], ylim=[-0.5, 2.5], xticks=data.index, yticks=[1, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3])
```
#### ax.set_title()
```python
ax.set_title("Example", size=20)
```
#### ax.set_xlabel(), ax.set_ylabel()
```python
ax.set_xlabel("xAxis", size=15)
```
#### ax.set_xlim(), ax.set_ylim()
```python
ax.set_xlim([1, 4])
```
#### ax.axes
#### ax.axis()
```python
ax.axis([2, 3, 4, 10])
```
```phthon
ax.axis("off")
```
##### ax.xaxis.set_visible(), ax.yaxis.set_visible()
```python
ax1.xaxis.set_visible(False)
```
##### ax.xaxis.set_label_position(), ax.yaxis.set_label_position()
```python
ax.xaxis.set_label_position("top")
```
##### ax.xaxis.set_ticks_position(), ax.yaxis.set_ticks_position()
```python
ax1.yaxis.set_ticks_position("right")
```
```python
ax.xaxis.set_ticks_position("top")
```
##### ax.xaxis.set_tick_position(), ax.yaxis.set_tick_position()
```python
ax2.yaxis.set_ticks_position("right")
```
#### ax.xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter(), ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter()
```python
ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter("{x:,.0f}"))
```
#### ax.invert_xaxis(), ax.invert_yaxis()
#### ax.set_xticks(), ax.set_yticks()
```python
ax.set_yticks(np.arange(1, 1.31, 0.05))
```
- 화면에 표시할 눈금을 설정합니다.
#### ax.tick_params()
```python
ax.tick_params(axis="x", labelsize=20, labelcolor="red", labelrotation=45, grid_linewidth=3)
```
#### ax.legend()
```python
ax.legend(fontsize=14, loc="best")
```
- `loc`="best" | "upper right" ...
#### ax.grid()
```python
ax.grid(axis="x", color="White", alpha=0.3, linestyle="--", linewidth=2)
```
#### ax.plot()
```python
ax.plot(df1.index, df1["mean"], linestyle="dashdot", linewidth=1.5, color="black", label="10년 전(09.08 ~ 10.07) 전국 실거래가")
```
- linestyle : "dashdot" | "dashed" | "solid" | "dotted"
#### ax.scatter()
```python
ax.scatter(gby["0.5km 내 교육기관 개수"], gby["실거래가"], s=70, c=gby["전용면적(m²)"], cmap="RdYlBu", alpha=0.7, edgecolors="black", linewidth=0.5)
```
#### ax.bar()
```python
ax.bar(x=nby_genre.index, height=nby_genre["movie_id"])
```
#### ax.barh()
```python
ax.barh(y=ipark["index"], width=ipark["가경4단지 84.8743A"], height=0.2, alpha=0.5, color="red", label="가경4단지 84.8743A", edgecolor="black", linewidth=1)
```
#### ax.hist()
```python
ax.hist(cnt_genre["genre"], bins=30)
```
#### ax.axhline()
```python
ax.axhline(y=mean, color="r", linestyle=":", linewidth=2)
```
#### ax.text()
```python
for _, row in ml_gby_ax1.iterrows():
    ax1.text(y=row["le"]-0.18, x=row["abs_error"], s=round(row["abs_error"], 1), va="center", ha="left", fontsize=10)
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
### sb.countplot()
```python
sb.countplot(ax=ax, data=cmts202011, x="dep")
```
- DataFrame
```python
sb.countplot(ax=ax, x=label_train)
```
- array, or list of arrays
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



# networkx
```python
improt networks as nx
```
## nx.Graph()
```python
g = nx.Graph()
```
## nx.DiGraph()
## nx.circular_layout()
```python
pos = nx.circular_layout(g)
```
## nx.draw_networks_nodex()
```python
nx.draw_networkx_nodes(g, pos, node_size=2000)
```
## nx.draw_networkx_edges()
```python
nx.draw_networkx_edges(g, pos, width=weights)
```
## nx.draw_networkx_labels()
```python
nx.draw_networkx_labels(g, pos, font_family=font_name, font_size=11)
```
## nx.draw_shell()
```python
nx.draw_shell(g, with_labels=False)
```
### g.add_nodes_from()
```python
g.add_nodes_from(set(df.index.get_level_values(0)))
```
### g.add_edge()
```python
for _, row in df.iterrows():
    g.add_edge(row.name[0], row.name[1], weight=row["cowork"]/200)
```
### g.edges()
```python
weights = [cnt["weight"] for (_, _, cnt) in g.edges(data=True)]
```



# MeCab
```python
!git clone https://github.com/SOMJANG/Mecab-ko-for-Google-Colab.git
%cd Mecab-ko-for-Google-Colab
!bash install_mecab-ko_on_colab190912.sh
```
- Google Colab에 설치
```python
!pip install mecab_python-0.996_ko_0.9.2_msvc-cp37-cp37m-win_amd64.whl
```
- Windows에 설치
- source : https://cleancode-ws.tistory.com/97, https://github.com/Pusnow/mecab-python-msvc/releases/tag/mecab_python-0.996_ko_0.9.2_msvc-2
```python
import MeCab
```
```python
class Mecab:
    def pos(self, text):
        p = re.compile(".+\t[A-Z]+")
        return [tuple(p.match(line).group().split("\t")) for line in MeCab.Tagger().parse(text).splitlines()[:-1]]
    
    def morphs(self, text):
        p = re.compile(".+\t[A-Z]+")
        return [p.match(line).group().split("\t")[0] for line in MeCab.Tagger().parse(text).splitlines()[:-1]]
    
    def nouns(self, text):
        p = re.compile(".+\t[A-Z]+")
        temp = [tuple(p.match(line).group().split("\t")) for line in MeCab.Tagger().parse(text).splitlines()[:-1]]
        nouns=[]
        for word in temp:
            if word[1] in ["NNG", "NNP", "NNB", "NNBC", "NP", "NR"]:
                nouns.append(word[0])
        return nouns
    
mcb = Mecab()
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
## nltk.tokenize
### nltk.tokenize.word_tokenize()
```python
nltk.tokenize.word_tokenize("Don't be fooled by the dark sounding name, Mr. Jone's Orphanage is as cheery as cheery goes for a pastry shop.")
```
### nltk.tokenize.sent_tokenize()
```python
nltk.tokenize.sent_tokenize("I am actively looking for Ph.D. students and you are a Ph.D student.")
```
### WordPunctTokenizer()
```python
from nltk.tokenize import WordPunctTokenizer
```
#### WordPunctTokenizer().tokenize()
```python
WordPunctTokenizer().tokenize("Don't be fooled by the dark sounding name, Mr. Jone's Orphanage is as cheery as cheery goes for a pastry shop.")
```
### TreebankWordTokenizer()
```python
from nltk.tokenize import TreebankWordTokenizer
```
#### TreebankWordTokenizer().tokenize()
```python
TreebankWordTokenizer().tokenize("Starting a home-based restaurant may be an ideal. it doesn't have a food chain or restaurant of their own.")
```
- Penn Treebank Tokenization.
## nltk.stem
### PorterStemmer()
```python
from nltk.stem import PorterStemmer
```
```python
ps = PorterStemmer()
```
#### ps.stem()
```python
[ps.stem(word) for word in ["formalize", "allowance", "electricical"]]
```
### WordNetLemmatizer()
```python
from nltk.stem import WordNetLemmatizer
```
```python
wnl = WordNetLemmatizer()
```
#### wnl.lemmatize()
```python
wnl.lemmatize("watched", "v")
```
## nltk.Text()
```python
text = nltk.Text(total_tokens, name="NMSC")
```
### text.tokens
### text.vocab()
- returns frequency distribution
#### text.vocab().most_common()
```python
text.vocab().most_common(10)
```
### text.plot()
```python
text.plot(50)
```
## nltk.download()
- "punkt", "wordnet", "stopwords", "movie_reviews"
## nltk.corpus
### stopwords
```python
from nltk.corpus import stopwords
```
#### stopwords.words()
```python
stopwords.words("english")
```
### movie_reviews
```python
from nltk.corpus import movie_reviews
```
#### movie_reviews.sents()
```python
sentences = [sent for sent in movie_reviews.sents()]
```
### nltk.corpus.treebank
#### nltk.corpus.treebank.tagged_sents()
```python
tagged_sents = nltk.corpus.treebank.tagged_sents()
```
## nltk.translate
### nltk.translate.bleu_score
```python
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
```
#### sentence_bleu()
```python
ref = [["this", "is", "a", "test"], ["this", "is" "test"]]
cand = ["this", "is", "a", "test"]
score = nltk.translate.bleu_score.sentence_bleu(ref, cand)
```
- `weights`=(1/2, 1/2, 0, 0)
#### corpus_bleu()
```python
refs = [[["this", "is", "a", "test"], ["this", "is" "test"]]]
cands = [["this", "is", "a", "test"]]
score = nltk.translate.bleu_score.corpus_bleu(refs, cands)
```
- `weights`=(1/2, 1/2, 0, 0)
#### SmoothingFunction()
## nltk.ngrams()
```python
nltk.ngrams("I am a boy", 3)
```



# khaiii
## KhaiiiApi
```python
from khaiii import KhaiiiApi
```
```python
api = KhaiiiApi()
```
### api.analyze()
#### word.morphs
##### morph.lex
##### morph.tag
```python
morphs = []
sentence = "하스스톤 전장이 새로 나왔는데 재밌어요!"
for word in api.analyze(sentence):
    for morph in word.morphs:
        morphs.append((morph.lex, morph.tag))
```



# kss
## kss.split_sentences()
```python
kss.split_sentences("딥 러닝 자연어 처리가 재미있기는 합니다. 그런데 문제는 영어보다 한국어로 할 때 너무 어려워요. 농담아니에요. 이제 해보면 알걸요?")
```



# pykospacing
```python
!pip install git+https://github.com/haven-jeon/PyKoSpacing.git --user
```
```python
!pip install keras==2.1.5
```
```python
import pykospacing
```
## pykospacing.spacing()
```
sent_ks = pykospacing.spacing("오지호는극중두얼굴의사나이성준역을맡았다.성준은국내유일의태백권전승자를가리는결전의날을앞두고20년간동고동락한사형인진수(정의욱분)를찾으러속세로내려온인물이다.")
```



# soynlp
## soynlp.normalizer
```python
from soynlp.normalizer import *
```
### emoticon_normalize()
```python
emoticon_normalize("앜ㅋㅋㅋㅋ이영화존잼쓰ㅠㅠㅠㅠㅠ", num_repeats=2)
```
### repeat_normalize()
```python
repeat_normalize("와하하하하하하하하하핫", num_repeats=2)
```



# hanspell
```python
!pip install git+https://github.com/ssut/py-hanspell.git
```
## spell_checker
```python
from hanspell import spell_checker
```
### spell_checker.check()
#### spell_checker.check().checked
```python
sent_ckd = spell_checker.check("맞춤법 틀리면 외 않되? 쓰고싶은대로쓰면돼지").checked
```
# collections
## Counter()
```python
from collections import Counter
```
```python
word2cnt = Counter(words)
```
- lst의 원소별 빈도를 나타내는 dic을 반환합니다.
### Counter[key]
### Counter().values()
```python
sum(Counter(nltk.ngrams(cand.split(), 2)).values())
```
### Counter().most_common()
## deque()
```python
from collections import deque
```
```python
dq = deque("abc")
```
### dq[]
```python
dq[2] = "d"
```
### dq.append()
### dq.appenleft()
### dq.pop()
### dq.popleft()
### dq.extend()
### dq.extendleft()
### dq.remove()



# functools
## reduce()
```python
from functools import reduce
```
```python
reduce(lambda acc, cur: acc + cur["age"], users, 0)
```
```python
reduce(lambda acc, cur: acc + [cur["mail"]], users, [])
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



# transformers
```python
!pip install --target=$my_path transformers==3.5.0
```
## BertModel
```python
from transformers import BertModel
```
```python
model = BertModel.from_pretrained("monologg/kobert")
```
## TFBertModel
```python
from transformers import TFBertModel
```
### TFBertModel.from_pretrained()
```python
model = TFBertModel.from_pretrained("monologg/kobert", from_pt=True,
                                    num_labels=len(tag2idx), output_attentions=False,
                                    output_hidden_states = False)
```
```python
bert_outputs = model([token_inputs, mask_inputs])
```
#### model.save()
```python
model.save("kobert_navermoviereview.h5", save_format="tf")
```
### BertModel.from_pretrained()
```python
model = BertModel.from_pretrained("monologg/kobert")
```



# tokenization_kobert
```python
urllib.request.urlretrieve("https://raw.githubusercontent.com/monologg/KoBERT-NER/master/tokenization_kobert.py", filename="tokenization_kobert.py")
```
## KoBertTokenizer
```python
from tokenization_kobert import KoBertTokenizer
```
- KoBertTokenizer 파일 안에 from transformers import PreTrainedTokenizer가 이미 되어있습니다.
### KoBertTokenizer.from_pretrained()
```python
tokenizer = KoBertTokenizer.from_pretrained("monologg/kobert")
```
#### tokenier.tokenize()
```python
tokenizer.tokenize("보는내내 그대로 들어맞는 예측 카리스마 없는 악역")
```
#### tokenizer.encode()
```python
tokenizer.encode("보는내내 그대로 들어맞는 예측 카리스마 없는 악역")
```
- `max_length`
- `padding="max_length"`
#### tokenizer.convert_tokens_to_ids()
```python
tokenizer.convert_tokens_to_ids("[CLS]")
```
- unknown token : 0, "[PAD]" : 1, "[CLS]" : 2, "[SEP]" : 3



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
## random.random()
- Returns a random number in [0, 1).
## random.sample()
```python
names = random.sample(list(set(data.index)), 20)
```
## random.shuffle()
- In-place function



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
```python
views["apt_name"] = views["pageTitle"].apply(lambda x:re.search(r"(.*)\|(.*)\|(.*)", x).group(2) if "|" in x else x)
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
- . : newline을 제외한 어떤 character
- \w, [a-zA-Z0-9_]: 어떤 알파벳 소문자 또는 알파벳 대문자 또는 숫자 또는 공백
- \W, [^a-zA-Z0-9_] : 어떤 알파벳 소문자 또는 알파벳 대문자 또는 숫자 또는 공백이 아닌
- [ㄱ-ㅣ가-힣] : 어떤 한글
- \d, [0-9] : 어떤 숫자
- \D, [^0-9] : 어떤 숫자가 아닌
- \s : 공백
- \S : 공백이 아닌 어떤 character
- \t : tab
- \n : newline
- \r : return
- [] : [] 안의 문자를 1개 이상 포함하는
- \* : 0개 이상의 바로 앞의 character(non-greedy way)
- \+ : 1개 이상의 바로 앞의 character(non-greedy way)
- \? : 1개 이하의 바로 앞의 character
- {m,n} : m개~n개의 바로 앞의 character(생략된 m은 0과 동일, 생략된 n은 무한대와 동일)(non-greedy way)
- {n} : n개의 바로 앞의 character
- ^ : 바로 뒤의 문자열로 시작하는
- $ : 바로 앞의 문자열로 끝나는



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
```python
for punct in string.punctuation:
    sw.add(punct)
```



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
```
```python
print("{}초 경과".format(round(time.time())-time_before))
```
## time.localtime()
## time.strftime()
```python
time.strftime("%Y%m%d", time.localtime(time.time()))
```



# tqdm
## tqdm.notebook
### tqdm
```python
from tqdm.notebook import tqdm
```
- for Jupyter Notebook
## tqdm.auto
### tqdm
```python
from tqdm.auto import tqdm
```
- for Google Colab
## tqdm.pandas()
- `df.progress_apply()`를 사용하기 위해 필요합니다.



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



# google
## google.colab
### drive
```python
from google.colab import drive
```
#### drive.mount()
```python
drive.mount("/content/drive")
```



# zipfile
```python
import zipfile
```
## zipfile.ZipFile()
```python
with zipfile.ZipFile("spa-eng.zip", "r") as f:
    file = f.open("spa-eng/spa.txt")
    data = pd.read_csv(file, names=["eng", "spa"], sep="\t")
```
### zipfile.ZipFile().extractall()
```python
zipfile.ZipFile("glove.6B.zip").extractall(cur_dir)
```



# lxml
## etree
```python
from lxml import etree
```
### etree.parse()
```python
with zipfile.ZipFile("ted_en-20160408.zip", "r") as z:
	target_text = etree.parse(z.open("ted_en-20160408.xml", "r"))
```


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
## os.chdir()
## os.environ
## os.pathsep
```python
os.environ["PATH"] += os.pathsep + "C:\Program Files (x86)/Graphviz2.38/bin/"
```
## os.path
### os.path.join()	
```python
os.path.join("C:\Tmp", "a", "b")
```
\>\>\> "C:\Tmp\a\b"
### os.path.exists()
```python
if os.path.exists("C:/Users/5CG7092POZ/train_data.json"):
```
## os.path.dirname()
```python
os.path.dirname("C:/Python35/Scripts/pip.exe")
```
\>\>\> "C:/Python35/Scripts"
- 경로 중 디렉토리명만 얻습니다.



# glob
## glob.glob()
```python
path = "./DATA/전체"
filenames = glob.glob(path + "/*.csv")
```



# pickle
```python
import pickle as pk
```
## pk.dump()
```python
with open("filename.pkl", "wb") as f:
	pk.dump(list, f)
```
## pk.load()
```python
with open("filename.pkl", "rb") as f:
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
```python
with open(f"{model_path}_hist", "w") as f:
	json.dump(hist.history, f)
```
## json.load()
```python
with open(path, "r", encoding="utf-8") as f:
    train_data = json.load(f)
```
```python
with open(f"{model_path}_hist", "r") as f:
	hist.history = json.load(f)
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
- `"RGB"` | `"RGBA"` | `"CMYK"` | `"L"` | `"1"`
### img.paste()
```python
img1.paste(img2, (20,20,220,220))
```
- img2.size와 동일하게 두 번째 parameter 설정.	
## Image.new()
```python
mask = Image.new("RGB", icon.size, (255, 255, 255))
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
- "Darwin", "Windows" 등 OS의 이름 반환합니다.



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
- `pygame.KEYUP` : 키보드를 누른 후 뗄 때
- `pygame.KEYDOWN` : 키보드를 누를 때
### pygame.init()
### pygame.quit()
### pygame.display.set_model()
```python
gamepad = pygame.display.set_mode((pad_width, pad_height))
```



# 
import openpyxl

wb = openpyxl.Workbook()
sheet = wb.active
sheet["H8"] = "=SUM(H6:H7)"
content = ["민수", "준공분", "거제2차", "15.06", "18.05", "1279"]
sheet.append(content)
wb.save("test.xlsx")




# csv
```python
import csv
```
## csv.QUOTE_NONE
```python
subws = pd.read_csv("imdb.vocab", sep="\t", header=None, quoting=csv.QUOTE_NONE)
```



# shutil
```python
import shutil
```
## shutil.copyfile()
```python
shutil.copyfile("./test1/test1.txt", "./test2.txt")
```
## shutil.copyfileobj()
```python
shutil.copyfileobj(urllib3.PoolManager().request("GET", url, preload_content=False), open(file_dir, "wb"))
```



# logging
## logging.basicConfig()
```python
logging.basicConfig(level=logging.ERROR)
```



# openpyxl
```python
import openpyxl
```
## openpyxl.Workbook()
```python
wb = openpyxl.Workbook()
```
### wb.active
```python
sheet = wb.active
```
### wb.save()
```python
wb.save("test.xlsx")
```
#### sheet.append()
```python
content = ["민수", "준공분", "거제2차", "15.06", "18.05", "1279"]
sheet.append(content)
```
#### sheet[]
```python
sheet["H8"] = "=SUM(H6:H7)"
```



# datetime
```python
impoirt datetime
```
## datetime.datetime
## datetime.timedelta
```python
day = start + datetime.timedelta(days=1)
```
## datetime.today()
## strftime()
```python
strftime("%Y-%m-%d")
```
- `%Y` : 4자리 연도 숫자 
- `%m` : 2자리 월 숫자
- `%d` : 2자리 일 숫자
- `%H` : 24시간 형식 2자리 시간 숫자
- `%M` : 2자리 분 숫자
- `%S` : 2자리 초 숫자
- `%A` : 요일 문자열(영어)
- `%B` : 월 문자열(영어)



# traceback
```python
import traceback
```
## traceback.format_exec()
```python
try:            
    cur.execute(query)            
    result = cur.fetchall()        
except Exception as e:            
    msg = traceback.format_exc()            
    msg += "\n\n Query: \n" + query            
    print(msg)  
```



# psycopg2
## psycopg2.connect()
```python
conn = psycopg2.connect(host="postgresql-production1.cluster-ro-cxb5zwyq1mu5.ap-northeast-2.rds.amazonaws.com", dbname="hdc", user="postgres", password="dtRW2aj14KYl7aQl8oZd")
```
### conn.cursor()
```python
cur = conn.cursor(cursor_factory=RealDictCursor)
```
#### cur.excute()
```python
cur.excute(query)
```
#### cur.fetchall()
```python
result = cur.fetchall() 
```
#### cur.close()
## psycopg2.extras
### RealDictCursor
```python
from psycopg2.extras import RealDictCursor
```
```python
jupyter notebook --NotebookApp.iopub_data_rate_limit=1.0e10
```



# bisect
## bisect_left(), bisect_right()
```python
from bisect import bisect_left, bisect_right
```
```python
def count_by_range(a, left, right):
    return bisect_right(a, right) - bisect_left(a, left)
```



# google
## google.colab
### google.colab.patches
#### cv2_imshow()
```python
from google.colab.patches import cv2_imshow
```



# colorsys
## colorsys.hsv_to_rgb()
```python
hsv_tuples = [(idx/n_clss, 1, 1) for idx in idx2cls.keys()]
colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
colors = list(map(lambda x: (int(x[0]*255), int(x[1]*255), int(x[2]*255)), colors))
```
```python
appo["end"] = appo["end"].dt.normalize()
```
```python
hr["제외여부"] = hr.apply(lambda x:"제외" if ("외주" in x["하위그룹"]) | ("촉탁" in x["하위그룹"]) | ("파견" in x["하위그룹"]) | (x["재직여부"]=="퇴직") else ("본부인원에서만 제외" if ("PM" in x["조직명"]) | ("신규준비" in x["직무"]) | (x["직무"]=="휴직") | (x["직무"]=="비상계획") | (x["직무"]=="축구협") | (x["직무"]=="비서") | ("조직명" in x["조직명"]) | (x["직무"]=="미화") else "포함"), axis=1)
```
