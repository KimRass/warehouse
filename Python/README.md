Written by KimRass

# Automatically Reload Modules
```python
%reload_ext autoreload
%autoreload 1
%aimport <<Module1>>, <<Module2>>, ...
```
- Reload all modules imported with `%aimport` every time before executing the Python code typed.

# Error Messages
## `TypeError`
### `TypeError: unhashable type: '<<Data Type>>'`
- The unhashable objects are not allowed in set or dictionary key. The dictionary or set hashes the object and uses the hash value as a primary reference to the key.
### `TypeError: '<<Data Type>>' object is not subscriptable`
- "Subscriptable" means that the object contain, or can contain, other objects.
### `TypeError: sequence index must be integer, not 'slice'`
- This error raises because `collections.deque()` cannot be sliced.
### `TypeError: '<<Data Type>>' object is not iterable`
### `TypeError: <<Function>> takes exactly <<Number of Needed Arguments>> argument (<<Number of Input Arguments>> given)`
### `TypeError: sequence item 0: expected str instance, int found`
### `TypeError: can only concatenate tuple (not "<<Data Type>>") to tuple`
### `TypeError: '<<Argument>>' is an invalid keyword argument for <<Function>>`
### `TypeError: unsupported operand type(s) for <<Operator>>: '<<Data Type1>>' and '<<Data Type2>>'`
### `TypeError: heap argument must be a list`
### `TypeError: Invalid comparison between dtype=<<Data Type>> and <<Data Type>>`
### `TypeError: 'range' object cannot be interpreted as an integer`
### `TypeError: '<<Class>>' object is not callable`
### `TypeError: object of type <<Data Type>> has no <<Function>>`
## `NameError`
### `NameError: name '<<Variable>>' is not defined`
## `ZeroDivisionError`
### `ZeroDivisionError: division by zero`
## `SyntaxError`
### `SyntaxError: unexpected EOF while parsing`
### `SyntaxError: invalid syntax`
### `SyntaxError: unmatched ')', SyntaxError: unmatched ']'`
### `SyntaxError: Generator expression must be parenthesized`
## `UnboundLocalError`
### `UnboundLocalError: local variable '<<Variable>>' referenced before assignment`
## `ValueError`
### `ValueError: invalid literal for int() with base 10: <<String>>`
### `ValueError: '<<Variable>>' is not in list`
### `ValueError: max() arg is an empty sequence`
### `ValueError: overflow in timedelta operation`
### `ValueError: too many values to unpack (expected <<Number>>)`
## `IndexError`
### `IndexError: list index out of range`
### `IndexingError: Too many indexers`
## `ImportError`
### `ImportError: cannot import name '<<Function>>' from '<<Package>>'`
## `AttributeError`
### `AttributeError: module '<Class>' has no attribute '<<Function>>'`
### `AttributeError: '<<Data Type>>' object has no attribute '<<Function>>'`
## `RuntimeError`
### `RuntimeError: dictionary changed size during iteration`
## `KeyError`
### `KeyError: <<Dictionary Key>>`
## `KeyboardInterrupt`
### `KeyboardInterrupt:`
## `MemoryError`
### `MemoryError: Unable to allocate ...`

# Python Built-in Functions
## `bin()`, `oct()`, `hex()`
- Source: https://wiki.python.org/moin/BitwiseOperators
- `~x`(`NOT`): Returns the complement.
- `x & y`(`AND`): Returns `1` if the corresponding bit of `x` and of `y` is `1`, otherwise returns `0`.
- `x | y`(`OR`): Returns `0` if the corresponding bit of `x` and of `y` is `0`, otherwise returns `1`.
- `x ^ y`(`XOR`): Returns the same bit as the corresponding bit in `x` if that bit in `y` is `0`, otherwise the complement.
- `x << y`, `x >> y`: Returns `x` with the bits shifted to the left(right) by `y` places (and new bits on the right-hand-side are zeros).
## `int()`
- `base`: (Default `10`) Number format.
## `round()`
```python
print(round(summ/leng, 1))
```
## `open()`
```python
f = open("D:/Github/Work/Tableau/datamart_password.txt", "r")
```
### `f.readline()`, `f.readlines()`
## `hash()`
## `display()`
## `print()`
- `end`: (default `"\n"`)
- `sep`: (default `" "`) Determine the value to join elements with.
```python
print(f"{'a':0>10}")
```
- You can also append characters other than white spaces, by adding the specified characters before the `>`(right align), `^`(center align) or `<`(left align) character:
## `isinstance()`
```python
if not isinstance(movie, frozenset):
    movie = frozenset(movie)
```
## `type()`
```python
type(test_X[0][0])
```
## `sum()`
```python
sum(sentences, [])
```
- 두번째 층의 대괄호 제거
## `assert`
```python
assert model_name in self.model_list, "There is no such a model."
```
## `eval()`
```python
A = [eval(f"A{i}") for i in range(N, 0, -1)]
```
- `eval()` is for expression and returns the value of expression.
## `exec()`
```python
for data in ["tasks", "comments", "projects", "prj_members", "members"]:
    exec(f"{data} = pd.read_csv('D:/디지털혁신팀/협업플랫폼 분석/{data}.csv')")
```
```python
exec(f"{table} = pd.DataFrame(result)")
```
- `exce()` is for statement and return `None`.
## `open()`
```python
with open("C:/Users/5CG7092POZ/nsmc-master/ratings_train.txt", "r", encoding="utf-8") as f:
    train_docs = [line.split("\t") for line in f.read().splitlines()][1:]
```
## `input()`
```python
A = list(map(int, in "A를 차례대로 입력 : ").split()))
```
## `ord()`
- Returns the unicode code of a specified character.
## `chr()`
- Returns the character that represents the specified unicode code.
## `Variable.data`
### `Variable.data.nbytes`
```python
print(f"{sparse_mat.data.nbytes:,}Bytes"
```
## List
- Mutable.
- Unhashable.
### `List.index()`
### `List.append()`
- Adds the argument as a single element to the end of a List. 
### `List.extend()`
- Iterates over the argument and adding each element to the List and extending the List.
### `List.insert()`
- idx, value 순으로 arg를 입력합니다.
### `List.remove()`
```python
features.remove("area")
```
### `List.count()`
### `sorted()`
```python
sorted(confs, key=lambda x:(x[0], x[1]))
```
- `reverse`: (Bool, default `False`)
- `key`: Define a function to sort by.
### `reversed()`
```python
list(reversed([int(i) for i in str(n)]))
```
### `map()`
```python
list(map(len, train_tkn))
```
```python
x_data = list(map(lambda word : [char2idx.get(char) for char in word], words))
```
### `filter()`
### `sum()`
```python
sum(sents, [])
```
### List Comprehension
```python
chars = set([char for word in words for char in word])
```
```python
idxs = [idx for idx, num in zip(range(len(nums)), nums) if num!=0]
```
## Set
- Mutable.
- Unhashable.
- No order.
### `<<Set1>> & <<Set2>>`
- Returns the union of `<<Set1>>` and `<<Set2>>`.
### `<<Set1>> | <<Set2>>`
- Returns the intersection of `<<Set1>>` and `<<Set2>>`.
### `Set.add()`
- Adds the argument as a single element to the end of a Set if it is not in the Set.
### `Set.update()`
- It expects a single or multiple iterable sequences as arguments and appends all the elements in these iterable sequences to the Set.
### `Set.discard()`
## Frozenset
- 구성 요소들이 순서대로 들어 있지 않아 인덱스를 사용한 연산을 할 수 없고
- 유일한 항목 1개만 들어가 있습니다.
- set 객체는 구성 항목들을 변경할 수 있지만 frozenset 객체는 변경할 수 없는 특성을 지니고 있어, set에서는 사용할 수 있는 메소드들 add, clear, discard, pop, remove, update 등은 사용할 수 없습니다. 
- 항목이 객체에 포함되어 있는지(membership)를 확인할 수 있으며,  항목이 몇개인지는 알아낼 수 있습니다.
- frozenset은 dictionary의 key가 될 수 있지만 set은 될 수 없음.
## `Dictionary`
- Mutable.
- Unhashable.
### `Dictionary[]`, `Dictionary.get()`
- key를 입력받아 value를 반환합니다.
### `Dictionary.items()`
```python
for key, value in dic.items():
    print(key, value)
```
### `Dictionary.setdefault()`
### `Dictionary.update()`
```python
dic.update({key1:value1, key2:value2})
```
### `Dictionary.pop()`
```python
dic.pop(<<key>>)
```
### `Dictionary.keys()`, `Dictionary.values()`
- Data type: `dict_keys`, `dict_values` respectively.
### `Dictionary.fromkeys()`
### `sorted()`
```python
word2cnt = dict(sorted(tkn.word_counts.items(), key=lambda x:x[1], reverse=True))
```
### Dictionary Comprehension
```python
min_dists = {i:0 if i == start else math.inf for i in range(1, V + 1)}
```
## String
- Immutable.
### `String.format()`
### `String.ljust()`, `String.rjust()`
```python
string.ljust(<<Target Length>>, <<Character to Pad>>)
```
### `String.zfill()`
### `String.join()`
```python
" ".join(["good", "bad", "worse", "so good"])
```
 - Join all items in a Tuple or List into a string, using `String`.
### `String.split()`
- Split a string into a list where each word is a list item.
- `maxsplit`: How many splits to do.
### `String.upper()`, `String.lower()`
```python
orders.columns = orders.columns.str.lower()
```
### `String.isupper()`, `String.islower()`
### `String.isalpha()`
### `String.isdigit()`
### `String.count()`
```python
"저는 과일이 좋아요".count("과일이")
```
### `String.find()`
- Return the first index of the argument.
### `String.startswith()`, `String.endswith()`
- Return `True` if a string starts with the specified prefix. If not, return `False`
### `String.strip()`, `String.lstrip()`, `String.rstrip()`
### `String.replace()`
- `count`: (int, optional) A number specifying how many occurrences of the old value you want to replace. Default is all occurrences.

# `math`
## `math.exp()`
## `math.log()`
## `math.log2()`
## `math.log10()`
## `math.factorial()`
## `math.comb()`
## `math.floor()`
## `math.ceil()`
## `math.gcd()`
## `math.isnan()`
## `math.inf`

# `pandas`
```python
import pandas as pd
```
## `pd.api`
### `pd.api.types`
#### `pd.api.types.is_string_dtype()`, `pd.api.types.is_numeric_dtype()`
```python
def load_table(table_name):
    conn = pymssql.connect(server="133.186.215.165", database="HDCMART", user="hdcmart_user", password=password, charset="UTF8")
    query = f"""
    SELECT *
    FROM {table_name}
    """
    table = pd.read_sql(query, conn)
    table.columns = table.columns.str.lower()
    for col in table.columns:
        if pd.api.types.is_string_dtype(table[col]):
            table[col] = table[col].str.encode("latin-1", errors="ignore").str.decode("euc-kr", errors="ignore")
    return table
```
## `pd.options`
### `pd.options.display`
#### `pd.options.display.max_rows`, `pd.options.display.max_columns`, `pd.options.display.width`, `pd.options.display.float_format`
### `pd.options.mode`
#### `pd.options.mode.chained_assignment`
```python
pd.options.display.max_columns = None
```
```python
pd.set_option("display.float_format", "{:.3f}".format)
```
```python
pd.options.mode.chained_assignment = None
```
- Ignore `SettingWithCopyWarning`
## `pd.DataFrame()`
- `data`: (Array, List, List of Tuples, Dictionary of List)
- `index`
- `columns`
## `pd.Series()`
```python
idf_ser = pd.Series(idf, index=vocab)
```
## `pd.read_csv()`
```python
data = pd.read_csv("fra.txt", names=["src", "tar", "CC"], sep="\t")
```
- `thousands=","`
- `float_precision="round_trip"`
- `skiprows`
- `error_bad_lines=False`
- `names`: List of column names to use.
- `index_col`
## `pd.read_excel()`
## `pd.read_table()`
```python
ratings_train = pd.read_table("ratings_train.txt", usecols=["document", "label"])
```
- `names`: List of column names to use.
- `usecols`
### `pd.read_sql()`
```python
mall = pd.read_sql(query, conn)
```
## `DataFrame.head()`, `Series.head()`, `DataFrame.tail()`, `Series.tail()`
## `DataFrame.to_csv()`
```python
data.to_csv("D:/☆디지털혁신팀/☆실거래가 분석/☆데이터/실거래가 전처리 완료_200928-3.csv")
```
- `index`: (Bool)
## `DataFrame.to_excel()`
- `sheet_name`: (String, default `"Sheet1"`)
- `na_rep`: (String, default `""`) Missing data reprsentation.
- `float_format`
- `header`: If a list of string is given it is assumed to be aliases for the column names.
- `merge_cells`
## `pd.crosstab()`
```python
pd.crosstab(index=data["count"], columns=data["weather"], margins=True)
```
## `pd.concat()`
```python
data_without = pd.concat([data_without, data_subset], axis=0)
```
```python
dfs = []
for filename in filenames:
    dfs.append(pd.read_csv(filename))
data = pd.concat(dfs, ignore_index=True, axis=0)
```
- `join`: (`"inner"`, `"outer"`, default `"outer"`)
- `ignore_index`: If `True`, do not use the index values along the concatenation axis. The resulting axis will be labeled `0`, …, `n - 1`.
## `pd.pivot_table()`
```python
pivot = pd.pivot_table(uses_df[["daytime", "weekday", "cnt"]], index="daytime", columns="weekday", values="cnt", aggfunc=np.sum)
```
## `pd.melt()`
```python
n_post = pd.melt(n_post, id_vars="Date", var_name="Emp", value_name="NPost", ignore_index=False)
```
```python
data = pd.melt(data, id_vars=basic_cols + money_cols, value_name="date")
data = pd.melt(data, id_vars=basic_cols + ["date"], var_name="classification", value_name="money")
```
- `pd.pivot_table()`의 반대 과정입니다.
## `pd.cut()`
```python
raw_all["temp_group"] = pd.cut(raw_all["temp"], 10)
```
## `pd.Categorical()`
```python
results["lat"] = pd.Categorical(results["lat"], categories=order)
results_ordered = results.sort_values(by="lat")
```
- dtype을 `category`로 변환.
- `ordered`: (Bool, deafult `False`): category들 간에 대소 부여.
- Reference: https://kanoki.org/2020/01/28/sort-pandas-dataframe-and-series/
## `pd.get_dummies()`
```python
data = pd.get_dummies(data, columns=["heating", "company1", "company2", "elementary", "built_month", "trade_month"], drop_first=False, dummy_na=True)
```
- `prefix`: String to append DataFrame column names. Pass a list with length equal to the number of columns when calling get_dummies on a DataFrame.
- `dop_first`: Whether to get k-1 dummies out of k categorical levels by removing the first level.
- `dummy_na`: Add a column to indicate NaNs, if False NaNs are ignored.
- 결측값이 있는 경우 `drop_first`, `dummy_na` 중 하나만 `True`로 설정해야 함
## `pd.to_datetime()`
```python
ratings_df["rated_at"] = pd.to_datetime(ratings_df["rated_at"], unit="s")
```
### `pd.to_datetime().dt`
#### `pd.to_datetime().dt.hour`, `pd.to_datetime().dt.day`,  `pd.to_datetime().dt.week`,  `pd.to_datetime().dt.dayofweek`, `pd.to_datetime().dt.month`, `pd.to_datetime().dt.quarter`, `pd.to_datetime().dt.year`
#### `pd.to_datetime().dt.normalize()`
```python
appo["end"] = appo["end"].dt.normalize()
```
## `pd.date_range()`
```python
raw.time = pd.date_range(start="1974-01-01", periods=len(raw), freq="M")
```
- Return a fixed frequency DatetimeIndex.
## `pd.merge()`
```python
data = pd.merge(data, start_const, on=["지역구분", "입찰년월"], how="left")
```
```python
pd.merge(df1, df2, left_on="id", right_on="movie_id")
```
```python
floor_data = pd.merge(floor_data, df_conv, left_index=True, right_index=True, how="left")
```
- `how`: (`"left"`, `"right"`, `"outer"`, `"inner"`, `"cross"` default `"inner"`)
## `pd.Grouper()`
```python
n_tasks_month = tasks.groupby(pd.Grouper(key="task_date", freq="M")).size()
```
## `pd.MultiIndex`
### `pd.MultiIndex.from_tuples()`
```python
order = pd.MultiIndex.from_tuples((hq, dep) for dep, hq in dep2hq.items())
```
## `pd.plotting`
### `pd.plotting.scatter_matrix()`
```python
pd.plotting.scatter_matrix(data, figsize=(18, 18), diagonal="kde")
```
### `pd.plotting.lag_plot()`
```python
fig = pd.plotting.lag_plot(ax=axes[0], series=resid_tr["resid"].iloc[1:], lag=1)
```
## `DataFrame.style`
### `DataFrame.style.set_precision()`
### `DataFrame.style.set_properties()`
- (`"background_color"`, `"font_size"`, `"color"`, `"border_color"`, ...)
### `DataFrame.style.bar()`
- `color`
### `DataFrame.style.background_gradient()`
```python
data.corr().style.background_gradient(cmap="Blues").set_precision(1).set_properties(**{"font-size":"9pt"})
```
- `cmap`
## `DataFrame.info()`
## `DataFrame.describe()`
```python
raw_data.describe(include="all").T
```
- Show number of samples, mean, standard deviation, minimum, Q1(lower quartile), Q2(median), Q3(upper quantile), maximum of each independent variable.
## `DataFrame.corr()`
## `DataFrame.shape`
## `DataFrame.ndim`
## `DataFrame.quantile()`
```python
top90per = plays_df[plays_df["plays"]>plays_df["plays"].quantile(0.1)]
```
## `DataFrame.groupby()`
```python
df.groupby(["Pclass", "Sex"], as_index=False)
```
- Return an iterable Tuple in the form of (a group, a DataFrame in the group).
### `DataFrame.groupby().groups`
### `DataFrame.groupby().mean()`, `DataFrame.groupby().count()`
- Return DataFrame.
### `DataFrame.groupby().size()`
- Return Series.
## `DataFrame.groupby().apply()`
```python
over4.groupby("user_id")["movie_id"].apply(set)
```
## `DataFrame.pivot()`
```python
df_pivoted = df.pivot("col1", "col2", "col3")
```
## `DataFrame.stack()`
- 열 인덱스 -> 행 인덱스로 변환
## `DataFrame.unstack()`
- 행 인덱스 -> 열 인덱스로 변환
- `pd.pivot_table()`과 동일
- `level`: index의 계층이 정수로 들어감
```python
groupby.unstack(level=-1, fill_value=None)
```
## `DataFrame.append()`
```python
df = df.append({"addr1":addr1, "addr2":addr2, "dist":dist}, ignore_index=True)
```
## `DataFrame.apply()`
```python
hr["코스트센터 분류"] = hr.apply(lambda x:"지사" if ("사업소" in x["조직명"]) or ("베트남지사" in x["조직명"]) else ("본사" if re.search("^\d", x["코스트센터"]) else "현장"), axis=1)
```
```python
hr["제외여부"] = hr.apply(lambda x:"제외" if ("외주" in x["하위그룹"]) | ("촉탁" in x["하위그룹"]) | ("파견" in x["하위그룹"]) | (x["재직여부"]=="퇴직") else ("본부인원에서만 제외" if ("PM" in x["조직명"]) | ("신규준비" in x["직무"]) | (x["직무"]=="휴직") | (x["직무"]=="비상계획") | (x["직무"]=="축구협") | (x["직무"]=="비서") | ("조직명" in x["조직명"]) | (x["직무"]=="미화") else "포함"), axis=1)
```
## `DataFrame.progress_apply()`
```python
data["morphs"] = data["review"].progress_apply(mcb.morphs)
```
- `tqdm.pandas()` should be followed.
## `DataFrame.rename()`
```python
data = data.rename({"단지명.1":"name", "세대수":"houses_buildings", "저/최고층":"floors"}, axis=1)
```
## `DataFrame.reindex()`
```python
pivot = pivot.reindex(dep_order)
```
## `DataFrame.insert()`
```python
data.insert(3, "age2", data["age"]*2)
```
## `DataFrame.sort_values()`
```python
results_groupby_ordered = results_groupby.sort_values(by=["error"], ascending=True, na_position="first", axis=0)
```
## `DataFrame.nlargest()`, `DataFrame.nsmallest()`
```python
df.nlargest(3, ["population", "GDP"], keep="all")
```
- `keep`: (`"first"`, `"last"`, `"all"`)
## `DataFrame.index`
### `DataFrame.index.name`, `DataFrame.index.names`
## `DataFrame.sort_index()`
## `DataFrame.set_index()`
```python
data = data.set_index(["id", "name"])
```
- `inplace`: (Bool, default `False`)
## `DataFrame.reset_index()`
- `drop`: (bool, default False) Reset the index to the default integer index.
- `level`: Only remove the given levels from the index. Removes all levels by default.
## `DataFrame.loc()`
```python
data.loc[data["buildings"]==5, ["age", "ratio2"]]
data.loc[[7200, "대림가경"], ["houses", "lowest"]]
```
## `DataFrame.isin()`
```python
train_val = data[~data["name"].isin(names_test)]
```
- `values`: (List, Dictionary)
## `DataFrame.query()`
```python
data.query("houses in @list")
```
- 외부 변수 또는 함수 사용 시 앞에 @을 붙임.
## `DataFrame.idxmax()`
```python
data["genre"] = data.loc[:, "unknown":"Western"].idxmax(axis=1)
```
## `DataFrame.drop()`
```python
data = data.drop(["Unnamed: 0", "address1", "address2"], axis=1)
```
```python
data = data.drop(data.loc[:, "unknown":"Western"].columns, axis=1)
```
## `DataFrame.duplicated()`
```python
df.duplicated(keep="first)
```
## `DataFrame.columns`
```python
concat.columns = ["n_rating", "cumsum"]
```
### `DataFrame.columns.drop()`
```python
uses_df.columns.drop("cnt")
```
- Make new Index with passed list of labels deleted.
### `DataFrame.columns.droplevel`
```python
df.columns = df.columns.droplevel([0, 1])
```
## `DataFrame.drop_duplicates()`
```python
df = df.drop_duplicates(["col1"], keep="first")
```
## `DataFrame.mul()`
```python
df1.mul(df2)
```
## `DataFrame.dot()`
```python
def cos_sim(x, y):
    return x.dot(y)/(np.linalg.norm(x, axis=1, ord=2)*np.linalg.norm(y, ord=2))
```
## `DataFrame.isna()`
## `DataFrame.notna()`
```python
retail[retail["CustomerID"].notna()]
```
## `DataFrame.insert()`
```python
asso_rules.insert(1, "antecedents_title", asso_rules["antecedents"].apply(lambda x : id2title[list(x)[0]]))
```
## `DataFrame.drop()`
## `DataFrame.dropna()`
```python
data = data.dropna(subset=["id"])
```
```python
df.loc[~df.index.isin(df.dropna().index)]
```
## `DataFrame.fillna()`
```python
data = data.fillna(method="ffill")
```
- `method="ffill"`: Propagate last valid observation forward to next valid backfill.
## `DataFrame.sample()`, `Series.sample()`
```python
baskets_df.sample(frac=0.05)
```
```python
set(n_per_movie_unseen.sample(n=100, replace=False, weights=n_per_movie).index)
```
- `random_state`: (Int)
## `DataFrame.iterrows()`
- Iterate over DataFrame rows as (index, Series) pairs.
## `DataFrame.iteritems()`, `DataFrame.items()`
```python
{k:v for k, v in x_train.iteritems()}
```
- Iterate over the DataFrame columns, returning a tuple with the column name and the content as a Series.
## `Series.iteritems()`, `Series.items()`
```python
for i, value in raw_data["quarter2"].items():
    print(i, value)
```
- Iterate over (index, value) tuples.
## `DataFrame.asfreq()`
## `DataFrame.rolling()`
```python
raw_all[["count"]].rolling(window=24, center=False).mean()
```
## `DataFrame.diff()`, `Series.diff()`
## `DataFrame.shift()`
```python
raw_data["count_lag1"] = raw_data["count"].shift(1)
```
## `DataFrame.mean()`
```python
ui.mean(axis=1)
```
## `DataFrame.mean().mean()`
## `DataFrame.add()`, `DataFrame.sub()`, `DataFrame.mul()`, `DataFrame.div()`, `DataFrame.pow()`
```python
adj_ui = ui.sub(user_bias, axis=0).sub(item_bias, axis=1)
```
## `Series.rename()`
```python
plays_df.groupby(["user_id"]).size().rename("n_arts")
```
## `DataFrame.value_counts()`, `Series.value_counts()`
```python
ratings_df["movie_id"].value_counts()
```
## `Series.nunique()`
```python
n_item = ratings_df["movie_id"].nunique()
```
## `Series.isnull()`
## `Series.map()`
```python
target_ratings["title"] = target_ratings["movie_id"].map(target)
```
```python
all_seen = ratings_df_target_set[ratings_df_target_set.map(lambda x : len(x)==5)].index
```
## `Series.astype()`
```python
data.loc[:, cats] = data.loc[:, cats].astype("category")
```
- `dtype`: (`"int32"`, `"int63"`, `"float64"`, `"object"`, `"category"`, `"string"`)
- `errors`: (`"raise"`, `"ignore"`, default `"raise"`)
## `Series.hist()`
## `Series.cumsum()`
```python
cumsum = n_rating_item.cumsum()/len(ratings_df)
```
## `Series.min()`, `Series.max()`, `Series.mean()`, `Series.std()`
## `Series.str`
### `Series.str.replace()`
```python
data["parking"] = data["parking"].str.replace("대", "", regex=False)
```
### `Series.str.split()`
```python
data["buildings"] = data.apply(lambda x : str(x["houses_buildings"]).split("총")[1][:-3], axis=1)
```
- `pat`: Same as `" "` When omitted.
### `Series.str.strip()`
```python
data["fuel"] = data["heating"].apply(lambda x:x.split(",")[1]).str.strip()
```
### `Series.str.contains()`
```python
train[train["token"].str.contains(".", regex=False)]
```
### `Series.str.encode()`, `Series.str.decode()`
```python
infos[col] = infos[col].str.encode("latin-1").str.decode("euc-kr")
```
- `error`: (`"strict"`, `"ingore"`, `"replace"`, default `"strict"`)
## `Series.cat`
### `Series.cat.categories`
### `Series.cat.set_categories()`
```python
ser.cat.set_categories([2, 3, 1], ordered=True)
```
- 순서 부여
### `Series.cat.codes`
```python
for cat in cats:
    data[cat] = data[cat].cat.codes
```
- Perform label encoding.
## `Series.items()`
```python
for k, v in target.items():
    queries.append(f"{k}-{v}")
```

# `numpy`
```python
import numpy as np
```
## `Array.size`
## `Array.astype()`
```python
x_train = x_train.astype("float32")
```
## `Array.ravel()`
```python
arr.ravel(order="F")	
```
- `order="C"` : row 기준
- `order="F"` : column 기준
## `Array.flatten()`
- 복사본 반환
## `Array.T`
## `Array.shape`
## `Array.transpose()`, `np.transpose()`
```python
conv_weights = np.fromfile(f, dtype=np.float32, count=np.prod(conv_shape)).reshape(conv_shape).transpose((2, 3, 1, 0))
```
## `np.inf`
## `np.set_printoptions()`
```python
np.set_printoptions(precision=3)
```
```python
np.set_printoptions(edgeitems=3, infstr="inf", linewidth=75, nanstr="nan", precision=8, suppress=False, threshold=1000, formatter=None)
```
- Go back to the default options.
## `np.load()`
```python
intent_train = np.load("train_text.npy").tolist()
```
## `np.logical_and()`, `np.logical_or()`
```python
mask = np.logical_or((pred_bbox[:, 0] > pred_bbox[:, 2]), (pred_bbox[:, 1] > pred_bbox[:, 3]))
```
## `np.array_equal()`
```python
np.array_equal(arr1, arr2)
```
## `np.arange()`
```python	
np.arange(5, 101, 5)
```
- `start`: Start of interval. The interval includes this value. The default start value is 0.
- `stop`: End of interval. The interval does not include this value, except in some cases where step is not an integer and floating point round-off affects the length of out.
- `step`
## `np.zeros()`, `np.ones()`
```python
np.ones(shape=(2, 3, 4))
```
## `np.empty()`
## `np.full()`
```python
img_paded = np.full(shape=(tar_height, tar_width, 3), fill_value=128.)
```
## `np.eye()`
```python
np.eye(4)
```
## `np.log()`
```python
np.log(data)
```
## `np.ones_like()`, `np.zeros_like()`
- `a`: (Array-like)
## `np.linspace()`
```python
np.linspace(-5, 5, 100)
```
## `np.meshgrid()`
```python
xs = np.linspace(0, output_size-1, output_size)
ys = np.linspace(0, output_size-1, output_size)
x, y = np.meshgrid(xs, ys)
```
## `np.any()`
```python
np.any(arr>0)
```
## `np.all()`
## `np.where()`
```python
np.min(np.where(cumsum >= np.cumsum(cnts)[-1]*ratio))
```
## `np.tanh()`
```python
temp = np.tanh(np.dot(Wh, h_t) + np.dot(Wx, x_t) + b)
```
## `np.shape()`
```python
np.shape(hidden_states)
```	
## `np.isin()`
```python
data[np.isin(data["houses"], list)]
```
## `np.prod()`
```python
conv_shape = (filters, in_dim, kernel_size, kernel_size)
conv_weights = np.fromfile(f, dtype=np.float32, count=np.product(conv_shape))
```
- Return the product of Array elements over a given axis.
- `axis`
## `np.argmax()`
## `np.swapaxes()`
```python
feature_maps = np.transpose(conv2d, (3, 1, 2, 0))
```
```python
feature_maps = np.swapaxes(conv2d, 0, 3)
```
## `np.max()`, `np.min()`
```python
np.max(axis=-1)
```
## `np.maximum()`, `np.minimum()`
- Element-wise minimum(maximum) of Array elements.
## `np.cumsum()`
```python
np.cumsum(cnt)
```
## `np.quantile()`
```python
lens = sorted([len(doc) for doc in train_X])
ratio = 0.99
max_len = int(np.quantile(lens, ratio))
print(f"가장 긴 문장의 길이는 {np.max(lens)}입니다.")
print(f"길이가 {max_len} 이하인 문장이 전체의 {ratio:.0%}를 차지합니다.")
```
## `np.einsum()`
```python
def get_p_pi(self, policy):
	p_pi = np.einsum("ik, kij -> ij", policy, self.P)
	return p_pi
```
## `np.concatenate()`
```python
intersected_movie_ids = np.concatenate([json.loads(row) for row in rd.mget(queries)], axis=None)
```
## `np.stack()`
## `np.delete()`
```python
idx_drop = [idx for idx, doc in enumerate(X_train) if len(doc) == 0]
X_train = np.delete(X_train, idx_drop, axis=0)
```
## `np.random`
### `np.random.seed()`
```python
np.random.seed(23)
```
### `np.random.random()`
```python
np.random.random((2, 3, 4))
```
- Create an Array of the given shape and populate it with random samples from a uniform distribution over \[0, 1).
- Similar with `np.random.rand()`
### `np.random.randint()`
```python
np.random.randint(1, 100, size=(2, 3, 4))	
```
### `np.random.choice()`
- Generates a random sample from a given 1-D array.
- If an ndarray, a random sample is generated from its elements. If an int, the random sample is generated as if a were np.arange(a).
```python
np.random.choice(5, size=(2, 3, 4))
```
### `np.random.normal()`
- `loc`: Mean.
- `scale`: Standard deviation.
- `size`: Output shape.
## `np.digitize()`
```python		
bins=range(0, 55000, 5000)
data["price_range"]=np.digitize(data["money"], bins)
```
## `np.isnan()`
## `np.nanmean()`
## `np.sort()`
## `np.reshape()`, `Array.reshape()`
```python
bn_weights = np.reshape(bn_weights, newshape=(4, filters))[[1, 0, 2, 3]]
```
```python
bn_weights = bn_weights.reshape((4, filters))[[1, 0, 2, 3]]
```
## `np.expand_dims()`
```python
np.expand_dims(mh_df.values, axis=1)
```
## `np.newaxis`(= `None`)
```python
iou = calculate_iou(best_bbox[None, :4], bboxes_cls[:, :4])
```
## `np.unique()`
```python
items, counts = np.unique(intersected_movie_ids, return_counts=True)
```
## `np.linalg`
### `np.linalg.norm()`
```python
np.linalg.norm(x, axis=1, ord=2)
```
- `ord=1`: L1 normalization.
- `ord=2`: L2 normalization.
### `np.linalg.solve()`
### `np.linalg.cond()`
### `np.linalg.hilbert()`
## `np.sqrt()`
## `np.power()`
## `np.exp()`
```python
def sig(x):
    return 1 / (1 + np.exp(-x))
```
## `np.add.outer()`, `np.multiply.outer()`
```python
euc_sim_item = 1 / (1 + np.sqrt(np.add.outer(square, square) - 2*dot))
```
## `np.fill_diagonal()`
```python
np.fill_diagonal(cos_sim_item, 0	
```
## `np.fromfile()`
- `count`: Number of items to read. `-1` means all items (i.e., the complete file).

# `bs4`
## `BeautifulSoup()`
```python
from bs4 import BeautifulSoup as bs
```
```python
soup = bs(xml,"lxml")
```
### `soup.find_all()`
#### `soup.find_all().find()`
#### `soup.find_all().find().get_text()`
```python
features = ["bjdcode", "codeaptnm", "codehallnm", "codemgrnm", "codesalenm", "dorojuso", "hocnt", "kaptacompany", "kaptaddr", "kaptbcompany",  "kaptcode", "kaptdongcnt", "kaptfax", "kaptmarea", "kaptmarea",  "kaptmparea_136", "kaptmparea_135", "kaptmparea_85", "kaptmparea_60",  "kapttarea", "kapttel", "kapturl", "kaptusedate", "kaptdacnt", "privarea"]
for item in soup.find_all("item"):
    for feature in features:
        try:
            kapt_data.loc[index, feature] = item.find(feature).get_text()
        except:
            continue
```

# `selenium`
## `webdriver`
```python
from selenium import webdriver
```
```python
driver = webdriver.Chrome("chromedriver.exe")
```
### `driver.get()`
```python
driver.get("https://www.google.co.kr/maps/")
```
### `driver.find_element_by_css_selector()`, `driver.find_element_by_tag_name()`, `driver.find_element_by_class_name()`, `driver.find_element_by_id()`, `driver.find_element_by_xpath()`,
#### `driver.find_element_by_*().text`
```python
df.loc[index, "배정초"]=driver.find_element_by_xpath("//\*[@id='detailContents5']/div/div[1]/div[1]/h5").text
```
#### `driver.find_element_by_*().get_attribute()`
```python
driver.find_element_by_xpath("//*[@id='detailTab" +str(j) + "']").get_attribute("text")
```
#### `driver.find_element_by_*().click()`
#### `driver.find_element_by_*().clear()`
```python
driver.find_element_by_xpath('//*[@id="searchboxinput"]').clear()
```
#### `driver.find_element_by_*().send_keys()`
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
### `driver.execute_script()`
```python
for j in [4,3,2]:
    button = driver.find_element_by_xpath("//\*[@id='detailTab"+str(j)+"']")
    driver.execute_script("arguments[0].click();", button)
```
### `driver.implicitly_wait()`
```python
driver.implicitly_wait(1)
```
### `driver.current_url`
### `driver.save_screenshot()`
```python
driver.save_screenshot(screenshot_title)
```
## `WebDriverWait()`
### `WebDriverWait().until()`
```python
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
```
```python
WebDriverWait(driver, 3).until(EC.presence_of_element_located((By.XPATH, "//\*[@id='detailContents5']/div/div[1]/div[1]/h5")))
```
- `By.ID`, `By.XPATH`
## `ActionChains()`
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
### `actions.click()`, `actions.double_click()`

# `urllib`
```python
import urllib
```
## `urllib.request`
### `urllib.request.urlopen()`
```python
xml = urllib.request.urlopen(full_url).read().decode("utf-8")
```
### `urllib.request.urlretrieve()`
```python
urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt", filename="ratings_train.txt")
```
- 해당 URL에 연결된 파일을 다운로드합니다.

# `urllib3`
```python
import urllib3
```
## `urllib3.PoolManager()`
### `urllib3.PoolManager().request()`
```python
urllib3.PoolManager().request("GET", url, preload_content=False)
```

# `pathlib`
```python
import pathlib
```
## `pathlib.Path()`
```python
data_dir = pathlib.Path(data_dir)
```

# `requests`
```python
import requests
```
## `requests.get()`
```python
req = requests.get("https://github.com/euphoris/datasets/raw/master/imdb.zip")
```
### `req.content`

# `wget`
````python
import wget
````
## `wget.download()`

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

# `cv2`
```python
!pip install opencv-python
```
```python
import cv2
```
## `cv2.waitKey()`
```python
k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
```
## `cv2.VideoCapture()`
```python
cap = cv2.VideoCapture(0)
```
## `cv2.destroyAllWindows()`
## `cv2.rectangle()`
```python
for i, rect in enumerate(rects_selected):
    cv2.rectangle(img=img, pt1=(rect[0], rect[1]), pt2=(rect[0]+rect[2], rect[1]+rect[3]), color=(0, 0, 255), thickness=2)
```
## `cv2.circle()`
```python
for i, rect in enumerate(rects_selected):
    cv2.circle(img, (rect[0]+1, rect[1]-12), 12, (0, 0, 255), 2))
```
## `cv2.getTextSize()`
```python
(text_width, text_height), baseline = cv2.getTextSize(text=label, fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                                                  fontScale=font_scale, thickness=bbox_thick)
```
## `cv2.puttext()`
```python
cv2.putText(img=img, text=label, org=(x1, y1-4), fonFace=cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=font_scale, color=text_colors, thickness=bbox_thick, lineType=cv2.LINE_AA)
```
## `cv2.resize()`
```python
img_resized = cv2.resize(img, dsize=(640, 480), interpolation=cv2.INTER_AREA)
```
- `dsize` : (new_width, new_height)
## `cv2.cvtColor()`
```python
img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
```
## `cv2.imread()`
```python
img = cv2.imread("300.jpg")
```
## `cv2.imwrite()`
```python
cv2.imwrite("/content/drive/My Drive/Computer Vision/fire hydrants.png", ori_img)
```
## `cv2.imshow()`
```python
cv2.imshow("img_resized", img_resized)
```
## `cv2.findContours()`
```python
mask = cv2.inRange(hsv,lower_blue,upper_blue)
contours, hierachy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
```
## `cv2.TERM_CRITERIA_EPS`, `cv2.TERM_CRITERIA_MAX_ITER`
```python
criteria = (type, max_iter, epsilon)
```
## `CV2.KMEANS_RANDOM_CENTERS`
```python
flags = cv2.KMEANS_RANDOM_CENTERS
```
- 초기 중심점을 랜덤으로 설정.
```python
compactness, labels, centers = cv2.kmeans(z, 3, None, criteria, 10, flags)
```

# `selectivesearch`
## `selectivesearch.selective_search()`
```python
_, regions = selectivesearch.selective_search(img_rgb, scale=100, min_size=2000)
```
```python
img_recs = cv2.rectangle(img=img_rgb_copy, pt1=(rect[0], rect[1]),
                                 pt2=(rect[0]+rect[2], rect[1]+rect[3]),
                                 color=green_rgb, thickness=2)
```

# `datetime`
## `datetime.today()`
## `datetime.datetime()`, `datetime.date()`
```python
datetime.datetime(2018, 5, 19)
```
- Require three parameters in sequence to create a date; year, month, day
### `datetime.datetime.now()`
### `datetime.datetime.strptime()`
```python
datetime.datetime.strptime(<<Date String>>, <<Format>>)
```
- Returns a datetime corresponding to <<Date String>>, parsed according to <<Format>>.
- `format`:
	- `"%Y"`: Year with century as a decimal number.
	- `"%y"`: Year without century as a zero-padded decimal number.
	- `"%m"`: Month as a zero-padded decimal number.
	- `"%d"`: Day of the month as a zero-padded decimal number.
	- `"%H"`: Hour (24-hour clock) as a zero-padded decimal number.
	- `"%I"`: Hour (12-hour clock) as a zero-padded decimal number.
	- `"%M"`: Minute as a zero-padded decimal number.
	- `"%S"`: Second as a zero-padded decimal number
	- `"%A"`: Weekday as locale’s full name.
	- `"%B"`: Month as locale’s full name.
	- `"%b"`: Month as locale’s abbreviated name.
### `datetime.datetime.strftime()`
```python
datetime.datetime.strftime(format)
```
- Returns a string representing the date and time, controlled by an explicit format string
```python
gby_month["ym"] = gby_month["date_created"].apply(lambda x:datetime.datetime.strftime(x, "%m"))
```
## `datetime.timedelta`
```python
day = start + datetime.timedelta(days=1)
```
### `datetime.timedelta.days`
### `datetime.timedelta.total_seconds()`
```python
(t2 - t1).total_seconds()
```

# `dateutil`
## `dateutil.relativedelta`
### `relativedelta`
```python
from dateutil.relativedelta import relativedelta
```
```python
data["년-월"] = data["년-월"].apply(lambda x:x + relativedelta(months=1) - datetime.timedelta(days=1))
```

# `time`
```python
import time
```
## `time.time()`
```python
time_before=round(time.time())
```
```python
print("{}초 경과".format(round(time.time())-time_before))
```
## `time.localtime()`
## `time.strftime()`
```python
time.strftime("%Y%m%d", time.localtime(time.time()))
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

# `mapboxgl`
## `mapboxgl.viz`
```python
from mapboxgl.viz import *
```
### `CircleViz()`
```python
viz = CircleViz(data=geo_data, access_token=token, center=[127.46,36.65], zoom=11, radius=2, stroke_color="black", stroke_width=0.5)
```
### `GraduatedCircleViz()`
```python
viz = GraduatedCircleViz(data=geo_data, access_token=token, height="600px", width="600px", center=(127.45, 36.62), zoom=11, scale=True, legend_gradient=True, add_snapshot_links=True, radius_default=4, color_default="black", stroke_color="black", stroke_width=1, opacity=0.7)
```
### `viz.style`
```python
viz.style = "mapbox://styles/mapbox/outdoors-v11"
```
- (`"mapbox://styles/mapbox/streets-v11"`, `"mapbox://styles/mapbox/outdoors-v11"`, `"mapbox://styles/mapbox/light-v10"`
- `"mapbox://styles/mapbox/dark-v10"`, `"mapbox://styles/mapbox/satellite-v9"`, `"mapbox://styles/mapbox/satellite-streets-v11"`, `"mapbox://styles/mapbox/navigation-preview-day-v4"`, `"mapbox://styles/mapbox/navigation-preview-night-v4"`, `"mapbox://styles/mapbox/navigation-guidance-day-v4"`, `"mapbox://styles/mapbox/navigation-guidance-night-v4"`)
### `viz.show()`
### `viz.create_html()`
```python
with open("D:/☆디지털혁신팀/☆실거래가 분석/☆그래프/1km_store.html", "w") as f:
    f.write(viz.create_html())
```
## `mapboxgl.utils`
```python
from mapboxgl.utils import df_to_geojson, create_color_stops, create_radius_stops
```
### `DataFrame.to_geojson()`
```python
geo_data = df_to_geojson(df=df, lat="lat", lon="lon")
```
### `viz.create_color_stops()`
```python
viz.color_property = "error"
viz.color_stops = create_color_stops([0, 10, 20, 30, 40, 50], colors="RdYlBu")
```
### `viz.create_radius_stops()`
```python
viz.radius_property = "errorl"
viz.radius_stops = create_radius_stops([0, 1, 2], 4, 7)
```

# `matplotlib`
```python
import matplotlib as mpl
```
## `mpl.font_manager`
### `mpl.font_manager.FontProperties()`
#### `mpl.font_manager.FontProperties().get_name()`
```python
fpath = "C:/Windows/Fonts/malgun.ttf"
font_name = mpl.font_manager.FontProperties(fname=fpath).get_name()
```
## `mpl.rc()`
```python
mpl.rc("font", family=font_name)
```
- `family`: (`"NanumBarunGothic"`)
```python
mpl.rc("axes", unicode_minus=False)
```
## `matplotlib.pyplot`
```python
import matplotlib.pyplot as plt
```
### `plt.setp()`
```python
plt.setp(obj=ax, yticks=ml_mean_gr_ax["le"], yticklabels=ml_mean_gr_ax.index)
```
### `plt.style.use()`
- (`"default"`, `"dark_background"`)
### `plt.subplot()`
```python
for i in range(9):
	ax = plt.subplot(3, 3, i + 1)
```
### `plt.subplots()`
```python
fig, axes = plt.subplots(2, 1, figsize=(16, 12))
```
### `plt.show()`
#### `fig.colorbar()`
```python
cbar = fig.colorbar(ax=ax, mappable=scatter)
```
#### `fig.set_size_inches()`
```python
fig.set_size_inches(10, 10)
```
##### `cbar.set_label()`
```python
cbar.set_label(label="전용면적(m²)", size=15)
```
#### `plt.savefig()`, `fig.savefig()`
```python
fig.savefig("means_plot_200803.png", bbox_inches="tight")
```
#### `fig.tight_layout()`
#### `ax.imshow()`
```python
a`x.imshow(image.numpy().reshape(3,3), cmap="Greys")
```
#### `ax.set()`
```python
ax.set(title="Example", xlabel="xAxis", ylabel="yAxis", xlim=[0, 1], ylim=[-0.5, 2.5], xticks=data.index, yticks=[1, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3])
```
- `title`
- `xlabel`, `ylabel`
- `xlim`, `ylim`
- `xticks`, `yticks`
#### `ax.set_title()`, `plt.title()`
- `size`: (float)
#### `ax.set_xlabel()`, `ax.set_ylabel()`
```python
ax.set_xlabel("xAxis", size=15)
```
#### `ax.set_xlim()`, `ax.set_ylim()`
```python
ax.set_xlim([1, 4])
```
#### `ax.set_xticks()`, `ax.set_yticks()`
#### `ax.axes`
#### `ax.axis()`
```python
ax.axis([2, 3, 4, 10])
```
```python
ax.axis("off")
```
##### `ax.xaxis.set_visible()`, `ax.yaxis.set_visible()`
```python
ax.xaxis.set_visible(False)
```
##### `ax.xaxis.set_label_position()`, `ax.yaxis.set_label_position()`
```python
ax.xaxis.set_label_position("top")
```
##### `ax.xaxis.set_ticks_position()`, `ax.yaxis.set_ticks_position()`
```python
ax.yaxis.set_ticks_position("right")
```
- (`"top"`, `"bottom"`, `"left"`, `"right"`)
##### `ax.xaxis.set_major_formatter()`, `ax.yaxis.set_major_formatter()`
```python
ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter("{x:,.0f}"))
```
#### `ax.invert_xaxis()`, `ax.invert_yaxis()`
#### `ax.tick_params()`
```python
ax.tick_params(axis="x", labelsize=20, labelcolor="red", labelrotation=45, grid_linewidth=3)
```
#### `ax.set_xticks()`, `ax.set_yticks()`
```python
ax.set_yticks(np.arange(1, 1.31, 0.05))
```
- 화면에 표시할 눈금을 설정합니다.
#### `ax.legend()`
```python
ax.legend(fontsize=14, loc="best")
```
- `loc`: (`"best"`, `"upper right"`)
#### `ax.grid()`
```python
ax.grid(axis="x", color="White", alpha=0.3, linestyle="--", linewidth=2)
```
#### `ax.plot()`, `DataFrame.plot.line()`, `Series.plot.line()`
```python
fig = raw_all[["count"]].rolling(24).mean().plot.line(ax=axes, lw=3, fontsize=20, xlim=("2012-01-01", "2012-06-01"), ylim=(0,  1000))
```
- `ls`: (`"-"`, `"--"`, `"-."`, `":"`)
- `lw`
- `c`: Specifies a color.
- `label`
- `fontsize`
- `title`
- `legend`: (bool)
- `xlim`, `ylim`
- Reference: https://matplotlib.org/stable/api/_as_gen/matplotlib.lines.Line2D.html#matplotlib.lines.Line2D
### `DataFrame.plot.pie()`, `Series.plot.pie()`
```python
cnt_genre.sort_values("movie_id", ascending=False)["movie_id"].plot.pie(ax=ax, startangle=90, legend=True)
```
#### `ax.scatter()`, `DataFrame.plot.scatter()`
```python
ax.scatter(gby["0.5km 내 교육기관 개수"], gby["실거래가"], s=70, c=gby["전용면적(m²)"], cmap="RdYlBu", alpha=0.7, edgecolors="black", linewidth=0.5)
```
```python
data[data.workingday == 0].plot.scatter(y="count", x="hour", c="temp", grid=False, figsize=(12, 5), cmap="Blues")
```
#### `ax.bar()`, `DataFrame.plot.bar()`, `Series.plot.bar()`
```python
ax.bar(x=nby_genre.index, height=nby_genre["movie_id"])
```
```python
data["label"].value_counts().plot.bar()
```
#### `ax.barh()`, `DataFrame.plot.barh()`, `Series.plot.barh()`
```python
ax.barh(y=ipark["index"], width=ipark["가경4단지 84.8743A"], height=0.2, alpha=0.5, color="red", label="가경4단지 84.8743A", edgecolor="black", linewidth=1)
```
#### `ax.hist()`, `DataFrame.plot.hist()`, `Series.plot.hist()`
```python
ax.hist(cnt_genre["genre"], bins=30)
```
```python
raw_data.hist(bins=20, grid=True, figsize=(16,12))
```
#### `ax.boxplot()`, `DataFrame.boxplot()`, `Series.boxplot()`
```python
raw_data.boxplot(column="count", by="season", grid=False, figsize=(12,5))
```
#### `ax.axhline()`, `ax.axvline()`
```python
fig = ax.axhline(y=mean, color="r", ls=":", lw=2)
```
#### `hlines()`, `vlines()`
#### `ax.text()`
```python
for _, row in ml_gby_ax.iterrows():
    ax.text(y=row["le"]-0.18, x=row["abs_error"], s=round(row["abs_error"], 1), va="center", ha="left", fontsize=10)
```
#### `ax.fill_between()`
```python
ax.fill_between(x, y1, y2, ...)
```

# `seaborn`
```python
import seaborn as sb
```
### `sb.set()`
- `palette`: (`"muted"`)
- `color_codes`: If `True` and `palette` is a seaborn palette, remap the shorthand color codes (e.g. `"b"`, `"g"`, `"r"`, etc.) to the colors from this palette.
- `font_scale`: (float)
### `sb.lmplot()`
```python
sb.lmplot(data=resid_tr.iloc[1:], x="idx", y="resid", fit_reg=True, line_kws={"color": "red"}, size=5.2, aspect=2, ci=99, sharey=True)
```
- `data`: (DataFrame)
- `fit_reg`: (bool) If `True`, estimate and plot a regression model relating the x and y variables.
- `ci`: (int in [0, 100] or None, optional) Size of the confidence interval for the regression estimate. This will be drawn using translucent bands around the regression line. The confidence interval is estimated using a bootstrap; for large datasets, it may be advisable to avoid that computation by setting this parameter to None.
- `aspect`: Aspect ratio of each facet, so that aspect\*height gives the width of each facet in inches.
### `sb.distplot()`
```python
fig, axes = plt.subplots(1, 1, figsize=(7, 5))
fig = sb.distplot(ax=axes, a=resid_tr["resid"].iloc[1:], norm_hist=True, fit=stats.norm)
```
- `a`: (Series, 1d-Array, or List)
- `norm_hist`: (bool, optional) If `True`, the histogram height shows a density rather than a count. This is implied if a KDE or fitted density is plotted.
### `sb.scatterplot()`
```python
sb.scatterplot(ax=ax, data=df, x="ppa", y="error", hue="id", hue_norm=(20000, 20040), palette="RdYlGn", s=70, alpha=0.5)
```
### `sb.lineplot()`
```python
ax = sb.lineplot(x=data.index, y=data["ppa_ratio"], linewidth=3, color="red", label="흥덕구+서원구 아파트 평균")
```
### `sb.barplot()`
```python
sb.barplot(ax=ax, x=area_df["ft_cut"], y=area_df[0], color="brown", edgecolor="black", orient="v")
```
### `sb.countplot()`
```python
sb.countplot(ax=ax, data=cmts202011, x="dep")
```
- DataFrame
```python
sb.countplot(ax=ax, x=label_train)
```
- `x`: (Array, List of Arrays)
### `sb.replot()`
```python
ax = sb.replot(x="total_bill", y="tip", col="time", hue="day", style="day", kind="scatter", data=tips)
```
### `sb.kedplot()`
```python
sb.kdeplot(ax=ax, data=data["ppa_root"])
```
### `sb.stripplot()`
```python
ax = sb.stripplot(x=xx, y=yy, data=results_order, jitter=0.4, edgecolor="gray", size=4)
```
### `sb.pairtplot()`
```python
ax = sb.pairplot(data_for_corr)
```
### `sb.heatmap()`
- Reference: http://seaborn.pydata.org/generated/seaborn.heatmap.html
```python
sb.heatmap(ax=ax, data=gby_occup_genre, annot=True, annot_kws={"size": 10}, fmt=".2f", linewidths=0.2, center=3, cmap="RdBu")
```

# `networkx`
```python
improt networks as nx
```
## `nx.Graph()`
```python
g = nx.Graph()
```
## `nx.DiGraph()`
## `nx.circular_layout()`
```python
pos = nx.circular_layout(g)
```
## `nx.draw_networks_nodex()`
```python
nx.draw_networkx_nodes(g, pos, node_size=2000)
```
## `nx.draw_networkx_edges()`
```python
nx.draw_networkx_edges(g, pos, width=weights)
```
## `nx.draw_networkx_labels()`
```python
nx.draw_networkx_labels(g, pos, font_family=font_name, font_size=11)
```
## `nx.draw_shell()`
```python
nx.draw_shell(g, with_labels=False)
```
### `g.add_nodes_from()`
```python
g.add_nodes_from(set(df.index.get_level_values(0)))
```
### `g.add_edge()`
```python
for _, row in df.iterrows():
    g.add_edge(row.name[0], row.name[1], weight=row["cowork"]/200)
```
### `g.edges()`
```python
weights = [cnt["weight"] for (_, _, cnt) in g.edges(data=True)]
```

# `wordcloud`
## `WordCloud`
```python
from wordcloud import WordCloud
```
```python
wc = WordCloud(font_path="C:/Windows/Fonts/HMKMRHD.TTF", relative_scaling=0.2, background_color="white", width=1600, height=1600, max_words=30000, mask=mask, max_font_size=80, background_color="white")
```
### `wc.generate_from_frequencies()`
```python
wc.generate_from_frequencies(words)
```
### `wc.generate_from_text`
### `wc.recolor()`
```python
wc.recolor(color_func=img_colors)
```
### `wc.to_file()`
```python
wc.to_file("test2.png")
```
## `ImageColorGenerator`
```python
from wordcloud import ImageColorGenerator
```
```python
img_arr = np.array(Image.open(pic))
img_colors = ImageColorGenerator(img_arr)
img_colors.default_color=[0.6, 0.6, 0.6]
```
## `STOPWORDS`
```python
from wordcloud import STOPWORDS
```

# `random`
```python
import random
```
## `random.seed()`
## `random.random()`
- Returns a random number in [0, 1).
## `random.sample()`
```python
names = random.sample(list(set(data.index)), 20)
```
## `random.shuffle()`
- In-place function
## `random.choice()`

# `re`
```python
 import re
```
- The meta-characters which do not match themselves because they have special meanings are: `.`, `^`, `$`, `*`, `+`, `?`, `{`, `}`, `[`, `]`, `(`, `)`, `\`, `|`.
- `.`: Match any single character except newline.
- `\n`: Newline
- `\r`: Return
- `\t`: Tab
- `\f`: Form
- `\w`, `[a-zA-Z0-9_]`: Match any single "word" character: a letter or digit or underbar. 
- `\W`, `[^a-zA-Z0-9_]`: Match any single non-word character.
- `\s`, `[ \n\r\t\f]`: Match any single whitespace character(space, newline, return, tab, form).
- `\S`: Match any single non-whitespace character.
- `[ㄱ-ㅣ가-힣]`: 어떤 한글
- `\d`, `[0-9]`: Match any single decimal digit.
- `\D`, `[^0-9]`: Match any single non-decimal digit.
- `\*`: 0개 이상의 바로 앞의 character(non-greedy way)
- `\+`: 1개 이상의 바로 앞의 character(non-greedy way)
- `\?`: 1개 이하의 바로 앞의 character
- `{m, n}`: m개~n개의 바로 앞의 character(생략된 m은 0과 동일, 생략된 n은 무한대와 동일)(non-greedy way)
- `{n}`: n개의 바로 앞의 character
- `^`: Match the start of the string.
- `$`: Match the end of the string.
## `re.search()`
```python
re.search(<<Pattern>>, <<String>>)
```
- Scan through string looking for the first location where the regular expression pattern produces a match, and return a corresponding match object. Return None if no position in the string matches the pattern; note that this is different from finding a zero-length match at some point in the string.
## `re.match()`
```python
re.match(<<Pattern>>, <<String>>)
```
- If zero or more characters at the beginning of string match the regular expression pattern, return a corresponding match object. Return None if the string does not match the pattern; note that this is different from a zero-length match.
### `re.search().group()`, `re.match().group()`
```python
re.search(r"(\w+)@(.+)", "test@gmail.com").group(0) #test@gmail.com
re.search(r"(\w+)@(.+)", "test@gmail.com").group(1) #test
re.search(r"(\w+)@(.+)", "test@gmail.com").group(2) #gmail.com
```
```python
views["apt_name"] = views["pageTitle"].apply(lambda x:re.search(r"(.*)\|(.*)\|(.*)", x).group(2) if "|" in x else x)
```
## `re.findall()`
```python
re.findall(rf"[ ][a-z]{{{n_wc}}}{chars}[ ]", words)
```
- Return all non-overlapping matches of pattern in string, as a list of strings. The string is scanned left-to-right, and matches are returned in the order found. If one or more groups are present in the pattern, return a list of groups; this will be a list of tuples if the pattern has more than one group. Empty matches are included in the result.
## `re.split()`
- `maxsplit`
## `re.sub()`
```python
expr = re.sub(rf"[0-9]+[{s}][0-9]+", str(eval(calc)), expr, count=1)
```
- count=0 : 전체 치환
## `re.compile()`
```python
p = re.compile(".+\t[A-Z]+")
```
- 이후 `p.search()`, `p.match()` 등의 형태로 사용합니다.

# `requests`
```python
import requests
```
## `requests.get()`
```python
req = requests.get(url)
```

# `statsmodels`
## `statsmodels.stats`
### `statsmodels.stats.outliers_influence`
#### `variance_inflation_factor()`
```python
from statsmodels.stats.outliers_influence import variance_inflation_factor
```
```python
vif = pd.DataFrame()
vif["vif"] = [variance_inflation_factor(x_tr.values, i) for i in range(x_tr.shape[1])]
vif["feat"] = x_tr.columns
```
## `statsmodels.api`
```python
import statsmodels.api as sm
```
### `sm.qqplot()`
```python
fig = sm.qqplot(ax=axes[0], data=data["value"], fit=True, line="45")
```
### `sm.tsa`
#### `sm.tsa.stattools`
### `sm.OLS()`
#### `sm.OLS().fit()`
```python
sm.OLS(y_tr, x_tr).fit()
```
##### `sm.OLS().fit().summary()`
- `coef`: The measurement of how change in the independent variable affects the dependent variable. Negative sign means inverse relationship. As one rises, the other falls.
- `R-squared`, `Adj. R-squared`
- `P>|t|`: p-value. Measurement of how likely the dependent variable is measured through the model by chance. That is, there is a (p-value) chance that the independent variable has no effect on the dependent variable, and the results are produced by chance. Proper model analysis will compare the p-value to a previously established threshold with which we can apply significance to our coefficient. A common threshold is 0.05.
- `Durbin-Watson`: Durbin-Watson statistic.
- `Skew`: Skewness.
- `Kurtosis`: Kurtosis.
- `Cond. No.`: Condition number of independent variables.
##### `sm.OLS().fit().rsquared_adj`
- Return `Adj. R-squared` of the independent variables.
##### `sm.OLS().fit().fvalue`
##### `sm.OLS().fit().f_pvalue`
##### `sm.OLS().fit().aic`
##### `sm.OLS().fit().bic`
##### `sm.OLS().fit().params`
- Return `coef`s of the independent variables.
##### `sm.OLS().fit().pvalues`
- Return `P>|t|`s of the independent variables.
##### `sm.OLS().fit().predict()`
### `sm.graphics`
#### `sm.graphics.tsa`
##### `sm.graphics.tsa.plot_acf()`, `sm.graphics.sta.plot_pacf()`
```python
fig = sm.graphics.tsa.plot_acf(ax=axes[0], x=resid_tr["resid"].iloc[1:], lags=100, use_vlines=True)
```
- `title`
### `sm.stats`
#### `sm.stats.diagnostic`

# `scipy`
```python
import scipy
```
## `scipy.sparse`
## `stats`
```python
from scipy import stats
```
### `stats.norm`
### `stats.beta`
#### `stats.beta.pdf()`
```python
ys = stats.beta.pdf(xs, a, b)
```
### `stats.shapiro()`
```python
Normality = pd.DataFrame([stats.shapiro(resid_tr["resid"])], index=["Normality"], columns=["Test Statistic", "p-value"]).T
```
- Return test statistic and p-value.
### `stats.boxcox_normalplot()`
```python
x, y = stats.boxcox_normplot(data["value"], la=-3, lb=3)
```
### `stats.boxcox()`
```python
y_trans, l_opt = stats.boxcox(data["value"])
```

# `openpyxl`
```python
import openpyxl
```
## `openpyxl.Workbook()`
```python
wb = openpyxl.Workbook()
```
## `openpyxl.load_workbook()`
```python
wb = openpyxl.load_workbook("D:/디지털혁신팀/태블로/HR분석/FINAL/★직급별인원(5년)_본사현장(5년)-태블로적용.xlsx")
```
## `openpyxl.worksheet`
### `openpyxl.worksheet.formula`
#### `ArrayFormula`
```python
from openpyxl.worksheet.formula import ArrayFormula
```
```python
f = ArrayFormula("E2:E11", "=SUM(C2:C11*D2:D11)")
```
### `wb[]`
### `wb.active`
```python
ws = wb.active
```
### `wb.worksheets`
```python
ws = wb.worksheets[0]
```
### `wb.sheetnames`
### `wb.create_sheet()`
```python
wb.create_sheet("Index_sheet")
```
#### `ws[]`
#### `ws.values()`
```python
df = pd.DataFrame(ws.values)
```
#### `ws.insert_rows()`, `ws.insert_cols()`, `ws.delete_rows()`, `ws.delete_cols()`
#### `ws.append()`
```python
content = ["민수", "준공분", "거제2차", "15.06", "18.05", "1279"]
ws.append(content)
```
#### `ws.merge_cells()`, `ws.unmerge_cells()`
```python
ws.merge_cells("A2:D2")
```
### `wb.sheetnames`
### `wb.save()`
```python
wb.save("test.xlsx")
```
### `wb.active`
```python
sheet = wb.active
```
### `wb.save()`
```python
wb.save("test.xlsx")
```
#### `sheet.append()`
```python
content = ["민수", "준공분", "거제2차", "15.06", "18.05", "1279"]
sheet.append(content)
```
#### `sheet[]`
```python
sheet["H8"] = "=SUM(H6:H7)"
```

# `pptx`
## `Presentation()`
```python
from pptx import Presentation
```
```python
prs = Presentation("sample.pptx")
```
### `prs.slides[].shapes[].text_frame.paragraphs[].text`
```python
prs.slides[<<Index of the Slide>>].shapes[<<Index of the Text Box>>].text_frame.paragraphs[<<Index of the Text in the Text Box>>].text
```
### `prs.slides[].shapes[].text_frame.paragraphs[].font`
#### `prs.slides[].shapes[].text_frame.paragraphs[].text.name`
```python
prs.slides[a].shapes[b].text_frame.paragraphs[c].font.name = "Arial"
```
#### `prs.slides[].shapes[].text_frame.paragraphs[].text.size`
```python
prs.slides[a].shapes[b].text_frame.paragraphs[c].font.size = Pt(16)
```
### `prs.save()`
```python
prs.save("파일 이름")
```

# `copy`
## `copy.copy()`
## `copy.deepcopy()`

# `csv`
```python
import csv
```
## `csv.QUOTE_NONE`
```python
subws = pd.read_csv("imdb.vocab", sep="\t", header=None, quoting=csv.QUOTE_NONE)
```

# `shutil`
```python
import shutil
```
## `shutil.copyfile()`
```python
shutil.copyfile("./test1/test1.txt", "./test2.txt")
```
## `shutil.copyfileobj()`
```python
shutil.copyfileobj(urllib3.PoolManager().request("GET", url, preload_content=False), open(file_dir, "wb"))
```

# `logging`
## `logging.basicConfig()`
```python
logging.basicConfig(level=logging.ERROR)
```

# `IPython`
## `IPython.display`
### `set_matplotlib_formats`
```python
from IPython.display import set_matplotlib_formats
```
```python
set_matplotlib_formats("retina")
```
- font를 선명하게 표시

# `itertools`
## `combinations()`, `permutations()`, `combinations_with_replacement()`
```python
from itertools import combinations, permutations
```
```python
movies = {a | b for a, b in combinations(movie2sup.keys(), 2)}
```
## `product()`
- `repeat`
```python
for i in product(range(3), range(3), range(3)):
    print(i)
```

# `collections`
## `Counter()`
```python
from collections import Counter
```
```python
word2cnt = Counter(words)
```
- lst의 원소별 빈도를 나타내는 dic을 반환합니다.
### `Counter[]`
### `Counter().values()`
```python
sum(Counter(nltk.ngrams(cand.split(), 2)).values())
```
### `Counter().most_common()`
## `deque()`
```python
from collections import deque
```
```python
dq = deque("abc")
```
- `maxlen`
### `dq[]`
```python
dq[2] = "d"
```
### `dq.append()`
### `dq.appendleft()`
### `dq.pop()`
### `dq.popleft()`
### `dq.extend()`
### `dq.extendleft()`
### `dq.remove()`

# `functools`
## `reduce()`
```python
from functools import reduce
```
```python
reduce(lambda acc, cur: acc + cur["age"], users, 0)
```
```python
reduce(lambda acc, cur: acc + [cur["mail"]], users, [])
```

# `platform`
```python
import platform
```
## `platform.system()`
```python
path = "C:/Windows/Fonts/malgun.ttf"
if platform.system() == "Darwin":
    mpl.rc("font", family="AppleGothic")
elif platform.system() == "Windows":
    font_name = mpl.font_manager.FontProperties(fname=path).get_name()
    mpl.rc('font', family=font_name)
```
- Returns the system/OS name, such as `'Linux'`, `'Darwin'`, `'Java'`, `'Windows'`. An empty string is returned if the value cannot be determined.

# `pprint`
## `pprint()`
```python
from pprint import pprint
```

# `google`
## `google.colab`
### `google.colab.patches`
#### `cv2_imshow()`
```python
from google.colab.patches import cv2_imshow
```

# `colorsys`
## `colorsys.hsv_to_rgb()`
```python
hsv_tuples = [(idx/n_clss, 1, 1) for idx in idx2cls.keys()]
colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
colors = list(map(lambda x: (int(x[0]*255), int(x[1]*255), int(x[2]*255)), colors))
```

# `zipfile`
```python
import zipfile
```
## `zipfile.ZipFile()`
```python
with zipfile.ZipFile("spa-eng.zip", "r") as f:
    file = f.open("spa-eng/spa.txt")
    data = pd.read_csv(file, names=["eng", "spa"], sep="\t")
```
### `zipfile.ZipFile().extractall()`
```python
zipfile.ZipFile("glove.6B.zip").extractall(cur_dir)
```

# `lxml`
## `etree`
```python
from lxml import etree
```
### `etree.parse()`
```python
with zipfile.ZipFile("ted_en-20160408.zip", "r") as z:
	target_text = etree.parse(z.open("ted_en-20160408.xml", "r"))
```

# `os`
```python
import os
```
## `os.getcwd()`
```python
cur_dir = os.getcwd()
```
## `os.makedirs()`
```python
os.makedirs(ckpt_dir, exist_ok=True)
```
## `os.chdir()`
## `os.environ`
## `os.pathsep`
```python
os.environ["PATH"] += os.pathsep + "C:\Program Files (x86)/Graphviz2.38/bin/"
```
## `os.path`
### `os.path.join()	`
```python
os.path.join("C:\Tmp", "a", "b")
```
### `os.path.exists()`
```python
if os.path.exists("C:/Users/5CG7092POZ/train_data.json"):
```
## `os.path.dirname()`
```python
os.path.dirname("C:/Python35/Scripts/pip.exe")
# >>> 'C:/Python35/Scripts'
```
- 경로 중 디렉토리명만 얻습니다.

# `glob`
## `glob.glob()`
```python
path = "./DATA/전체"
filenames = glob.glob(path + "/*.csv")
```

# `pickle`
```python
import pickle as pk
```
## `pk.dump()`
```python
with open("filename.pkl", "wb") as f:
	pk.dump(list, f)
```
## `pk.load()`
```python
with open("filename.pkl", "rb") as f:
	data = pk.load(f)
```
- 한 줄씩 load

# `json`
```python
import json
```
## `json.dump()`
```python
with open(path, "w", encoding="utf-8") as f:
    json.dump(train_data, f, ensure_ascii=False, indent="\t")
```
```python
with open(f"{model_path}_hist", "w") as f:
	json.dump(hist.history, f)
```
## `json.load()`
```python
with open(path, "r", encoding="utf-8") as f:
    train_data = json.load(f)
```
```python
with open(f"{model_path}_hist", "r") as f:
	hist.history = json.load(f)
```

# `Image`
```python
from PIL import Image
```
## `Image.open()`
```python
img = Image.open("20180312000053_0640 (2).jpg")
```
### `img.size`
### `img.save()`
### `img.thumbnail()`
```python
img.thumbnail((64, 64))
```
### `img.crop()`
```python
img_crop = img.crop((100, 100, 150, 150))	
```
### `img.resize()`
```python
img = img.resize((600, 600))
```
### `img.convert()`
```python
img.convert("L")
```
- (`"RGB"`, `"RGBA"`, `"CMYK"`, `"L"`, `"1"`)
### `img.paste()`
```python
img1.paste(img2, (20,20,220,220))
```
- img2.size와 동일하게 두 번째 parameter 설정.	
## `Image.new()`
```python
mask = Image.new("RGB", icon.size, (255, 255, 255))
```

# `datasketch`
## `MinHash`
```python
from datasketch import MinHash
```
```python
mh = MinHash(num_perm=128)
```
- MinHash는 각 원소 별로 signature를 구한 후, 각 Signature 중 가장 작은 값을 저장하는 방식입니다. 가장 작은 값을 저장한다 해서 MinHash라고 불립니다.
### `mh.update()`
```python
for value in set_A:
    mh.update(value.encode("utf-8"))
```
### `mh.hashvalues`

# `redis`
## `Redis`
```python
from redis import Redis
```
```python
rd = Redis(host="localhost", port=6379, db=0)
```
### `rd.set()`
```python
rd.set("A", 1)
```
### `rd.delete()`
```python
rd.delete("A")
```
### `rd.get()`
```python
rd.get("A")
```

# `google_drive_downloader`
```python
!pip install googledrivedownloader
```
## `GoogleDriveDownloader`
```python
from google_drive_downloader import GoogleDriveDownloader as gdd
```
### `gdd.download_file_from_google_drive()`
```python
gdd.download_file_from_google_drive(file_id="1uPjBuhv96mJP9oFi-KNyVzNkSlS7U2xY", dest_path="./movies.csv")
```

# `sys`
```python
import sys
```
## `sys.maxsize()`
## `sys.path`
```python
sys.path.append("c:/users/82104/anaconda3/envs/tf2.3/lib/site-packages")
```
## `sys.stdin`
### `sys.stdin.readline()`
```python
cmd = sys.stdin.readline().rstrip()
```
## `sys.getrecursionlimit()`
## `sys.setrecursionlimit()`
```python
sys.setrecursionlimit(10**9)
```

# `tqdm`
## `tqdm.notebook`
### `tqdm`
```python
from tqdm.notebook import tqdm
```
- for Jupyter Notebook
## `tqdm.auto`
### `tqdm`
```python
from tqdm.auto import tqdm
```
- for Google Colab
## `tqdm.pandas()`
- `DataFrame.progress_apply()`를 사용하기 위해 필요합니다.

# `warnings`
```python
import warnings
```
## `warnings.filterwarnings()`
```python
warnings.filterwarnings("ignore", category=DeprecationWarning)
```