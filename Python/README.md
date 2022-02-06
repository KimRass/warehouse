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
- ***"Subscriptable" means that the object contain, or can contain, other objects.***
- The subscripting is nothing but indexing in Python.
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
### `SyntaxError: positional argument follows keyword argument`
- Source: https://www.geeksforgeeks.org/how-to-fix-syntaxerror-positional-argument-follows-keyword-argument-in-python/
- There are two kind of arguments, namely, keyword and positional. As the name suggests, the keyword argument is identified by a function based on some key whereas the positional argument is identified based on its position in the function definition.
- Positional arguments can be written in the beginning before any keyword argument is passed.
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
## `open(file, mode, encoding)`
```python
with open() as f:
    ...
```
- `mode`: (`"r"`, `"rb"`, `"w"`, `"wb"`, ...) 

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
- Subscriptable.
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
- Not subscriptable.
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
- Subscriptable.
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
- Subscriptable.
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
data.columns = data.columns.str.lower()
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
## `pd.read_csv([thousands=","], [float_precision], [skiprows], [error_bad_lines], [index_col], [sep], [names], [parse_dates], [infer_datetime_format], [dayfirst])`
- `names`: List of column names to use.
- `parse_dates`: (List of columns names)
- `infer_datetime_format` (bool) If `True` and `parse_dates` is enabled, pandas will attempt to infer the format of the datetime strings in the columns, and if it can be inferred, switch to a faster method of parsing them.
## `pd.read_excel()`
## `pd.read_table([usecols], [names])`
- `usecols`
- `names`: List of column names to use.
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
- `join`: (`"inner"`, `"outer"`, default `"outer"`)
- `ignore_index`: If `True`, do not use the index values along the concatenation axis. The resulting axis will be labeled `0`, …, `n - 1`.
## `pd.pivot_table(data, values, index, columns, [aggfunc], [fill_value], [drop_na], [margins], [margins_name], [sort=True])`
- Reference: https://pandas.pydata.org/docs/reference/api/pandas.pivot_table.html
- `aggfunc`: (`"mean"`, `"sum"`)
- `margins=True`: Add all row / columns (e.g. for subtotal / grand totals).
```python
pivot = pd.pivot_table(data, index=["dept", "name"], columns=["lvl1", "lvl2"], values="Number of Records", aggfunc="sum", fill_value=0, margins=True)
pivot = pivot.sort_values(["All"], ascending=False)
pivot = pivot.sort_values([("All", "")], axis=1, ascending=False)
```
## `pd.melt()`
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
## `sort_values(by, [axis=0], [ascending=True])`
## `DataFrame.nlargest()`, `DataFrame.nsmallest()`
```python
df.nlargest(3, ["population", "GDP"], keep="all")
```
- `keep`: (`"first"`, `"last"`, `"all"`)
## Manipulating Index
```python
data.index.name
# data.index.names
```
### `sort_index()`
### `set_index([drop])`
### `reset_index([drop], [level])`
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
## `duplicated([keep])`
## `drop_duplicates(subset, [keep])`
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
## `DataFrame.mul()`
```python
df1.mul(df2)
```
## `dot()`
### `DataFrame.dot(other)`
- other: DataFrame
	- Computes the matrix multiplication between the DataFrame and other DataFrame.
- other: Series or Array
	- Computes the inner product, instead of the matrix product here.
### `DataFrame.dot(Series)`

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
## `sample([replace], [weights], [frac], [random_state])`
- `replace`: (bool) Allow or disallow sampling of the same row more than once.
- `weights`
	- *Default `None` results in equal probability weighting.*
	- If passed a Series, will align with target object on index. Index values in weights not found in sampled object will be ignored and index values in sampled object not in weights will be assigned weights of zero.
	- If called on a DataFrame, will accept the name of a column when axis = 0. Unless weights are a Series, weights must be same length as axis being sampled.
	- *If weights do not sum to 1, they will be normalized to sum to 1. Missing values in the weights column will be treated as zero. Infinite values not allowed.*
## `iterrows()`
```python
for name, row in Data.iterrows():
	...
```
- Iterate over DataFrame rows as (index, Series) pairs.
## `iteritems()`
```python
{name:ser for name, ser in x_train.iteritems()}
```
- ***Iterate over the DataFrame columns, returning a tuple with the column name and the content as a Series.***
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
## `np.set_printoptions([edgeitems], [infstr], [linewidth], [nanstr], [precision], [suppress], [threshold], [formatter])`
## `Array.size`
## `Array.astype()`
```python
x_train = x_train.astype("float32")
```
## `np.inf`
## `np.load("file_name.npy")`
## `np.logical_and()`, `np.logical_or()`
## `np.array_equal()`
## `np.linspace()`
## `np.meshgrid()`
## `np.isin()`
## `np.digitize()`
```python		
bins=range(0, 55000, 5000)
data["price_range"]=np.digitize(data["money"], bins)
```
## `np.reshape(newshape)`, `Array.reshape(newshape)`
## `np.unique()`
```python
items, counts = np.unique(intersected_movie_ids, return_counts=True)
```
## `np.fromfile()`
- `count`: Number of items to read. `-1` means all items (i.e., the complete file).

# Array 생성 함수
## `np.full(shape, fill_value)`
## `np.eye(a)`
- `a`: (Array-like)
## `np.ones_like()`, `np.zeros_like()`
## `np.zeros(shape)`, `np.ones(shape)`
## `np.arange([start], stop, [step])`
- `start`: (default 0) Start of interval. The interval includes this value.
- `stop`: End of interval. The interval does not include this value, except in some cases where step is not an integer and floating point round-off affects the length of out.
- `step`: (default 1)
## `np.split(ary, indices_or_sections, [axis=0])`
- `indices_or_sections`
	- (int) `ary` will be divided into `indices_or_sections` equal arrays along `axis`. If such a split is not possible, an error is raised.
## `np.sqrt()`
## `np.power()`
## `np.exp()`
## `np.isnan()`
## `np.nanmean()`
## `np.sort()`
## `np.any()`
## `np.all()`
## `np.where()`
```python
np.min(np.where(cumsum >= np.cumsum(cnts)[-1]*ratio))
```
## `np.tanh()`
## `np.shape()`
## `np.empty()`

# Functions for Manipulating Matrices
## `np.add.outer()`, `np.multiply.outer()`
```python
euc_sim_item = 1 / (1 + np.sqrt(np.add.outer(square, square) - 2*dot))
```
## Diagonal
### `np.fill_diagonal()`
### `np.diag(v, k=0)`
- If `v` is a 2-D array, returns a copy of its `k`-th diagonal.
- If `v` is a 1-D array, returns a 2-D array with `v` on the `k`-th diagonal.

## Linear Algebra
### `np.linalg.norm()`
```python
np.linalg.norm(x, axis=1, ord=2)
```
- `ord=1`: L1 normalization.
- `ord=2`: L2 normalization.

## 모양 변화
### `np.expand_dims()`
```python
np.expand_dims(mh_df.values, axis=1)
```
## `np.einsum()`
## `np.concatenate()`
## `np.stack()`
## `np.delete()`
## `np.argmax()`
## `np.swapaxes()`
## `np.max()`, `np.min()`
## `np.maximum()`, `np.minimum()`
- Element-wise minimum(maximum) of Array elements.
## `np.cumsum()`
- `axis`
## `np.prod()`
- Return the product of Array elements over a given axis.
- `axis`
## `np.quantile()`
```python
lens = sorted([len(doc) for doc in train_X])
ratio = 0.99
max_len = int(np.quantile(lens, ratio))
print(f"가장 긴 문장의 길이는 {np.max(lens)}입니다.")
print(f"길이가 {max_len} 이하인 문장이 전체의 {ratio:.0%}를 차지합니다.")
```
## `Array.ravel()`
```python
arr.ravel(order="F")	
```
- `order="C"` : row 기준
- `order="F"` : column 기준
## `Array.flatten()`
- 복사본 반환
## `Array.transpose()`
```python
conv_weights = np.fromfile(f, dtype=np.float32, count=np.prod(conv_shape)).reshape(conv_shape).transpose((2, 3, 1, 0))
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

# `random`, `np.random`, `tf.random`
## Seed
### `random.seed()`
### `np.random.seed()`
### `tf.random.set_seed()`
### `tf.random.normal()`
## Sample
### `random.random()`
- Returns a random number in [0, 1).
### `np.random.random()`
```python
np.random.random((2, 3, 4))
```
- Create an Array of the given shape and populate it with random samples from a uniform distribution over [0, 1).
### `np.random.rand()`
### `random.sample(sequence, k)`
### `random.choices(sequence, k, [weights])`
### `np.random.choice(size, [replace], [p])`
- Generates a random sample from a given 1-D array.
- `replace`: (bool)
- `p`: The probabilities associated with each entry in `a`. If not given, the sample assumes a uniform distribution over all entries in `a`
### `random.randint(a, b)`
- Return a random integer N such that `a` <= N <= `b`.
- Alias for `randrange(a, b+1)`.
### `np.random.randint(low, [high], size)`
- Return random integers from `low` (inclusive) to `high` (exclusive).
- Return random integers from the "discrete uniform" distribution of the specified dtype in the "half-open" interval [`low`, `high`). If `high` is None (the default), then results are from [`0`, `low`).
## Shuffle
### `random.shuffle()`
- In-place function

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

# `zipfile`
```python
import zipfile
```
## `zipfile.ZipFile(file, mode).extractall(path)`

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
## `pk.load()`

# `json`
- Reference: https://docs.python.org/3/library/json.html
```python
import json
```
## `json.dump(obj, fp, [ensure_ascii], [indent])`
- `ensure_ascii`: If `True` (the default), the output is guaranteed to have all incoming non-ASCII characters escaped. If `False`, these characters will be output as-is.
- `indent`: If a string (such as `"\t"`), that string is used to indent each level.
## `json.load(f)`

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
- For Jupyter Notebook
	```python
	from tqdm.notebook import tqdm
	```
- For Google Colab
	```python
	from tqdm.auto import tqdm
	```
## `tqdm.pandas()`
- `DataFrame.progress_apply()`를 사용하기 위해 필요합니다.

# `warnings`
```python
import warnings

# `category`: (`DeprecatingWarning`)
warnings.filterwarnings("ignore", category=DeprecationWarning)
```

# Download Files
- Using `urllib.request.urlretrieve()`
	```python
	urllib.request.urlretrieve(url=url, filename=filename)
	```
- Using `google_drive_downloader.GoogleDriveDownloader.download_file_from_google_drive()`
	```python
	# Install: !pip install googledrivedownloader
	
	from google_drive_downloader import GoogleDriveDownloader as gdd

	gdd.download_file_from_google_drive(file_id, dest_path)
	```
- Using `tensorflow.keras.utils.get_file()`
	```python
	# `fname`: Name of the file. If an absolute path is specified the file will be saved at that location. If `None`, the name of the file at origin will be used. By default the file is downloaded to `~/.keras/datasets`.
	# `origin`: Original URL of the file.
	path_to_downloaded_file = tf.keras.utils.get_file([fname], origin, [untar])
	```
	
	# Setting
## Display Hangul
```python
path = "C:/Windows/Fonts/malgun.ttf"
if platform.system() == "Darwin":
    mpl.rc("font", family="AppleGothic")
elif platform.system() == "Windows":
    font_name = mpl.font_manager.FontProperties(fname=path).get_name()
    mpl.rc('font', family=font_name)
```
- `family`: (`"NanumBarunGothic"`)
## Display Minus Sign
```python
mpl.rc("axes", unicode_minus=False)
```
## Plot Style
```python
plt.style.use("dark_background")
```
- (`"default"`, `"dark_background"`)

# Charts
## Common Arguments
- `lw`
- `label`
- `fontsize`
- `title`
- `legend`: (bool)
- `xlim`, `ylim`
- `figsize`
- `grid`
- `marker`
	- `"o"`: Circle
	- `"*"`: Star
- `linestyle` (= `ls`)
	- `"-"`: Solid line
	- `"--"`: Dashed line
	- `"-."`: Dash-dot line
	- `":"`: Dotted line
- `color` (= `c`)
	- `"b"`: Blue
	- `"g"`: Green
	- `"r"`: Red
	- `"c"`: Cyan
	- `"m"`: Magenta
	- `"y"`: Yellow
	- `"k"` black
	- `"w"`: White
## Line Chart
- Reference: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html
### `plt.plot();`
### `DataFrame.plot.line([style=[...])`, `Series.plot.line()`
- `style`: For example, `["k--", "bo-", "r*"]`
- `figsize`
## Pie Chart
### `DataFrame.plot.pie()`, `Series.plot.pie()`
- `startangle`
### `sb.lineplot(x, y, [linewidth], [color], [label])`
## Scatter Plot
### `plt.scatter([s], [c], [cmap], [alpha], [edgecolors], [linewidth])`
### `DataFrame.plot.scatter(y, x, c, grid, figsize, cmap)`
- `s`
- `cmap`: ("RdYlBu", ...)
- `alpha`
- `edgecolors`
### `sb.scatterplot(ax, data, x, y, hue, hue_norm, palette, s, alpha)`
## Bar Chart
### `plt.bar(x, height)`
### `DataFrame.plot.barh(x, y, [stacked], [color])`, `Series.plot.barh()`
- Reference: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.plot.barh.html
- `x`: If not specified, the index of the DataFrame is used.
- `y`: If not specified, all numerical columns are used.
### `plt.barh(y, width, height, [alpha], [color], [label], [edgecolor], [linewidth])`
### `sb.barplot(ax, x, y, color, edgecolor, orient)`
## Histogram
### `ax.hist()`, `DataFrame.plot.hist()`, `Series.plot.hist()`
```python
ax.hist(cnt_genre["genre"], bins=30)
```
```python
raw_data.hist(bins=20, grid=True, figsize=(16,12))
```
## Box Plot
### `ax.boxplot()`, `DataFrame.boxplot()`, `Series.boxplot()`
```python
raw_data.boxplot(column="count", by="season", grid=False, figsize=(12,5))
```
## Horizontal Line & Vertical Line
### `axhline(y, [color], [ls], [lw], [label])`, `axvline()`
### `hlines()`, `vlines()`
## `ax.text()`
```python
for _, row in ml_gby_ax.iterrows():
    ax.text(y=row["le"]-0.18, x=row["abs_error"], s=round(row["abs_error"], 1), va="center", ha="left", fontsize=10)
```
- `va`
- `ha`
- `fontsize`
- `s`
## `ax.fill_between()`
```python
ax.fill_between(x, y1, y2, ...)
```
## Heat Map
- Using `seaborn.heatmap()`
	- Reference: http://seaborn.pydata.org/generated/seaborn.heatmap.html
	```python
	# `center`: The value at which to center the colormap when plotting divergant data. Using this parameter will change the default `cmap` if none is specified.
	# `mask`: If passed, data will not be shown in cells where `mask` is True. Cells with missing values are automatically masked.
	sb.heatmap([ax], data, [annot=True], [annot_kws={"size"}], [fmt=".2f"], [linewidths], [linecolor], [center], [cmap], [cbar=True], [mask]);
	```
- Using `matplotlib.pyplot.pcolormesh()`
	```python
	plt.pcolormesh([cmap]);
	plt.colorbar();
	```
## `sb.countplot(ax, [data], [x], [y], [hue], [orient])`
- Reference: https://seaborn.pydata.org/generated/seaborn.countplot.html
- Show the counts of observations in each categorical bin using bars.
- [`data`]: (DataFrame)
- [`orient`]: (`"v"`, `"h"`)
## ETC
### `sb.lmplot(data, x, y, fit_reg, line_kws, size, aspect, ci, sharey)`
- `data`: (DataFrame)
- `fit_reg`: (bool) If `True`, estimate and plot a regression model relating the x and y variables.
- `ci`: (int in [0, 100] or None, optional) Size of the confidence interval for the regression estimate. This will be drawn using translucent bands around the regression line. The confidence interval is estimated using a bootstrap; for large datasets, it may be advisable to avoid that computation by setting this parameter to None.
- `aspect`: Aspect ratio of each facet, so that aspect\*height gives the width of each facet in inches.
### `sb.distplot(a, norm_hist)`
- `a`: (Series, 1d-Array, or List)
- `norm_hist`: (bool, optional) If `True`, the histogram height shows a density rather than a count. This is implied if a KDE or fitted density is plotted.
### `sb.replot(x, y, col, hue, style, kind, data)`
### `sb.kedplot(ax, data)`
### `sb.stripplot(x, y, data, jitter, edgecolor, size)`
### `sb.pairtplot()`

# Grid
#### `ax.grid()`
```python
ax.grid(axis="x", color="White", alpha=0.3, linestyle="--", linewidth=2)
```

# Size
## Set figure size
```python
plt.figure(figsize=(w, h))
```
```python
fig.set_size_inches(w, h)
```

# Axis
## Axis off
```python
ax.axis("off")
```
## Set Axis Invisible
```python
ax.xaxis.set_visible(False)
# ax.yaxis.set_visible(False)
```
## `ax.axis()`
```python
ax.axis([2, 3, 4, 10])
```
## Set Axis Range
```python
ax.set_xlim([x1, x2])
# ax.set_ylim([y1, y2])
```
## Set Label Position
```python
ax.xaxis.set_label_position("top")
```
## Invert Axis
```python
plt.gca().invert_xaxis()
# plt.gca().invert_yaxis()
```
```python
ax.invert_xaxis()
# ax.invert_yaxis()`
```

# Tick
## `ax.tick_params(axis, [labelsize], [labelcolor], [labelrotation], [grid_linewidth])`
- `axis`: (`"x"`, `"y"`)
## Set Tick
```python
ax.set_yticks(np.arange(1, 1.31, 0.05))
```
- 화면에 표시할 눈금을 설정합니다.
## Set Tick Position
```python
ax.yaxis.set_ticks_position("right")
```
- (`"top"`, `"bottom"`, `"left"`, `"right"`)
## Set Tick Format
```python
ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter("{x:,.0f}"))
```

# Label
## `ax.set_xlabel()`, `ax.set_ylabel()`
## `ax.set_xticklabels([va])`, `ax.set_yticklabels([ha])`

# Title
## Set title
```python
plt.title()
# ax.set_title()
```
- `size`: (float)

# Legend
## Place a legend
```python
plt.plot(xs, ys, label=label)
plt.legend()
```
- `loc`: (`"best"`, `"center"`, `"upper left"`, `"upper right"`, `"lower left"`, `"lower right"`, `"upper center"`, `"lower center"`, `"center left"`, `"center right"`)
- `bbox_to_anchor`: Box that is used to position the legend in conjunction with `loc`. This argument allows arbitrary placement of the legend.
- `fancybox`: Whether round edges should be enabled.
- `shadow`: Whether to draw a shadow behind the legend.
- `ncol`: The number of columns that the legend has.
- `fontsize`

# Color Bar
```python
cbar = fig.colorbar(ax=ax, mappable=scatter)
```
### Set Color Bar Label
```python
cbar.set_label(label="전용면적(m²)", size=15)
```

# Save Figure
```python
plt.savefig()
# fig.savefig()
```
- `bbox_inches="tight"`

# Subplots
## `plt.subplot()`
```python
for i in range(9):
	ax = plt.subplot(3, 3, i + 1)
```
## `plt.subplots()`
```python
fig, axes = plt.subplots(nrows, ncols, ...)
```
- `figsize`
- `sharex`, `sharey`: (bool) Controls sharing of properties among x (`sharex`) or y (`sharey`) axes.

# Options
## `plt.setp()`
## `plt.show()`
## `fig.tight_layout()`
## `plt.imshow([cmap])`
## `plt.set()`
```python
ax.set(title="Example", xlabel="xAxis", ylabel="yAxis", xlim=[0, 1], ylim=[-0.5, 2.5], xticks=data.index, yticks=[1, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3])
```
- `title`
- `xlabel`, `ylabel`
- `xlim`, `ylim`
- `xticks`, `yticks`
- Reference: https://matplotlib.org/stable/api/_as_gen/matplotlib.lines.Line2D.html#matplotlib.lines.Line2D
## `sb.set()`
- `palette`: (`"muted"`)
- `color_codes`: If `True` and `palette` is a seaborn palette, remap the shorthand color codes (e.g. `"b"`, `"g"`, `"r"`, etc.) to the colors from this palette.
- `font_scale`: (float)