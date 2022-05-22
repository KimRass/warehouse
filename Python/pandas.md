```python
import pandas as pd
```
# Options
```python
pd.options.display.max_rows = None
pd.options.display.max_columns = None
pd.options.display.width
pd.options.display.float_format
# Ignore `SettingWithCopyWarning`
pd.options.mode.chained_assignment = None
```
```python
pd.set_option("display.float_format", "{:.3f}".format)
```

```python
DataFrame.style.set_precision()
# (`"background_color"`, `"font_size"`, `"color"`, `"border_color"`, ...)
DataFrame.style.set_properties()
# (`color`)
DataFrame.style.bar()
DataFrame.style.background_gradient()

# Examples
data.corr().style.background_gradient(cmap="Blues").set_precision(1).set_properties(**{"font-size":"9pt"})
```

# Read Data
```python
# `names`: List of column names to use.
# `parse_dates`: (List of column names)
# `infer_datetime_format`: (bool) If `True` and `parse_dates` is enabled, pandas will attempt to infer the format of the datetime strings in the columns, and if it can be inferred, switch to a faster method of parsing them.
pd.read_csv([thousands=","], [float_precision], [skiprows], [error_bad_lines], [index_col], [sep], [names], [parse_dates], [infer_datetime_format], [dayfirst])
```
```python
# `usecols`
# `names`: List of column names to use.
pd.read_table([usecols], [names])
```
```python
pd.read_sql(query, conn)
```

# Save DataFrame
```python
# `index`: (Bool)
DataFrame.to_csv()
```
```python
# `sheet_name`: (String, default `"Sheet1"`)
# `na_rep`: (String, default `""`) Missing data reprsentation.
# `float_format`
# `header`: If a list of string is given it is assumed to be aliases for the column names.
# merge_cells
DataFrame.to_excel(sheet_name, na_rep, float_format, header, merge_cells)`
```

```python
pd.api.types.is_string_dtype()
pd.api.types.is_numeric_dtype()
```
```python
# `data`: (Array, List, List of Tuples, Dictionary of List)
pd.DataFrame(data, index, columns)`
```
```python
pd.crosstab(index, columns, margins)

pd.crosstab(index=data["count"], columns=data["weather"], margins=True)
```
# Concat DataFrames
```python
# `join`: (`"inner"`, `"outer"`, default `"outer"`)
# `ignore_index`: If `True`, do not use the index values along the concatenation axis. The resulting axis will be labeled `0`, …, `n - 1`.
pd.concat(objs, [join], [ignore_index])
```
# Pivot Table
```python
# Reference: https://pandas.pydata.org/docs/reference/api/pandas.pivot_table.html
# `aggfunc`: (`"mean"`, `"sum"`)
# `margins=True`: Add all row / columns (e.g. for subtotal / grand totals).
pd.pivot_table(data, values, index, columns, [aggfunc], [fill_value], [drop_na], [margins], [margins_name], [sort=True])

# Examples
pivot = pd.pivot_table(data, index=["dept", "name"], columns=["lvl1", "lvl2"], values="Number of Records", aggfunc="sum", fill_value=0, margins=True)
```
```python
pd.melt()
```
```python
pd.cut()
```
```python
pd.Categorical([ordered])

# Reference: https://kanoki.org/2020/01/28/sort-pandas-dataframe-and-series/
# dtype을 `category`로 변환.
# `ordered`: (Bool, deafult `False`): category들 간에 대소 부여.
# Examples
results["lat"] = pd.Categorical(results["lat"], categories=order)
```
# `pd.get_dummies()`
```python
data = pd.get_dummies(data, columns=["heating", "company1", "company2", "elementary", "built_month", "trade_month"], drop_first=False, dummy_na=True)
```
- `prefix`: String to append DataFrame column names. Pass a list with length equal to the number of columns when calling get_dummies on a DataFrame.
- `dop_first`: Whether to get k-1 dummies out of k categorical levels by removing the first level.
- `dummy_na`: Add a column to indicate NaNs, if False NaNs are ignored.
- 결측값이 있는 경우 `drop_first`, `dummy_na` 중 하나만 `True`로 설정해야 함
```python
# `how`: (`"left"`, `"right"`, `"outer"`, `"inner"`, `"cross"`)
pd.merge(on, how, [left_on], [right_on], [left_index], [right_index], [how="inner"])

# Eamples
data = pd.merge(data, start_const, on=["지역구분", "입찰년월"], how="left")

pd.merge(df1, df2, left_on="id", right_on="movie_id")

floor_data = pd.merge(floor_data, df_conv, left_index=True, right_index=True, how="left")
```
## `pd.MultiIndex.from_tuples()`
```python
order = pd.MultiIndex.from_tuples((hq, dep) for dep, hq in dep2hq.items())
```
## `pd.plotting.scatter_matrix()`
```python
pd.plotting.scatter_matrix(data, figsize=(18, 18), diagonal="kde")
```
## `pd.plotting.lag_plot()`
```python
fig = pd.plotting.lag_plot(ax=axes[0], series=resid_tr["resid"].iloc[1:], lag=1)
```

# `DataFrame.describe()`
```python
raw_data.describe(include="all").T
```
- Show number of samples, mean, standard deviation, minimum, Q1(lower quartile), Q2(median), Q3(upper quantile), maximum of each independent variable.
# `DataFrame.corr()`
# `DataFrame.shape`
# `DataFrame.ndim`
```python
DataFrame.quantile()

# Examples
top90per = plays_df[plays_df["plays"]>plays_df["plays"].quantile(0.1)]
```
```python
gby = DataFrame.groupby([as_index])

gby.groups
gby.mean() # DataFrame
gby.count() # DataFrame
gby.size() # Series
gby.apply()

# Examples
df.groupby(["Pclass", "Sex"], as_index=False)
over4.groupby("user_id")["movie_id"].apply(set)
```
# `DataFrame.pivot()`
```python
df_pivoted = df.pivot("col1", "col2", "col3")
```
# `DataFrame.stack()`
- 열 인덱스 -> 행 인덱스로 변환
# `unstack([level=-1], [fill_value=None])`
- Reference: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.unstack.html
- Pivot a level of the (necessarily hierarchical) index labels. Returns a DataFrame having a new level of column labels whose inner-most level consists of the pivoted index labels. If the index is not a MultiIndex, the output will be a Series (the analogue of stack when the columns are not a MultiIndex).
- `level`: Level(s) of index to unstack, can pass level name.
- `fill_value`: Replace NaN with this value if the unstack produces missing values.
# `DataFrame.append()`
```python
df = df.append({"addr1":addr1, "addr2":addr2, "dist":dist}, ignore_index=True)
```
# `DataFrame.apply()`
```python
hr["코스트센터 분류"] = hr.apply(lambda x:"지사" if ("사업소" in x["조직명"]) or ("베트남지사" in x["조직명"]) else ("본사" if re.search("^\d", x["코스트센터"]) else "현장"), axis=1)
```
```python
hr["제외여부"] = hr.apply(lambda x:"제외" if ("외주" in x["하위그룹"]) | ("촉탁" in x["하위그룹"]) | ("파견" in x["하위그룹"]) | (x["재직여부"]=="퇴직") else ("본부인원에서만 제외" if ("PM" in x["조직명"]) | ("신규준비" in x["직무"]) | (x["직무"]=="휴직") | (x["직무"]=="비상계획") | (x["직무"]=="축구협") | (x["직무"]=="비서") | ("조직명" in x["조직명"]) | (x["직무"]=="미화") else "포함"), axis=1)
```
# `progress_apply()`, `progress_map()`
- `tqdm.pandas()` should be followed.
# `DataFrame.rename()`
```python
data = data.rename({"단지명.1":"name", "세대수":"houses_buildings", "저/최고층":"floors"}, axis=1)
```
# `DataFrame.reindex()`
```python
pivot = pivot.reindex(dep_order)
```
# `DataFrame.insert()`
```python
data.insert(3, "age2", data["age"]*2)
```
# `sort_values(by, [axis=0], [ascending=True])`
# `DataFrame.nlargest()`, `DataFrame.nsmallest()`
```python
df.nlargest(3, ["population", "GDP"], keep="all")
```
- `keep`: (`"first"`, `"last"`, `"all"`)
# Manipulating Index
```python
data.index.name
# data.index.names
```
## `sort_index()`
## `set_index([drop])`
## `reset_index([drop=False], [level], [inplace=False])`
- `drop`: (bool) Reset the index to the default integer index.
- `level`: Only remove the given levels from the index. Removes all levels by default.
# `DataFrame.loc()`
```python
data.loc[data["buildings"]==5, ["age", "ratio2"]]
data.loc[[7200, "대림가경"], ["houses", "lowest"]]
```
# `DataFrame.isin()`
```python
train_val = data[~data["name"].isin(names_test)]
```
- `values`: (List, Dictionary)
# `DataFrame.query()`
```python
data.query("houses in @list")
```
- 외부 변수 또는 함수 사용 시 앞에 @을 붙임.
# `DataFrame.idxmax()`
```python
data["genre"] = data.loc[:, "unknown":"Western"].idxmax(axis=1)
```
# `DataFrame.drop()`
```python
data = data.drop(["Unnamed: 0", "address1", "address2"], axis=1)
```
```python
data = data.drop(data.loc[:, "unknown":"Western"].columns, axis=1)
```
## `DataFrame.columns.drop()`
```python
uses_df.columns.drop("cnt")
```
- Make new Index with passed list of labels deleted.
## `DataFrame.columns.droplevel`
```python
df.columns = df.columns.droplevel([0, 1])
```
## Check If DataFrame Is Empty
```python
df.empty
```

# Duplicates
## `duplicated([keep])`
## `drop_duplcates(subset, [keep])`

# Math Operation
## Multiply
```python
DataFrame.mul()
```
```python
# other: DataFrame
    # Computes the matrix multiplication between the DataFrame and other DataFrame.
# other: Series or Array
	# Computes the inner product, instead of the matrix product here.
DataFrame.dot(other)
```
```python
DataFrame.add()
DataFrame.sub()
DataFrame.pow()

# Examples
adj_ui = ui.sub(user_bias, axis=0).sub(item_bias, axis=1)
```

# Treat Missing Values
```python
DataFrame.isna()
DataFrame.notna()
```
```python
DataFrame.dropna(subset)

# Examples
data = data.dropna(subset=["id"])
```
```python
# `method="ffill"`: Propagate last valid observation forward to next valid backfill.
DataFrame.fillna(method)

# Examples
data = data.fillna(method="ffill")
```

# Treat String
## `Series.str.replace()`
```python
data["parking"] = data["parking"].str.replace("대", "", regex=False)
```
## `Series.str.split()`
```python
data["buildings"] = data.apply(lambda x : str(x["houses_buildings"]).split("총")[1][:-3], axis=1)
```
- `pat`: Same as `" "` When omitted.
## `Series.str.strip()`
```python
data["fuel"] = data["heating"].apply(lambda x:x.split(",")[1]).str.strip()
```
## `Series.str.contains()`
```python
train[train["token"].str.contains(".", regex=False)]
```
## `Series.str.encode()`, `Series.str.decode()`
```python
infos[col] = infos[col].str.encode("latin-1").str.decode("euc-kr")
```
- `error`: (`"strict"`, `"ingore"`, `"replace"`, default `"strict"`)

# `sample([replace], [weights], [frac], [random_state])`
- `replace`: (bool) Allow or disallow sampling of the same row more than once.
- `weights`
	- *Default `None` results in equal probability weighting.*
	- If passed a Series, will align with target object on index. Index values in weights not found in sampled object will be ignored and index values in sampled object not in weights will be assigned weights of zero.
	- If called on a DataFrame, will accept the name of a column when axis = 0. Unless weights are a Series, weights must be same length as axis being sampled.
	- *If weights do not sum to 1, they will be normalized to sum to 1. Missing values in the weights column will be treated as zero. Infinite values not allowed.*

# Treat Category
```python
Series.cat.categories
# `ordered`: 순서 부여
Series.cat.set_categories()

# Examples
ser.cat.set_categories([2, 3, 1], ordered=True)
```
## `Series.cat.codes`
```python
# Perform label encoding.
for cat in cats:
    data[cat] = data[cat].cat.codes
```

# Iteration
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
# Iterate over (index, value) tuples.

# Examples
for i, value in raw_data["quarter2"].items():
    print(i, value)
```
# `Series.items()`
```python
# Examples
for k, v in target.items():
    queries.append(f"{k}-{v}")
```

# `DataFrame.insert()`
```python
asso_rules.insert(1, "antecedents_title", asso_rules["antecedents"].apply(lambda x : id2title[list(x)[0]]))
```
# `DataFrame.drop()`
# `Series.rename()`
```python
plays_df.groupby(["user_id"]).size().rename("n_arts")
```
# `Series.map()`
```python
target_ratings["title"] = target_ratings["movie_id"].map(target)
```
```python
all_seen = ratings_df_target_set[ratings_df_target_set.map(lambda x : len(x)==5)].index
```
# `Series.astype()`
```python
data.loc[:, cats] = data.loc[:, cats].astype("category")
```
- `dtype`: (`"int32"`, `"int63"`, `"float64"`, `"object"`, `"category"`, `"string"`)
- `errors`: (`"raise"`, `"ignore"`, default `"raise"`)

# Aggregate
## Count of Unique Values
```python
# `sort`: (bool)
Series.value_counts([sort], [ascending])
```
## `Series.nunique()`
```python
n_item = ratings_df["movie_id"].nunique()
```
## `Series.cumsum()`
```python
cumsum = n_rating_item.cumsum()/len(ratings_df)
```