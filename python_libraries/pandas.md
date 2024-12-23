```python
import pandas as pd
```
# Options
```python
pd.options.display.max_rows = None
pd.options.display.max_columns = None
pd.options.display.width
pd.options.display.float_format
pd.options.display.max_colwidth
# Ignore `SettingWithCopyWarning`
pd.options.mode.chained_assignment = None
```
```python
pd.set_option("display.float_format", "{:.3f}".format)
```

```python
<df>.style.set_precision()
# (`"background_color"`, `"font_size"`, `"color"`, `"border_color"`, ...)
<df>.style.set_properties()
# (`color`)
<df>.style.bar()
<df>.style.background_gradient()

# Examples
data.corr().style.background_gradient(cmap="Blues").set_precision(1).set_properties(**{"font-size":"9pt"})
```

# Read Data
```python
# Reference: https://pandas.pydata.org/docs/reference/api/pandas.read_excel.html
# `sheet_name`
# `usecols`: Column names to be used.
# `dtype`: e.g., `dtype={"원문": str, "원문_converted": str, "원문_joined": str}`
# `header`: Row (0-indexed) to use for the column labels of the parsed DataFrame. If a list of integers is passed those row positions will be combined into a `MultiIndex`. Use `None` if there is no header.
# Example
pd.read_excel("...", dtype={"Index no.": str})

# `usecols`: List of columns to read.
# `parse_dates`: (List of column names)
# `infer_datetime_format`: (bool) If `True` and `parse_dates` is enabled, pandas will attempt to infer the format of the datetime strings in the columns, and if it can be inferred, switch to a faster method of parsing them.
pd.read_csv([thousands=","], [float_precision], [skiprows], [error_bad_lines], [index_col], [sep], [names], [parse_dates], [infer_datetime_format], [dayfirst])

# `usecols`
# `names`: List of column names to use.
# `on_bad_lines="skip"`
pd.read_table([usecols], [names])

pd.read_pickle()

pd.read_sql(query, conn)
```

# Save DataFrame
```python
# Save DataFrame to csv File
# `index`: (Bool)
<df>.to_csv()

# Save DataFrame to xlsx File
# `sheet_name`: (String, default `"Sheet1"`)
# `na_rep`: (String, default `""`) Missing data reprsentation.
# `float_format`
# `header`: If a list of string is given it is assumed to be aliases for the column names.
# merge_cells
<df>.to_excel(sheet_name, na_rep, float_format, header, merge_cells)

# Save DataFrame to xlsx File with Multiple Sheets
writer = pd.ExcelWriter("....xlsx", engine="xlsxwriter")
df1.to_excel(writer, sheet_name="sheet1")
df2.to_excel(writer, sheet_name="sheet2")
...
writer.save()

# Save DataFrame to pkl File
<df>.to_pickle(path)
```

# Copy DataFrame
```python
# Deep copy
<df>.copy()
# Shallow copy
<df>.copy(deep=False)
```

# Check for Data Type
```python
pd.api.types.is_string_dtype()
pd.api.types.is_numeric_dtype()
```

# Create DataFrame
```python
# `data`: (Array, List, List of Tuples, Dictionary of List, List of Dictionary)
pd.DataFrame(data, index, columns)
```
## From Dictionary
```python
# Example
pd.DataFrame.from_dict(cnts_img, orient="index", columns=["cnt_total"])
```

# Crosstab
```python
pd.crosstab(index, columns, margins)
# Example
pd.crosstab(index=data["count"], columns=data["weather"], margins=True)
```

# Concat DataFrames
```python
# `axis`: (default `0`)
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
# Dummy Variable
```python
# `prefix`: String to append DataFrame column names. Pass a list with length equal to the number of columns when calling get_dummies on a DataFrame.
# `dop_first`: Whether to get k-1 dummies out of k categorical levels by removing the first level.
# `dummy_na`: Add a column to indicate NaNs, if False NaNs are ignored.
	# 결측값이 있는 경우 `drop_first`, `dummy_na` 중 하나만 `True`로 설정해야 함
# Example
data = pd.get_dummies(
	data,
	columns=["heating", "company1", "company2", "elementary", "built_month", "trade_month"],
	drop_first=False,
	dummy_na=True
)
```
# Merge DataFrames
```python
# `how`: (`"left"`, `"right"`, `"outer"`, `"inner"`, `"cross"`)
pd.merge(on, how, [left_on], [right_on], [left_index], [right_index], [how="inner"])
# Eamples
data = pd.merge(data, start_const, on=["지역구분", "입찰년월"], how="left")
data = pd.merge(df1, df2, left_on="id", right_on="movie_id")
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

# DataFrame to Dictionary
```python
# `orient`: ("dict", "list", "series", "split", "records", "index")
df.to_dict(orient="records")
```

# Describe DataFrame
```python
# Show number of samples, mean, standard deviation, minimum, Q1(lower quartile), Q2(median), Q3(upper quantile), maximum of each independent variable.
# Example
raw_data.describe(include="all").T
```
# `<df>.corr()`

# `<df>.shape`

# `<df>.ndim`

# Quantile
```python
<df>.quantile()

# Examples
top90per = plays_df[plays_df["plays"]>plays_df["plays"].quantile(0.1)]
```

# Group DataFrame
```python
# `dropna`: (Bool, default True)
gby = <df>.groupby([as_index])

for name, df_grouped in gby:
	...
gby.groups
gby.get_group()
gby.mean() # DataFrame
gby.count() # DataFrame
gby.size() # Series
gby.apply()

# Examples
df.groupby(["Pclass", "Sex"], as_index=False)
over4.groupby("user_id")["movie_id"].apply(set)
```

# `<df>.pivot()`
```python
df_pivoted = <df>.pivot("col1", "col2", "col3")
```

# `<df>.stack()`
- 열 인덱스 -> 행 인덱스로 변환

# `unstack([level=-1], [fill_value=None])`
- Reference: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.unstack.html
- Pivot a level of the (necessarily hierarchical) index labels. Returns a DataFrame having a new level of column labels whose inner-most level consists of the pivoted index labels. If the index is not a MultiIndex, the output will be a Series (the analogue of stack when the columns are not a MultiIndex).
- `level`: Level(s) of index to unstack, can pass level name.
- `fill_value`: Replace NaN with this value if the unstack produces missing values.

# `<df>.append()`
```python
df = df.append(
	<Dictionary>,
	ignore_index=True
)
```
# `<df>.apply()`
```python
# Examples
hr["코스트센터 분류"] = hr.apply(
	lambda x: "지사" if ("사업소" in x["조직명"]) or ("베트남지사" in x["조직명"]) else ("본사" if re.search("^\d", x["코스트센터"]) else "현장"), axis=1
)
```
```python
# Example
copied[["r", "g", "b"]] = copied.apply(lambda x: _to_tuple(x["text_color"]), axis=1, result_type="expand")
```

# Progress Bar
```python
from tqdm.auto import tqdm

tqdm.pandas()
...

<DatFrame or Series>.progress_apply(...)
<Series>.progress_map(...)
```
# Rename Column
```python
# Example
data = data.rename({"단지명.1":"name", "세대수":"houses_buildings", "저/최고층":"floors"}, axis=1)
```
# Sort DataFrame
```python
# `by`: List로 여러 개의 컬럼을 받을 경우 List의 우에서 좌로 정렬을 적용합니다.
sort_values(by, [axis=0], [ascending=True], [inplace=False])
# Example
df_sim_sents.sort_values(by=["similarity", "id1", "id2"], ascending=[False, True, True], inplace=True)
```
## Custom Order
```python
# Example
df["languages"] = pd.Categorical(
	df["languages"], [
		"ENKO", "KOEN", "KOZH-CN", "ENZH-CN", "ZH-CNKO", "ZH-CNEN", "JAZH-CN", "KOJA", "ENJA", "JAKO", "JAEN", "ZH-CNJA"
	]
)
df["client"] = pd.Categorical(
	df["client"], ["Apple", "Google", "Papago"]
)
df.sort_values(["languages", "client"], inplace=True)
```
## Sort DataFrame by Column Frequency
```python
# Example
word2freq = df.groupby(["어휘"]).size()
df["freq"] = df["어휘"].map(word2freq)
df.sort_values(["freq"], inplace=True)
```

# Index
```python
data.index.name
# data.index.names
```
## Sort by Index
```python
sort_index()
```
## Set Index
```python
set_index([drop])
```
## Reset Index
```python
# `drop`: (bool) Reset the index to the default integer index.
# `level`: Only remove the given levels from the index. Removes all levels by default.
reset_index([drop=False], [level], [inplace=False])
```
## Index of the Maximum Value
```python
# Example
data["genre"] = data.loc[:, "unknown":"Western"].idxmax(axis=1)
```

# `<df>.loc()`
```python
# Example
data.loc[data["buildings"]==5, ["age", "ratio2"]]
data.loc[[7200, "대림가경"], ["houses", "lowest"]]
```
# `<df>.isin()`
```python
# `values`: (List, Dictionary)
train_val = data[~data["name"].isin(names_test)]
```

# Columns
## Drop Column.
```python
# Example
data.drop(data.loc[:, "unknown":"Western"].columns, axis=1, inplace=True)
```
## `<df>.columns.droplevel`
```python
# Example
df.columns = df.columns.droplevel([0, 1])
```
## Insert Column
```python
# Example
df_sim_sents.insert(2, "text1", df_sim_sents["id1"].map(id2text))
```

# Check If DataFrame Is Empty
```python
<df>.empty
```

# Duplicates
## Check If Duplicated
```python
# `keep`: (`"first"`, `"last"`, False)
	# `keep=False`: Mark all duplicates as `True`.
duplicated([keep="first"])
```
## Drop Duplicates
```python
# `keep`: (`"first"`, `"last"`, False)
drop_duplcates(subset, [keep="first"])
```

# Math Operations
## Multiply
```python
<df>.mul()
```
```python
# other: DataFrame
    # Computes the matrix multiplication between the DataFrame and other DataFrame.
# other: Series or Array
	# Computes the inner product, instead of the matrix product here.
<df>.dot(other)
```
```python
<df>.add()
<df>.sub()
<df>.pow()

# Examples
adj_ui = ui.sub(user_bias, axis=0).sub(item_bias, axis=1)
```

# Treat Missing Values
## Check If Missing Value
```python
<ser>.isna()
<ser>.notna()
```
## Drop Missing Values
```python
<df>.dropna([subset], [inplace])
```
## Fill Missing Values
```python
# `method="ffill"`: Propagate last valid observation forward to next valid backfill.
<df>.fillna(method)
# Example
data = data.fillna(method="ffill")
```

# Treat String
## `<ser>.str.len()`
## `<ser>.str.replace()`
```python
data["parking"] = data["parking"].str.replace("대", "", regex=False)
```
## `<ser>.str.split()`
```python
data["buildings"] = data.apply(lambda x : str(x["houses_buildings"]).split("총")[1][:-3], axis=1)
```
- `pat`: Same as `" "` When omitted.
## `<ser>.str.strip()`
```python
data["fuel"] = data["heating"].apply(lambda x:x.split(",")[1]).str.strip()
```
## `<ser>.str.contains()`
```python
train[train["token"].str.contains(".", regex=False)]
```
## `<ser>.str.encode()`, `<ser>.str.decode()`
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
<ser>.cat.categories
# `ordered`: 순서 부여
<ser>.cat.set_categories()

# Examples
ser.cat.set_categories([2, 3, 1], ordered=True)
```
## `<ser>.cat.codes`
```python
# Perform label encoding.
for cat in cats:
    data[cat] = data[cat].cat.codes
```

# Iteration
## Iterate over Rows
```python
# 일반적으로 `iterrows()` -> `itertuples()` -> `values` 순으로 속도가 빨라집니다.
for name, row in <df>.iterrows():
	# type: DataFrame row
	...
for row in <df>.itertuples():
	...
for value in <df>.values:
	# type: Array
	...
# `items()`와 `iteritems()`는 서로 동일합니다.
for idx, value in raw_data["quarter2"].items():
    ...
```
## Iterate over Columns
```python
for name, col in <df>.iteritems():
	...
```

# Insert Column
```python
# Example
asso_rules.insert(1, "antecedents_title", asso_rules["antecedents"].apply(lambda x : id2title[list(x)[0]]))
```
# `<df>.drop()`
# `<ser>.rename()`
```python
plays_df.groupby(["user_id"]).size().rename("n_arts")
```
# `<ser>.map()`
```python
target_ratings["title"] = target_ratings["movie_id"].map(target)
```
```python
all_seen = ratings_df_target_set[ratings_df_target_set.map(lambda x : len(x)==5)].index
```
# `<ser>.astype()`
```python
data.loc[:, cats] = data.loc[:, cats].astype("category")
```
- `dtype`: (`"int32"`, `"int63"`, `"float64"`, `"object"`, `"category"`, `"string"`)
- `errors`: (`"raise"`, `"ignore"`, default `"raise"`)

# Logical Operations
## Count of Unique Values
```python
# `sort`: (bool)
<ser>.value_counts([sort], [ascending])
```
## Number of Unique Values
```python
n_item = ratings_df["movie_id"].nunique()
```
## Cumulative Summation
```python
# Example
cumsum = n_rating_item.cumsum() / len(ratings_df)
```
## Top n Largest Values
```python
# `keep`: (`"first"`, `"last"`, `"all"`)
# Example
df.nlargest(3, ["population", "GDP"], keep="all")
```
## Replace Values Where the Condition Is False
```python
<df>.where(cond, other, [inplace=False])
# Example
df_db.where(df_db < -60, -60)
```
## Logical Operators
```python
<df>.eq
<df>.ne
<df>.le
<df>.lt
<df>.ge
<df>.gt
```