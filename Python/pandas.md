```python
import pandas as pd
```
# `pd.api`
## `pd.api.types`
## `pd.api.types.is_string_dtype()`, `pd.api.types.is_numeric_dtype()`
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
# `pd.options`
## `pd.options.display`
## `pd.options.display.max_rows`, `pd.options.display.max_columns`, `pd.options.display.width`, `pd.options.display.float_format`
## `pd.options.mode`
## `pd.options.mode.chained_assignment`
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
# `pd.DataFrame()`
- `data`: (Array, List, List of Tuples, Dictionary of List)
- `index`
- `columns`
# `pd.Series()`
```python
idf_ser = pd.Series(idf, index=vocab)
```
# `pd.read_csv([thousands=","], [float_precision], [skiprows], [error_bad_lines], [index_col], [sep], [names], [parse_dates], [infer_datetime_format], [dayfirst])`
- `names`: List of column names to use.
- `parse_dates`: (List of column names)
- `infer_datetime_format` (bool) If `True` and `parse_dates` is enabled, pandas will attempt to infer the format of the datetime strings in the columns, and if it can be inferred, switch to a faster method of parsing them.
```python
import csv

subws = pd.read_csv("imdb.vocab", sep="\t", header=None, quoting=csv.QUOTE_NONE)
```
# `pd.read_excel()`
# `pd.read_table([usecols], [names])`
- `usecols`
- `names`: List of column names to use.
## `pd.read_sql()`
```python
mall = pd.read_sql(query, conn)
```
# `DataFrame.head()`, `Series.head()`, `DataFrame.tail()`, `Series.tail()`
# `DataFrame.to_csv()`
```python
data.to_csv("D:/☆디지털혁신팀/☆실거래가 분석/☆데이터/실거래가 전처리 완료_200928-3.csv")
```
- `index`: (Bool)
# `DataFrame.to_excel()`
- `sheet_name`: (String, default `"Sheet1"`)
- `na_rep`: (String, default `""`) Missing data reprsentation.
- `float_format`
- `header`: If a list of string is given it is assumed to be aliases for the column names.
- `merge_cells`
# `pd.crosstab()`
```python
pd.crosstab(index=data["count"], columns=data["weather"], margins=True)
```
# `pd.concat()`
- `join`: (`"inner"`, `"outer"`, default `"outer"`)
- `ignore_index`: If `True`, do not use the index values along the concatenation axis. The resulting axis will be labeled `0`, …, `n - 1`.
# `pd.pivot_table(data, values, index, columns, [aggfunc], [fill_value], [drop_na], [margins], [margins_name], [sort=True])`
- Reference: https://pandas.pydata.org/docs/reference/api/pandas.pivot_table.html
- `aggfunc`: (`"mean"`, `"sum"`)
- `margins=True`: Add all row / columns (e.g. for subtotal / grand totals).
```python
pivot = pd.pivot_table(data, index=["dept", "name"], columns=["lvl1", "lvl2"], values="Number of Records", aggfunc="sum", fill_value=0, margins=True)
pivot = pivot.sort_values(["All"], ascending=False)
pivot = pivot.sort_values([("All", "")], axis=1, ascending=False)
```
# `pd.melt()`
# `pd.cut()`
```python
raw_all["temp_group"] = pd.cut(raw_all["temp"], 10)
```
# `pd.Categorical()`
```python
results["lat"] = pd.Categorical(results["lat"], categories=order)
results_ordered = results.sort_values(by="lat")
```
- dtype을 `category`로 변환.
- `ordered`: (Bool, deafult `False`): category들 간에 대소 부여.
- Reference: https://kanoki.org/2020/01/28/sort-pandas-dataframe-and-series/
# `pd.get_dummies()`
```python
data = pd.get_dummies(data, columns=["heating", "company1", "company2", "elementary", "built_month", "trade_month"], drop_first=False, dummy_na=True)
```
- `prefix`: String to append DataFrame column names. Pass a list with length equal to the number of columns when calling get_dummies on a DataFrame.
- `dop_first`: Whether to get k-1 dummies out of k categorical levels by removing the first level.
- `dummy_na`: Add a column to indicate NaNs, if False NaNs are ignored.
- 결측값이 있는 경우 `drop_first`, `dummy_na` 중 하나만 `True`로 설정해야 함
# `pd.merge()`
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
# `pd.MultiIndex`
## `pd.MultiIndex.from_tuples()`
```python
order = pd.MultiIndex.from_tuples((hq, dep) for dep, hq in dep2hq.items())
```
# `pd.plotting`
## `pd.plotting.scatter_matrix()`
```python
pd.plotting.scatter_matrix(data, figsize=(18, 18), diagonal="kde")
```
## `pd.plotting.lag_plot()`
```python
fig = pd.plotting.lag_plot(ax=axes[0], series=resid_tr["resid"].iloc[1:], lag=1)
```
# `DataFrame.style`
## `DataFrame.style.set_precision()`
## `DataFrame.style.set_properties()`
- (`"background_color"`, `"font_size"`, `"color"`, `"border_color"`, ...)
## `DataFrame.style.bar()`
- `color`
## `DataFrame.style.background_gradient()`
```python
data.corr().style.background_gradient(cmap="Blues").set_precision(1).set_properties(**{"font-size":"9pt"})
```
- `cmap`
# `DataFrame.info()`
# `DataFrame.describe()`
```python
raw_data.describe(include="all").T
```
- Show number of samples, mean, standard deviation, minimum, Q1(lower quartile), Q2(median), Q3(upper quantile), maximum of each independent variable.
# `DataFrame.corr()`
# `DataFrame.shape`
# `DataFrame.ndim`
# `DataFrame.quantile()`
```python
top90per = plays_df[plays_df["plays"]>plays_df["plays"].quantile(0.1)]
```
# `DataFrame.groupby()`
```python
df.groupby(["Pclass", "Sex"], as_index=False)
```
- Return an iterable Tuple in the form of (a group, a DataFrame in the group).
## `DataFrame.groupby().groups`
## `DataFrame.groupby().mean()`, `DataFrame.groupby().count()`
- Returns DataFrame.
## `DataFrame.groupby().size()`
- Return Series.
# `DataFrame.groupby().apply()`
```python
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
## `reset_index([drop], [level])`
- `drop`: (bool, default False) Reset the index to the default integer index.
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
# `duplicated([keep])`
# `drop_duplcates(subset, [keep])`
# `DataFrame.columns`
```python
concat.columns = ["n_rating", "cumsum"]
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
# `DataFrame.mul()`
```python
df1.mul(df2)
```
# `dot()`
## `DataFrame.dot(other)`
- other: DataFrame
	- Computes the matrix multiplication between the DataFrame and other DataFrame.
- other: Series or Array
	- Computes the inner product, instead of the matrix product here.
## `DataFrame.dot(Series)`

# `DataFrame.isna()`
# `DataFrame.notna()`
```python
retail[retail["CustomerID"].notna()]
```
# `DataFrame.insert()`
```python
asso_rules.insert(1, "antecedents_title", asso_rules["antecedents"].apply(lambda x : id2title[list(x)[0]]))
```
# `DataFrame.drop()`
# `DataFrame.dropna()`
```python
data = data.dropna(subset=["id"])
```
```python
df.loc[~df.index.isin(df.dropna().index)]
```
# `DataFrame.fillna()`
```python
data = data.fillna(method="ffill")
```
- `method="ffill"`: Propagate last valid observation forward to next valid backfill.
# `sample([replace], [weights], [frac], [random_state])`
- `replace`: (bool) Allow or disallow sampling of the same row more than once.
- `weights`
	- *Default `None` results in equal probability weighting.*
	- If passed a Series, will align with target object on index. Index values in weights not found in sampled object will be ignored and index values in sampled object not in weights will be assigned weights of zero.
	- If called on a DataFrame, will accept the name of a column when axis = 0. Unless weights are a Series, weights must be same length as axis being sampled.
	- *If weights do not sum to 1, they will be normalized to sum to 1. Missing values in the weights column will be treated as zero. Infinite values not allowed.*
# `iterrows()`
```python
for name, row in Data.iterrows():
	...
```
- Iterate over DataFrame rows as (index, Series) pairs.
# `iteritems()`
```python
{name:ser for name, ser in x_train.iteritems()}
```
- ***Iterate over the DataFrame columns, returning a tuple with the column name and the content as a Series.***
# `Series.iteritems()`, `Series.items()`
```python
for i, value in raw_data["quarter2"].items():
    print(i, value)
```
- Iterate over (index, value) tuples.
# `DataFrame.mean()`
```python
ui.mean(axis=1)
```
# `DataFrame.mean().mean()`
# `DataFrame.add()`, `DataFrame.sub()`, `DataFrame.mul()`, `DataFrame.div()`, `DataFrame.pow()`
```python
adj_ui = ui.sub(user_bias, axis=0).sub(item_bias, axis=1)
```
# `Series.rename()`
```python
plays_df.groupby(["user_id"]).size().rename("n_arts")
```
# `DataFrame.value_counts()`, `Series.value_counts()`
```python
ratings_df["movie_id"].value_counts()
```
# `Series.nunique()`
```python
n_item = ratings_df["movie_id"].nunique()
```
# `Series.isnull()`
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
# `Series.hist()`
# `Series.cumsum()`
```python
cumsum = n_rating_item.cumsum()/len(ratings_df)
```
# `Series.min()`, `Series.max()`, `Series.mean()`, `Series.std()`
# `Series.str`
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
# `Series.cat`
## `Series.cat.categories`
## `Series.cat.set_categories()`
```python
ser.cat.set_categories([2, 3, 1], ordered=True)
```
- 순서 부여
## `Series.cat.codes`
```python
for cat in cats:
    data[cat] = data[cat].cat.codes
```
- Perform label encoding.
# `Series.items()`
```python
for k, v in target.items():
    queries.append(f"{k}-{v}")
```