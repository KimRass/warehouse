# pandas
```python
import pandas as pd
```
## pd.set_option()
```python
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
```
## pd.DataFrame()
```python
list_for_data = [(re.match(r"(\[)(\w+)(\])", line[0]).group(2), re.findall(r"(\d\] )(.*)$", line[0])[0][1]) for line in raw_data if re.match(r"(\[)(\w+)(\])", line[0])]
data = pd.DataFrame(list_for_data, columns=["user", "content"])
```
## pd.read_csv()
```python
raw_data = pd.read_csv("C:/Users/00006363/☆데이터/실거래가_충청북도_2014.csv", thousands=",", encoding="Ansi", skiprows=15)
```
## pd.read_excel()
## pd.read_pickle()
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
## df.merge()
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
## df.str
### df.str.replace()
```python
data["parking"] = data["parking"].str.replace("대", "", regex=False)
```
### split()
```python
data["buildings"] = data.apply(lambda x : str(x["houses_buildings"]).split("총")[1][:-3], axis=1)
```
- .split()는 .split(" ")과 같음
### df.str.strip()
```python
data["fuel"] = data["heating"].apply(lambda x:x.split(",")[1]).str.strip()
```
### df.str.contains()
```python
raw_data[raw_data["시군구"].str.contains("충청북도")]
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
## df.drop()
```python
data = data.drop(["Unnamed: 0", "address1", "address2"], axis=1)
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
data = data.sample(frac=1)
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
## ser.astype()
- "int32", "int63", "float64", "object", "category", 
## ser.hist()
## ser.cumsum()
```python
cumsum = n_rating_item.cumsum()/len(ratings_df)
```
## ser.min(), ser.max(), ser.mean(), ser.std()
