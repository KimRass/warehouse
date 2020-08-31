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
