# Date
## DATEDIFF()
```
DATEDIFF("quarter", [고객별 최초 구매일], [고객별 최초 재구매일 ])
```
## DATEADD()
```
DATEADD("month", 1, DATEADD("day", -1, DATE(STR([Year Month]) + "01")))
```
## DATETRUNC()
```
DATETRUNC("week", [Order Date])
```



# Logical
## IF, ELSEIF, ELSE, THEN, END
```
IF [검색할 변수] == "처리결과"
THEN [처리결과]
ELSEIF [검색할 변수] == "본부"
THEN [본부]
ELSEIF [검색할 변수] == "팀명"
THEN [팀명]
END
```
## ZN()
```
SUM(IF (DATEDIFF("year", [계약금1차일자], [기준 날짜]) == 0
AND [계약금1차일자] <= [기준 날짜])
THEN [_]
END)
- ZN(SUM(IF (DATEDIFF("year", [계약금1차일자], [기준 날짜]) == 0
AND [해약일자] <= [기준 날짜]
AND [계약여부] == "N")
THEN [_]
END))
```
## COUNT()
```
COUNT(IF [계약금1차일자] <= [기준 날짜]
THEN [계약여부]
END)
- COUNT(IF ([계약금1차일자] <= [기준 날짜]
AND [계약여부] = "N")
THEN [계약여부]
END)
```
## COUNTD()
```
COUNTD(
IF [분양 완료?] == "분양 진행"
THEN [현장명]
END)
```



# Aggregate
## AVG()
## {FIXED:}
```
([기준 날짜] >= [분양일])
AND ([기준 날짜] <= [완공예정])
OR
([기준 날짜] > [완공예정]
AND {FIXED [사업지], [분양구분], [분양종류]:[현재 미분양]} > 0)
```
```
{FIXED [본부], [팀명]:
COUNT(IF YEAR([접수일]) == 2021
THEN 1
ELSE NULL
END)}
```
```
IF {COUNT([Orders])}
== {FIXED [Region]:COUNT([Orders])}
THEN [State]
ELSE [Region]
END
```
- Compute an aggregate using only the specified dimensions.
- Context Filters -> {FIXED:} -> Dimensition Filters(Actions, Groups)
- Reference: https://www.youtube.com/watch?v=P-yj-Jzkq_c&list=PLyipEw5AFv5QvjCCYw_ODFTSKVXhkDiQW&index=3



# Table Calculation
- Dimensition Filters(Actions, Groups) -> Table Calculations -> Table Calculation Filters(FIRST(), LAST())
## FIRST(), LAST()
```
LAST() <= 23
```
- Return the number of rows from the current row to the first(last) row in the partition.
- Reference: https://www.youtube.com/watch?v=k41o1m9xsR8&list=PLyipEw5AFv5QvjCCYw_ODFTSKVXhkDiQW&index=4&t=47s
## INDEX()
- Return the index of the current row in the partition.
## SIZE()
- Return the number of rows in the partition.
## WINDOW_AVG(), WINDOW_COUNT(), WINDOW_MAX(), WINDOW_MIN(), WINDOW_SUM()
```
(SUM([Sales]) - WINDOW_MIN(SUM([Sales])))/(WINDOW_MAX(SUM([Sales])) - WINDOW_MIN(SUM([Sales])))
```
- Reference: https://www.youtube.com/watch?v=8YMWH_ozstE&t=610s
## RANK()
```
RANK(SUM([Size]))
```
## TOTAL()
## LOOKUP()
```
LOOKUP(ATTR([Customer Name]), 0)
```
- Return the value of the given expression in a target row, specified as a relative offset from the current row.
- Reference: https://www.youtube.com/watch?v=IRZAbkrkj60&list=PLyipEw5AFv5QvjCCYw_ODFTSKVXhkDiQW&index=4&t=22s
## SCRIPT_STR()
``````
RIGHT(
SCRIPT_STR(
"import urllib
from bs4 import BeautifulSoup as bs
url = f'http://www.opinet.co.kr/api/avgLastWeek.do?prodcd=B027&code=F916210128&out=xml'
xml = urllib.request.urlopen(url).read().decode('utf8')
soup = bs(xml, 'lxml')
return soup.find('sta_dt').get_text()",
ATTR([댓글생성자id])), 6)
```