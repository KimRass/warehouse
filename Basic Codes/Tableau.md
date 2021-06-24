# Date
## DATEDIFF
```
DATEDIFF("quarter", [고객별 최초 구매일], [고객별 최초 재구매일 ])
```
## DATEADD
```
DATEADD("month", 1, DATEADD("day", -1, DATE(STR([Year Month]) + "01")))
```



# Logical
## IF, ELSEIF, ELSE, THEN, END
```
IF [검색할 변수] = "처리결과"
THEN [처리결과]
ELSEIF [검색할 변수] = "본부"
THEN [본부]
ELSEIF [검색할 변수] = "팀명"
THEN [팀명]
END
```
## ZN
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
## COUNTD
```
COUNTD(
IF [분양 완료?] = "분양 진행"
THEN [현장명]
END)
```



# Aggregate
## AVG
## FIXED
```
([기준 날짜] >= [분양일])
AND ([기준 날짜] <= [완공예정])
OR
([기준 날짜] > [완공예정]
AND {FIXED [사업지], [분양구분], [분양종류]:[현재 미분양]} > 0)
```



# Table Calculation
## WINDOW_AVG
## INDEX
## RANK
## SCRIPT_STR
```
RANK(SUM([Size]))
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