Written by KimRass

# Order of Operations
- Context Filters -> `{FIXED:}` -> Dimensition Filters(Actions, Groups) -> `{INCLUDE:}`, `{EXCLUDE:}` -> Blending -> Table Calculations -> Table Calculation Filters(`FIRST()`, `LAST()`) -> Manually Hidden Marks

# Number

# String
## `LEFT()`, `RIGHT()`
```
LEFT("█████",Round([사용하실 측정값]*5,0))
```
## `MID()`
## CONTAINS()
```
CONTAINS([mbr_id], "MBR")
```

# Date
## `DATEDIFF()`
```
DATEDIFF("quarter", [고객별 최초 구매일], [고객별 최초 재구매일 ])
```
## `DATEADD()`
```
DATEADD("month", 1, DATEADD("day", -1, DATE(STR([Year Month]) + "01")))
```
```
IF DAY(TODAY()) < 21
THEN DATETRUNC("month", DATEADD("month", - 1, TODAY())) - 1
ELSE DATETRUNC("month", TODAY()) - 1
END
```
## `DATETRUNC()`
```
DATETRUNC("week", [Order Date])
```
## `DATEPARSE()`
```
DATEPARSE("YYYYMM", STR([연월]))
```

# `Logical`
## `CASE WHEN THEN END`
```
CASE [Sort Criteria Selection]
WHEN "Sales" THEN SUM([Sales])
WHEN "Profit" THEN SUM([Profit])
WHEN "Profit Ratio" THEN [Profit Ratio]
WHEN "Discount" THEN AVG([Discount])
END
```
## `IF ELSEIF ELSE THEN END`
```
IF [검색할 변수] == "처리결과"
THEN [처리결과]
ELSEIF [검색할 변수] == "본부"
THEN [본부]
ELSEIF [검색할 변수] == "팀명"
THEN [팀명]
END
```
## `IN()`
## `ZN()`
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
## `COUNT()`, `COUNTD()`
```
COUNT(IF [계약금1차일자] <= [기준 날짜]
THEN [계약여부]
END)
- COUNT(IF ([계약금1차일자] <= [기준 날짜]
AND [계약여부] = "N")
THEN [계약여부]
END)
```

# Aggregate
## ATTR()
- Return the value of the given expression if only have a single value for all rows in the group, otherwise display `*`. NULL is ignored.
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
- Reference: https://www.youtube.com/watch?v=P-yj-Jzkq_c&list=PLyipEw5AFv5QvjCCYw_ODFTSKVXhkDiQW&index=3
## {INCLUDE:}
```
{INCLUDE [City]:AVG([Sales])}
```
- Compute an aggregate using the specified dimensions and the view dimensions.
- Reference: https://www.youtube.com/watch?v=JW3iIdyT_hM&list=PLyipEw5AFv5RVvw9X4a-Q-LQxbBqsU9Z1&index=4
## {EXCLUDE:}
```
{EXCLUDE [Sub-Category]:AVG([Sales])}
```
- Compute and aggregate excluding the specified dimensions if present in the view.
- Reference: https://www.youtube.com/watch?v=RWIhdRiQ3Ic&list=PLyipEw5AFv5RVvw9X4a-Q-LQxbBqsU9Z1&index=3

# Table Calculation
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
```
WINDOW_AVG(SUM([Sales]), -[Window Selection], 0)
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
## RUNNING_COUNT(), RUNNING_SUM()
```
RUNNING_SUM(COUNTD([Product Name]))/TOTAL(COUNTD([Product Name]))
```
```
RUNNING_SUM(
COUNT(
IF [Pivot Field Names] == "가입일"
THEN [Pivot Field Values]
END))
- RUNNING_SUM(
COUNT(
IF [Pivot Field Names] == "탈퇴일"
THEN [Pivot Field Values]
END))
```
## SCRIPT_BOOL(), SCRIPT_INT(), SCRIPT_REAL(), SCRIPT_STR()
```
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

# Spatial
## DISTANCE()
## MAKEPOINT()

# ELSE
## RANDOM()
```
127 + 2.5*(RANDOM() - 0.2)
```

# Functions
## Custom Number Format
```
"+"0;"-"0
```
```
(+0.0%);(-0.0%)
```
```
(000,000,000)
```
- `Default Properties` -> `Number Format...` -> `Custom`
- `Format` -> `Numbers:` -> `Custom`
- Reference: https://www.youtube.com/watch?v=QhRjOF3M60k
## Histogram
- `Create` -> `Bins...`
- Reference: https://www.youtube.com/watch?v=C1uAQBIPYk4
## Highlight Table
- `Use Separate Legends` <-> `Combine Legends`
- Reference: https://www.youtube.com/watch?v=YXYaDq3qtsw
## Nested Sorting
- `Create` -> `Combined Field`
- Reference: https://www.youtube.com/watch?v=sEUttHntepU
## Copy Format
- `Copy Formatting` -> `Paste Formatting`
## Match Mark Color
- `Mark Cards` -> `Label` -> `Font` -> `Match Mark Color`

# Shortcut Keys
## Ctrl + W
- Swap Rows and Columns.
## Ctrl + M
- New Worksheet.
## Alt + Shift + Backspace
- Clear Worksheet.
## Alt + I
- `Filters`
## Ctrl + F
- Search.
## Ctrl + Tab, Ctrl + PgDn
- Move to next Worksheet, Dashboard, or Story.
## Ctrl + Shift + Tab, Ctrl + PgUp
- Move to previous Worksheet Dashboard, or Story.
## Ctrl + B, Ctrl + Shift + B
- Zoom Out or Zoom In respectively.
## Alt + Shift + (C, X, F, or T)
- Move Fileds to Columns, Rows, Filters or Text respectively.
## Ctrl + H
- Presentation Mode.
## Alt + A -> C
- `Create Calculate Field...`
## ALt + A -> O -> G(T)
- `Show Row(Column) Grand Totals`
## Alt + F -> A
- `Save As...`
## Alt + D -> R
- `Edit Blend Relationships...`
## Alt + D -> X
- `Refresh All Extracts...`
## Alt + W -> O
- `Tooltip...`
## Alt + W -> C -> I, Alt + B -> C
- `Image...`, `Copy Image` respectively
## Alt + B -> X
- `Export Image...`
## Alt + S -> W
- `Publish Workbook...`
## Alt + O -> D or W
- `Dashboard...` or `Workbook...`

# `Publish Workbook to Tableau Server`
## `More Options`
### `Show sheets as tabs`
### `Show selections`
### `Include external files`

# URL Parameters
## `:showAppBanner=false`
## `:display_count=n`
## `:showVizHome=n`
## `:origin=viz_share_link`
## `:embed=yes`
## `:toolbar=top`

# Tableau Server
## `Server Status`
### `Background Tasks for Extracts`

# Options
## `Dashboard`
- Uncheck `Add Phone Layouts to New Dashboards`
## `Worksheet`
- Uncheck `Show Sort Contols`
## `Analysis`
- `Table Layout` -> Toggle `Show Field Labels for Rows`

# `tableauserverclient`
```python
import tableauserverclient as TSC
```
## `TSC.PersonalAccessTokenAuth()`
```python
tableau_auth = TSC.PersonalAccessTokenAuth("admin_TOKEN", token)
```
## `TSC.Server()`
```python
server = TSC.Server("http://218.153.56.75/", use_server_version=True)
```
### `server.auth`
#### `server.auth.sign_in()`
```python
with server.auth.sign_in(tableau_auth):
```
#### `server.auth.sign_out()`
### `server.projects`
#### `server.projects.get()`
### `server.groups`
#### `server.groups.get()`
```python
all_groups, pagination_item = server.groups.get(req_options=req_opts)
```
#### `server.groups.populate_users()`
```python
group_user = list()
with server.auth.sign_in(tableau_auth):
    groups, pagination_item = server.groups.get(req_options=req_opts)
    for group in groups[1:]:
        pagination_item = server.groups.populate_users(group, req_options=req_opts)
        group_user.extend([(group.name,) + empls[user.name] + (user.site_role,) if user.name in empls.keys() else (group.name, None, None, None, user.name, user.site_role) for user in group.users])
```
#### `server.groups.create()`
### `server.users`
#### `server.users.add()`
#### `servers.users.update()`
#### `server.users.remove()`
### `server.workbooks`
## `TSC.GroupItem()`
## `TSC.UserItem`
### `TSC.UserItem.Roles`
## `TSC.RequestOptions()`
```python
req_opts = TSC.RequestOptions(pagesize=1000)
```
#### `TSC.UserItem.Roles.Creator`, `TSC.UserItem.Roles.Explorer`, `TSC.UserItem.Roles.ExplorerCanPublish`, `TSC.UserItem.Roles.ServerAdministrator`, `TSC.UserItem.Roles.SiteAdministratorCreator`, `TSC.UserItem.Roles.Unlicensed`, `TSC.UserItem.Roles.ReadOnly`, `TSC.UserItem.Roles.Viewer`