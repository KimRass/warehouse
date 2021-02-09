## window_avg
```
- sum([]) = window_avg(sum([]))
```
## fixed, 
```
sum([사내 외])/sum({fixed [부서] : sum([피벗 필드 값])})
```
## if + or
```
if [피벗 필드명] = "재택"
or [피벗 필드명] = "외근"
or [피벗 필드명] = "휴가"
then [피벗 필드 값]
end
```
## zn
```
ZN(SUM([종가])) - LOOKUP(ZN(SUM([종가])), -1)
```
## datediff
```
DATEDIFF('quarter', [고객별 최초 구매일], [고객별 최초 재구매일 ])
```
# dateutil
## relativedelta
from dateutil import relativedelta
```
(start + relativedelta.relativedelta(months=1)).replace(day=1) - timedelta(days=1)
```
