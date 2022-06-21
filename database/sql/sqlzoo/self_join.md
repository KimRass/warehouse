- Source: https://sqlzoo.net/wiki/Self_join
# 1
```sql
SELECT COUNT(*)
FROM stops
```
# 2
```sql
SELECT id
FROM stops
WHERE name = 'Craiglockhart'
```
# 3
```sql
SELECT id, name
FROM stops
WHERE id IN (
	SELECT stop
	FROM route
	WHERE num = '4' AND company = 'LRT')
```
# 4
```sql
SELECT company, num, COUNT(*)
FROM route
WHERE stop = 149 OR stop = 53
GROUP BY company, num
HAVING COUNT(*) = 2
```
# 5
```sql
SELECT ro1.company, ro1.num, ro1.stop, ro2.stop
FROM route AS ro1 INNER JOIN route AS ro2 ON (ro1.num = ro2.num AND ro1.company = ro2.company)
WHERE ro1.stop = 53 AND ro2.stop = 149
ORDER BY ro1.stop, ro2.stop
```
# 6
```sql
SELECT ro1.company, ro1.num, st1.name, st2.name
FROM route AS ro1 INNER JOIN route AS ro2 ON (ro1.company = ro2.company AND ro1.num = ro2.num) INNER JOIN stops AS st1 ON (ro1.stop = st1.id) INNER JOIN stops AS st2 ON (ro2.stop = st2.id)
WHERE st1.name = 'Craiglockhart' AND st2.name = 'London Road'
```
# 7
```sql
SELECT DISTINCT ro1.company, ro1.num
FROM route AS ro1 INNER JOIN route AS ro2 ON ro1.num = ro2.num AND ro2.company = ro2.company
WHERE ro1.stop = 115 ANd ro2.stop = 137
```