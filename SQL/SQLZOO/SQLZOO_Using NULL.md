- Source: https://sqlzoo.net/wiki/Using_Null
# 1
```sql
SELECT name
FROM teacher
WHERE dept IS NULL
```
# 5
```sql
SELECT name, COALESCE(mobile, '07986 444 2266')
FROM teacher
```
# 6
```sql
SELECT te.name, COALESCE(de.name, 'None')
FROM teacher AS te LEFT JOIN dept AS de ON te.dept = de.id
```
# 8
```sql
SELECT de.name, COALESCE(A.cnt, 0)
FROM dept AS de LEFT JOIN (
	SELECT dept, COUNT(*) AS cnt
	FROM teacher
	GROUP BY dept) AS A ON de.id = A.dept
 ```
 # 10
 ```sql
SELECT name, (CASE WHEN (dept = 1 OR dept = 2) THEN 'Sci' WHEN dept = 3 THEN 'Art' ELSE 'None' END)
FROM teacher
```