- Source: https://www.hackerrank.com/challenges/symmetric-pairs/problem?isFullScreen=true
```sql
SELECT x, y
FROM functions
WHERE CONCAT(x, " ", y) IN (SELECT CONCAT(y, " ", x) FROM functions) AND x <= y
ORDER BY x, y;
```
```sql
SELECT A.x, A.y
FROM functions AS A INNER JOIN functions AS B ON A.y = B.x
WHERE A.x = B.y AND A.x <= A.y
ORDER BY A.x ASC;
```