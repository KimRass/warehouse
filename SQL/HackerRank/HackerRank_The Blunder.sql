- Source: https://www.hackerrank.com/challenges/the-blunder/problem?isFullScreen=true
```sql
SELECT CAST(CEILING(AVG(CAST(salary AS FLOAT)) - AVG(CAST(REPLACE(CAST(salary AS FLOAT), "0", "") AS FLOAT))) AS INT)
FROM employees;
```