- Source: https://www.hackerrank.com/challenges/sql-projects/problem?isFullScreen=true
```sql
SELECT A.start_date, MIN(B.end_date)
FROM (SELECT start_date FROM projects WHERE start_date NOT IN (SELECT end_date FROM projects)) AS A, (SELECT end_date FROM projects WHERE end_date NOT IN (SELECT start_date FROM projects)) AS B
WHERE A.start_date < B.end_date
GROUP BY A.start_date
ORDER BY DATEDIFF(DAY, A.start_date, MIN(B.end_date)) ASC, A.start_date ASC;
```