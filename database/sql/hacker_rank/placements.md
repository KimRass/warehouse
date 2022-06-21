- Source: https://www.hackerrank.com/challenges/placements/problem?isFullScreen=true
```sql
SELECT st.name
FROM students AS st INNER JOIN friends AS fr ON st.id = fr.id INNER JOIN packages AS pa1 ON fr.id = pa1.id INNER JOIN packages AS pa2 ON fr.friend_id = pa2.id
WHERE pa2.salary > pa1.salary
ORDER BY pa2.salary ASC;
```