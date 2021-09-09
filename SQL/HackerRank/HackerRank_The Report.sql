- Source: https://www.hackerrank.com/challenges/the-report/problem?isFullScreen=true
```sql
SELECT (CASE WHEN gr.grade >= 8 THEN st.name ELSE NULL END), gr.grade, st.marks
FROM students AS st, grades AS gr
WHERE st.marks BETWEEN gr.min_mark AND gr.max_mark
ORDER BY gr.grade DESC, st.name ASC, st.marks ASC;
```