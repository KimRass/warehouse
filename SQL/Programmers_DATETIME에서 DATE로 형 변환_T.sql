https://programmers.co.kr/learn/courses/30/lessons/59414

SELECT animal_id, name, DATE_FORMAT(datetime, "%Y-%m-%d")
FROM animal_ins
ORDER BY animal_id;