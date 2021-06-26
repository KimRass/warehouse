https://programmers.co.kr/learn/courses/30/lessons/59410

SELECT animal_type,
IFNULL(name, "No name"),
sex_upon_intake
FROM animal_ins
ORDER BY animal_id;