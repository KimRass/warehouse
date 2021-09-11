https://programmers.co.kr/learn/courses/30/lessons/59047

SELECT animal_id, name
FROM animal_ins
WHERE animal_type = "Dog"
AND UPPER(name) LIKE "%EL%"
ORDER BY name;