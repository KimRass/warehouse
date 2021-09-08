https://programmers.co.kr/learn/courses/30/lessons/59040

SELECT animal_type, COUNT(animal_id)
FROM animal_ins
GROUP BY animal_type
HAVING animal_type in ("Cat", "Dog")
ORDER BY animal_type;