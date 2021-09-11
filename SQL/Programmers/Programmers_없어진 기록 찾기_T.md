https://programmers.co.kr/learn/courses/30/lessons/59042

SELECT animal_outs.animal_id, animal_outs.name
FROM animal_outs
LEFT OUTER JOIN animal_ins
ON animal_ins.animal_id = animal_outs.animal_id
WHERE animal_ins.animal_id IS NULL
ORDER BY animal_id;