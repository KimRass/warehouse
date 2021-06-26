https://programmers.co.kr/learn/courses/30/lessons/59044

SELECT animal_ins.name, animal_ins.datetime
FROM animal_ins
LEFT OUTER JOIN animal_outs
ON animal_ins.animal_id = animal_outs.animal_id
WHERE animal_outs.datetime IS NULL
ORDER BY animal_ins.datetime ASC
LIMIT 3;