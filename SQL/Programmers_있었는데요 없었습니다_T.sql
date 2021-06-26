https://programmers.co.kr/learn/courses/30/lessons/59043

SELECT animal_ins.animal_id, animal_ins.name
FROM animal_ins
INNER JOIN animal_outs
ON animal_ins.animal_id = animal_outs.animal_id
WHERE animal_ins.datetime > animal_outs.datetime
ORDER BY animal_ins.datetime;