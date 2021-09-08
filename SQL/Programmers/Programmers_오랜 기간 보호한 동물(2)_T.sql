https://programmers.co.kr/learn/courses/30/lessons/59411

SELECT animal_outs.animal_id, animal_outs.name
FROM animal_outs
LEFT OUTER JOIN animal_ins
ON animal_outs.animal_id = animal_ins.animal_id
ORDER BY animal_outs.datetime - animal_ins.datetime DESC
LIMIT 2;