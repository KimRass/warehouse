https://programmers.co.kr/learn/courses/30/lessons/59045

SELECT animal_ins.animal_id, animal_ins.animal_type, animal_ins.name
FROM animal_ins
INNER JOIN animal_outs
ON animal_ins.animal_id = animal_outs.animal_id
WHERE animal_ins.sex_upon_intake LIKE "Intact%"
AND (animal_outs.sex_upon_outcome LIKE "Spayed%"
OR animal_outs.sex_upon_outcome LIKE "Neutered%")
ORDER BY animal_ins.animal_id;