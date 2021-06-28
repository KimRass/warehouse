https://programmers.co.kr/learn/courses/30/lessons/77487

SELECT places.id, places.name, places.host_id
FROM places
RIGHT OUTER JOIN (SELECT host_id
FROM places
GROUP BY host_id
HAVING COUNT(id) >= 2)
AS heavy
ON places.host_id = heavy.host_id
ORDER BY id;