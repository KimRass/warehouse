https://programmers.co.kr/learn/courses/30/lessons/59412

SELECT HOUR(datetime) HOUR, COUNT(*)
FROM animal_outs
GROUP BY HOUR(datetime)
HAVING HOUR BETWEEN 9 AND 19
ORDER BY HOUR;