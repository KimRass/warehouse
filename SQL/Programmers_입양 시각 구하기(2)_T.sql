https://programmers.co.kr/learn/courses/30/lessons/59413

SET @hour := -1;

SELECT (@hour := @hour + 1) as "hour",
(SELECT COUNT(*) FROM animal_outs WHERE @hour = HOUR(datetime)) as "count"
FROM animal_outs
WHERE @hour < 23;