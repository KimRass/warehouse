- Source: https://www.hackerrank.com/challenges/challenges/problem?isFullScreen=true
```sql
SELECT ch.hacker_id, ha.name, COUNT(ch.challenge_id)
FROM challenges AS ch INNER JOIN hackers AS ha ON ch.hacker_id = ha.hacker_id
GROUP BY ch.hacker_id, ha.name
HAVING COUNT(ch.challenge_id) IN (
    SELECT tot
    FROM (
        SELECT COUNT(challenge_id) AS tot
        FROM challenges
        GROUP BY hacker_id) AS B
    GROUP BY tot
    HAVING COUNT(tot) = 1 OR tot = (SELECT MAX(A.tot)
                                    FROM (SELECT COUNT(challenge_id) AS tot
                                          FROM challenges
                                          GROUP BY hacker_id) AS A))
ORDER BY COUNT(ch.challenge_id) DESC, hacker_id ASC;
```