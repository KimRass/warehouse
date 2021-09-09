- Source: https://www.hackerrank.com/challenges/contest-leaderboard/problem?isFullScreen=true
```sql
SELECT hacker_id, name, SUM(score)
FROM (
    SELECT su.hacker_id, ha.name, MAX(su.score) AS score
    FROM submissions AS su INNER JOIN hackers as ha ON su.hacker_id = ha.hacker_id
    GROUP BY su.hacker_id, ha.name, su.challenge_id) AS A
GROUP BY hacker_id, name
HAVING SUM(score) > 0
ORDER BY SUM(score) DESC, hacker_id ASC;
```