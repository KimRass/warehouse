- Source: https://sqlzoo.net/wiki/The_JOIN_operation
# 1
```sql
SELECT matchid, player
FROM goal
WHERE teamid = 'GER'
```
# 9
```sql
SELECT et.teamname, A.cnt
FROM (
	SELECT teamid, COUNT(*) AS cnt
	FROM goal
	GROUP BY teamid) AS A INNER JOIN eteam AS et ON A.teamid = et.id
ORDER BY et.teamname
```
# 10
```sql
SELECT ga.stadium, A.cnt
FROM game AS ga INNER JOIN (
	SELECT matchid, COUNT(*) AS cnt
	FROM goal
	GROUP BY matchid) AS A ON ga.id = A.matchid
```
# 11
```sql
SELECT A.matchid, B.mdate, A.cnt
FROM (
	SELECT matchid, COUNT(*) AS cnt
	FROM goal
	WHERE matchid IN (
		SELECT id
		FROM game
		WHERE team1 = 'POL' OR team2 = 'POL')
	GROUP BY matchid) AS A INNER JOIN (
	SELECT id, mdate
	FROM game
	WHERE team1 = 'POL' OR team2 = 'POL') AS B ON A.matchid = B.id
```
# 12
```sql
SELECT A.matchid, ga.mdate, A.cnt
FROM game AS ga, (
	SELECT matchid, COUNT(*) AS cnt
	FROM goal
	WHERE teamid = 'GER'
	GROUP BY matchid) AS A
WHERE A.matchid = ga.id
```
# 13
```sql
SELECT mdate, team1, COALESCE(SUM(cnt1), 0), team2, COALESCE(SUM(cnt2), 0)
FROM (
	SELECT ga.id, ga.mdate, ga.team1, (CASE WHEN ga.team1 = go.teamid THEN 1 END) AS cnt1, ga.team2, (CASE WHEN 
ga.team2 = go.teamid THEN 1 END) AS cnt2
	FROM goal AS go INNER JOIN game AS ga ON go.matchid = ga.id) AS A
GROUP BY id, mdate, team1, team2
ORDER BY mdate, id, team1, team2
```