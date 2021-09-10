- Source: https://sqlzoo.net/wiki/More_JOIN_operations
# 6
```sql
SELECT ac.name
FROM casting AS ca, actor AS ac
WHERE movieid IN (
	SELECT id
	FROM movie
	WHERE title = 'Casablanca') AND ca.actorid = ac.id
```
# 11
```sql
SELECT yr, COUNT(*)
FROM movie
WHERE id IN (
	SELECT movieid
	FROM casting
	WHERE actorid = (
		SELECT id
		FROM actor
		WHERE name = 'Rock Hudson'))
GROUP BY yr
HAVING COUNT(*) > 2
```
# 12
```sql
SELECT mo.title, ac.name
FROM casting AS ca, movie AS mo, actor AS ac
WHERE movieid IN (
	SELECT movieid
	FROM casting
	WHERE actorid = (
		SELECT id
		FROM actor
		WHERE name = 'Julie Andrews')) AND ord = 1 AND ca.movieid = mo.id AND ca.actorid = ac.id
```
# 13
```sql
SELECT name
FROM actor
WHERE id IN (
	SELECT actorid
	FROM casting
	WHERE ord = 1
	GROUP BY actorid
	HAVING COUNT(*) >= 15)
ORDER BY name ASC;
```
# 14
```sql
SELECT title, A.cnt
FROM movie AS mo, (
	SELECT movieid, COUNT(*) AS cnt
	FROM casting
	GROUP BY movieid) AS A
WHERE mo.id = A.movieid AND mo.yr = 1978
ORDER BY A.cnt DESC, title ASC;
```
# 15
```sql
SELECT name
FROM actor
WHERE id IN (
	SELECT actorid
	FROM casting
	WHERE movieid IN (
		SELECT movieid
		FROM casting
		WHERE actorid IN (
			SELECT id
			FROM actor
			WHERE name = 'Art Garfunkel')));
```