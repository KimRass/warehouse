- Source: https://sqlzoo.net/wiki/SELECT_within_SELECT_Tutorial
# 1
```sql
SELECT name
FROM world
WHERE population > (
	SELECT population
	FROM world
	WHERE name = 'Russia')
```
# 5
```sql
SELECT name, CONCAT(ROUND(population/(
	SELECT population
	FROM world
	WHERE name = 'Germany')*100, 0), '%')
FROM world
WHERE continent = 'Europe'
```
# 10*
```sql
SELECT wo2.name, wo2.continent
FROM world AS wo2, (
	SELECT continent, MAX(population) AS mx2
	FROM (
		SELECT wo.name, wo.continent, wo.population
		FROM world AS wo, (
			SELECT continent, MAX(population) AS mx
			FROM world
			GROUP BY continent) AS A
			WHERE wo.continent = A.continent AND wo.population != A.mx) AS B
	GROUP BY continent) AS C
WHERE wo2.continent = C.continent AND wo2.population > 3*C.mx2
```