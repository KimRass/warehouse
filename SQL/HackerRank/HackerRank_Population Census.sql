- Source: https://www.hackerrank.com/challenges/asian-population/problem?isFullScreen=true
```sql
SELECT SUM(ci.population)
FROM city AS ci, country AS co
WHERE ci.countrycode = co.code AND co.continent = "Asia";
```