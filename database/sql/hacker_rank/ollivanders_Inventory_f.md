- Source: https://www.hackerrank.com/challenges/harry-potter-and-wands/problem?isFullScreen=true&h_r=next-challenge&h_v=zen
```sql
SELECT id, age, coins, power
FROM (
    SELECT pr.age, wa.power, MIN(wa.coins_needed) AS coins
    FROM wands as wa, wands_property as pr
    WHERE wa.code = pr.code
    GROUP BY pr.age, wa.power
    ORDER BY power DESC, age DESC) AS A
WH
```