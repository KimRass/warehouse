# Concept
## PRIMARY KEY
- The `PRIMARY KEY` constraint uniquely identifies each record in a table. Primary keys must contain UNIQUE values, and cannot contain NULL values. A table can have only ONE primary key; and in the table, this primary key can consist of single or multiple columns (fields).
## FOREIGN KEY
- The `FOREIGN KEY` constraint is used to prevent actions that would destroy links between tables. A `FOREIGN KEY` is a field (or collection of fields) in one table, that refers to the `PRIMARY KEY` in another table. The table with the foreign key is called the child table, and the table with the primary key is called the referenced or parent table.
- The `FOREIGN KEY` constraint prevents invalid data from being inserted into the foreign key column, because it has to be one of the values contained in the parent table.



# Data Types
## String
- `CHAR(size)`: A fixed length string.
- `VARCHAR(size)`: A variable length string.
## Numeric
- `BOOL`: Zero is considered as false, nonzero values are considered as true.
- `INT(size)`: A medium integer. Signed range is from -2147483648 to 2147483647. Unsigned range is from 0 to 4294967295. The `size` parameter specifies the maximum display width (which is 255).
- `DOUBLE(size, n)`: A normal-size floating point number. The total number of digits is specified in `size`. The number of digits after the decimal point is specified in the `d` parameter.
## Data and Time
- `DATE`: A date. Format: `YYYY-MM-DD`.
- `TIME`: A time. Format: `hh:mm:ss`.
- `DATETIME`: A date and time combination. Format: `YYYY-MM-DD hh:mm:ss`.



# ORDER
- ORDER BY -> LIMIT
- GROUP BY -> HAVING
- WHERE -> GROUP BY



# COLUMN_NAME
```sql
SELECT COLUMN_NAME
FROM INFORMATION_SCHEMA.COLUMNS;
```
# IN
```sql
SELECT name, addr
FROM usertbl
WHERE addr IN ("서울", "경기", "충청");
```
# BETWEEN AND
```sql
SELECT name, height
FROM sqlDB.usertbl
WHERE height BETWEEN 180 AND 183;
```
# LIKE
```sql
SELECT DISTINCT city
FROM station
WHERE city LIKE "%a" OR city LIke "%e" OR city LIKE "%i" OR city LIKE "%o" OR city LIKE "%u";
```
- `%` represents zero, one, or multiple characters.
- `_` represents one, single character.
# GROUP BY HAVING
- `GROUP BY`를 통해 만들어진 Groups에만 조건을 적용.
```sql
SELECT animal_type, COUNT(animal_id)
FROM animal_ins
GROUP BY animal_type
HAVING animal_type in ("Cat", "Dog")
ORDER BY animal_type;
```
## IS NULL, IS NOT NULL
```sql
SELECT animal_id
FROM animal_ins
WHERE name IS NULL;
```
## CASE WHEN THEN ELSE END
- Search for conditions sequentially.
```sql
SELECT CASE WHEN (a >= b + c OR b >= c + a OR c >= a + b) THEN "Not A Triangle" WHEN (a = b AND b = c) THEN "Equilateral" WHEN (a = b OR b = c OR c = a) THEN "Isosceles" ELSE "Scalene" END
FROM triangles;
```
```sql
SELECT
    n,
    CASE
    WHEN p IS NULL
    THEN " Root"
    WHEN n IN (SELECT DISTINCT p FROM bst)
    THEN " Inner"
    ELSE " Leaf"
    END
FROM
    bst
ORDER BY
    n;
```
# INNER JOIN ON, LEFT OUTER JOIN ON
```sql
SELECT
	animal_ins.animal_id, animal_ins.name
FROM
	animal_ins
	INNER JOIN animal_outs
	ON animal_ins.animal_id = animal_outs.animal_id
WHERE
	animal_ins.datetime > animal_outs.datetime
ORDER BY
	animal_ins.datetime;
```
```sql
SELECT
	animal_ins.name,
	animal_ins.datetime
FROM
	animal_ins
	LEFT OUTER JOIN animal_outs	ON animal_ins.animal_id = animal_outs.animal_id
WHERE
	animal_outs.datetime IS NULL
ORDER BY
	animal_ins.datetime ASC
LIMIT 3;
```
## CONCAT()
```sql
SELECT
CONCAT(CASE hdc_mbr.mbr_mst.mbr_sex
WHEN "0" THEN "남자"
WHEN "`" THEN "여자"
END, "/", hdc_mbr.mbr_mst.mbr_birth)
```
## HOUR()
```sql
SELECT HOUR(datetime) AS HOUR, COUNT(*)
FROM animal_outs
GROUP BY HOUR
HAVING HOUR BETWEEN 9 AND 19
ORDER BY HOUR;
```
- HOUR()는 일반조건이므로 (COUNT()와 달리) HAIVNG과 함께 쓸 수 없다.
# ANY()
```sql
SELECT name
FROM usertbl
WHERE height > ANY (SELECT height FROM usertbl WHERE addr="서울");
```
## IFNULL()
```sql
SELECT animal_type,
IFNULL(name, "No name"),
sex_upon_intake
FROM animal_ins
ORDER BY animal_id
```
## DATE_FORMAT()
```sql
SELECT animal_id, name, DATE_FORMAT(datetime, "%Y-%m-%d")
FROM animal_ins
ORDER BY animal_id;
```
## CONCAT_WS()
```sql
SELECT CONCAT_WS("/", "2020", "01", "12");
```
## FORMAT()
```sql
SELECT FORMAT(123456.123456, 4);
```
- 출력할 소수점 이하 자릿수 지정. 
## INSERT()
```sql
SELECT INSERT("abcdefghi", 3, 2, "####");
```
## LEFT(), RIGHT(), SUBSTRING()
```sql
SELECT DISTINCT city
FROM station
WHERE RIGHT(city, 1) IN ("a", "e", "i", "o", "u");
```
## LOWER(), UPPER(), INITCAP()
### LPAD(), RPAD()
```sql
SELECT LPAD("이것이", 5, "##")
```
## LTRIM(), RTRIM()
```sql
SELECT LTRIM("    좌측공백제거");
```
## TRIM()
### BOTH, LEADING, TRAILING
```sql
SELECT TRIM("   좌우측공백삭제   ";
SELECT TRIM(BOTH "ㅋ" FROM "ㅋㅋㅋ좌우측문자삭제ㅋㅋㅋ");
SELECT TRIM(LEADING "ㅋ" FROM "ㅋㅋㅋ좌측문자삭제ㅋㅋㅋ");
SELECT TRIM(TRAILING "ㅋ" FROM "ㅋㅋㅋ우측문자삭제ㅋㅋㅋ");
```
## TO_CHAR()
```sql
TO_CHAR(mbrmst.mbr_leaving_expected_date, "YYYY/MM/DD") AS mbr_leaving_expected_dt
```
## INSERT INTO VALUES
```sql
INSERT INTO customers 
(customername, address, city, postalcode, country)
VALUES ("Hekkan Burger", "Gateveien 15", "Sandnes", "4306", "Norway");
```
- If you are adding values for all the columns of the table, you do not need to specify the column names in the SQL query.
### REFERENCES
```sql
CREATE TABLE orders
(orderid INT NOT NULL,
ordernumber INT NOT NULL,
personid INT,
PRIMARY KEY (orderid),
FOREIGN KEY (personid) REFERENCES persons(personid));
```
## UNION, UNION ALL
- `UNION` selects only distinct values by default. To allow duplicate values, use `UNION ALL`
## CAST()
```sql
SELECT
    name + "(" + LEFT(occupation, 1) + ")"
FROM
    occupations
ORDER BY
    name;
SELECT
    "There are a total of " + CAST(COUNT(occupation) AS CHAR) + " " + LOWER(occupation) + "s."
FROM
    occupations
GROUP BY
    occupation
ORDER BY
    COUNT(occupation),
    LOWER(occupation);
```
## PIVOT, UNPIVOT
```sql
SELECT 반정보, 과목, 점수
FROM dbo.성적	UNPIVOT (점수	FOR 과목 IN (국어, 수학, 영어)) AS UNPVT
```
- `PIVOT` transforms rows to columns.
- `UNPIVOT` transforms columns to rows.
# TOP (for MS SQL Server), LIMIT (for MySQL)
```sql
SELECT TOP 1 months*salary, COUNT(employee_id)
FROM employee
GROUP BY months*salary
ORDER BY months*salary DESC;
```