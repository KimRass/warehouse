# Data Types
## String
- `CHAR(size)`: A fixed length string.
- `VARCHAR(size)`: A variable length string.
## Numeric
- `BOOL`: Zero is considered as false, nonzero values are considered as true.
- `INT`
- `FLOAT(n)`
## Data and Time
- `DATE`: A date. Format: `YYYY-MM-DD`.
- `TIME`: A time. Format: `hh:mm:ss`.
- `DATETIME`: A date and time combination. Format: `YYYY-MM-DD hh:mm:ss`.



# ORDER
- ORDER BY -> LIMIT
- GROUP BY -> HAVING
- WHERE -> GROUP BY



# Error Messages
## Column <<COLUMN1>> is invalid in the ORDER BY clause because it is not contained in either an aggregate function or the GROUP BY clause.



# DDL(Data Definition Language)
## CREATE TABLE
### CREATE TABLE PRIMARY KEY
### CREATE TABLE FOREIGN KEY REFERENCES
```sql
CREATE TABLE orders (order_id INT NOT NULL, order_no INT NOT NULL,
person_id INT, PRIMARY KEY(order_id), FOREIGN KEY(person_id) REFERENCES persons(person_id));
```
## RENAME TABLE TO
```sql
RENAME TABLE old_table1 TO new_table1,
	old_table2 TO new_table2,
	old_table3 TO new_table3;
```
```sql
RENAME TABLE current_db.table_name TO other_db.table_name;
```
## DROP TABLE
## DROP TABLE IF EXISTS
## ALTER TABLE
### ALTER TABLE ADD
```sql
ALTER TABLE users
	ADD birth_date CHAR(6) NULL;
```
#### ALTER TABLE ADD FIRST
#### ALTER TABLE ADD AFTER
### ALTER TABLE MODIFY
#### ALTER TABLE MODIFY FIRST
#### ALTER TABLE MODIFY AFTER
#### ALTER TABLE MODIFY CONSTRAINT
```sql
ALTER TABLE users
	MODIFY user_id VARCHAR(16) CONSTRAINT
### ALTER TABLE CHANGE
- Change column name
```sql
ALTER TABLE users
	CHANGE userid user_id VARCHAR(16) NOT NULL;
```
### ALTER TABLE ALTER COLUMN
```sql
ALTER TABLE users
	ALTER COLUMN user_id VARCHAR(16) NOT NULL;
```
### ALTER TABLE DROP COLUMN
```sql
ALTER TABLE users
	DROP COLUMN user_age;
```
### ALTER TABLE ADD CONSTRAINT
#### ALTER TABLE ADD CONSTRAINT PRIMARY KEY
#### ALTER TABLE ADD CONSTRAINT FOREIGN KEY REFERENCES
### ALTER TABLE DROP CONSTRAINT
### ALTER TABLE DROP FOREIGN KEY



# INFORMATION_SCHEMA
## INFORMATION_SCHEMA.TABLES
```sql
SELECT *
FROM INFORMATION_SCHEMA.TABLES
WHERE TABLE_NAME = 'users';
```
## INFORMATION_SCHEMA.COLUMNS
```sql
SELECT COLUMN_NAME
FROM INFORMATION_SCHEMA.COLUMNS
WHERE TABLE_NAME = 'users';
```
## INFORMATION_SCHEMA.TABLE_CONSTRAINTS
```sql
SELCT *
FROM INFORMATION_SCHEMA.TALBE_CONTSTRAINTS
WHERE TABLE_NAME = `users`;
```



## INSERT()
```sql
SELECT INSERT("abcdefghi", 3, 2, "####");
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


# Logical Functions
## COALESCE()
```sql
COALESCE(mobile, '07986 444 2266')
```
## IS NULL, IS NOT NULL
## CASE WHEN THEN ELSE END
- Search for conditions sequentially.
```sql
SELECT CASE WHEN (a >= b + c OR b >= c + a OR c >= a + b) THEN "Not A Triangle" WHEN (a = b AND b = c) THEN "Equilateral" WHEN (a = b OR b = c OR c = a) THEN "Isosceles" ELSE "Scalene" END
FROM triangles;
```
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
## IN
```sql
SELECT name, addr
FROM usertbl
WHERE addr IN ("서울", "경기", "충청");
```



# Numeric Functions
## CEILING(), FLOOR()
## MIN(), MAX(), AVG(), SUM()
## FORMAT()
```sql
SELECT FORMAT(123456.123456, 4);
```
- 출력할 소수점 이하 자릿수 지정. 
# BETWEEN AND
```sql
SELECT name, height
FROM sqlDB.usertbl
WHERE height BETWEEN 180 AND 183;
```



# String Functions
## REPLACE()
```sql
SELECT CAST(CEILING(AVG(CAST(salary AS FLOAT)) - AVG(CAST(REPLACE(CAST(salary AS FLOAT), "0", "") AS FLOAT))) AS INT)
FROM employees;
```
## LEFT(), RIGHT(), SUBSTRING()
```sql
SELECT DISTINCT city
FROM station
WHERE RIGHT(city, 1) IN ("a", "e", "i", "o", "u");
```
## LOWER(), UPPER(), INITCAP()
## LPAD(), RPAD()
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
## CONCAT()
```sql
SELECT
CONCAT(CASE hdc_mbr.mbr_mst.mbr_sex
WHEN "0" THEN "남자"
WHEN "`" THEN "여자"
END, "/", hdc_mbr.mbr_mst.mbr_birth)
```
## CONCAT_WS()
```sql
SELECT CONCAT_WS("/", "2020", "01", "12");
```
## LIKE
```sql
SELECT DISTINCT city
FROM station
WHERE city LIKE "%a" OR city LIke "%e" OR city LIKE "%i" OR city LIKE "%o" OR city LIKE "%u";
```
- `%` represents zero, one, or multiple characters.
- `_` represents one, single character.



# Date Functions
## HOUR()
```sql
SELECT HOUR(datetime) AS HOUR, COUNT(*)
FROM animal_outs
GROUP BY HOUR
HAVING HOUR BETWEEN 9 AND 19
ORDER BY HOUR;
```
- HOUR()는 일반조건이므로 (COUNT()와 달리) HAIVNG과 함께 쓸 수 없다.
```
## DATEADD()
## DATEDIFF()
```sql
DATEDIFF(DAY, A.start_date, MIN(B.end_date))
```



# Subquery
```sql
SELECT hacker_id, name
FROM (
    SELECT su.hacker_id, ha.name,
        CASE WHEN su.score = di.score THEN 1 ELSE 0 END AS is_full_score
    FROM submissions AS su LEFT OUTER JOIN challenges AS ch ON su.challenge_id = ch.challenge_id
        LEFT OUTER JOIN difficulty AS di ON ch.difficulty_level = di.difficulty_level
        LEFT OUTER JOIN hackers AS ha ON su.hacker_id = ha.hacker_id) AS A
GROUP BY hacker_id, name
HAVING SUM(is_full_score) > 1
ORDER BY SUM(is_full_score) DESC, hacker_id ASC;
```
- Subquery로 생성한 Table은 항상 `AS`로 이름을 지정해줘야 합니다.