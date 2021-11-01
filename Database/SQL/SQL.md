# Data Types
## String
- `CHAR(size)`: A fixed length string.
	- Blank-Padded Comparison Semantics: With blank-padded semantics, if the two values have different lengths, then Oracle first adds blanks to the end of the shorter one so their lengths are equal. Oracle then compares the values character by character up to the first character that differs. The value with the greater character in the first differing position is considered greater. If two values have no differing characters, then they are considered equal. This rule means that two values are equal if they differ only in the number of trailing blanks.
- `VARCHAR(size)`: A variable length string.
	- Nonpadded Comparison Semantics: With nonpadded semantics, Oracle compares two values character by character up to the first character that differs. The value with the greater character in that position is considered greater. If two values of different length are identical up to the end of the shorter one, then the longer value is considered greater. If two values of equal length have no differing characters, then the values are considered equal.
## Numeric
- `BOOL`: Zero is considered as false, nonzero values are considered as true.
- `INT`
- `FLOAT(n)`
## Data and Time
- A later date is considered greater than an earlier one.
- `DATE`: A date. Format: `YYYY-MM-DD`.
- `TIME`: A time. Format: `hh:mm:ss`.
- `DATETIME`: A date and time combination. Format: `YYYY-MM-DD hh:mm:ss`.



# Execution Order
- `FROM` -> `WHERE` -> `GROUP BY` -> `HAVING` -> `SELECT` -> `ORDER BY` -> `LIMIT` or `TOP`



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
### CREATE INDEX ON
```sql
CREATE INDEX idx ON persons(person_id);
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
## `DROP TABLE`
## DROP TABLE IF EXISTS
## `ALTER TABLE`
### ALTER TABLE ADD
```sql
ALTER TABLE users
	ADD birth_date CHAR(6) NULL;
```
#### ALTER TABLE ADD FIRST
#### ALTER TABLE ADD AFTER
#### ALTER TABLE ADD PRIMARY KEY
##### ALTER TABLE ADD ()
### `ALTER TABLE ALTER COLUMN` (for MS SQL Server), `ALTER TABLE MODIFY` (for Oracle), `ALTER TABLE MODIFY COLUMN` (for MySQL)
- Change the data type of a column in a table.
```sql
ALTER TABLE users
ALTER COLUMN user_id VARCHAR(16) NOT NULL;
```
```sql
ALTER TABLE ex_table
MODIFY COLUMN sFifth VARCHAR(55);
```
#### `ALTER TABLE MODIFY FIRST`
#### `ALTER TABLE MODIFY AFTER`
### `ALTER TABLE MODIFY CONSTRAINT`
```sql
ALTER TABLE users
MODIFY user_id VARCHAR(16) CONSTRAINT
### `ALTER TABLE CHANGE`
- Change column name
```sql
ALTER TABLE users
	CHANGE userid user_id VARCHAR(16) NOT NULL;
```
### `ALTER TABLE CHANGE COLUMN`
```sql
ALTER TABLE ex_table CHANGE COLUMN nSecond sSecond VARCHAR(22);
```
### ALTER TABLE DROP COLUMN
```sql
ALTER TABLE users
	DROP COLUMN user_age;
```
### `ALTER TABLE ADD COLUMN`
```sql
ALTER TABLE Customers
ADD Email VARCHAR(255);
```
### `ALTER TABLE ADD CONSTRAINT`
#### ALTER TABLE ADD CONSTRAINT PRIMARY KEY
#### ALTER TABLE ADD CONSTRAINT FOREIGN KEY REFERENCES
### ALTER TABLE DROP CONSTRAINT
### ALTER TABLE DROP FOREIGN KEY
## `CREATE VIEW AS SELECT FROM`
## `DROP VIEW`



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
## `INSERT INTO () VALUES()`
```sql
INSERT INTO customers 
(customername, address, city, postalcode, country)
VALUES ("Hekkan Burger", "Gateveien 15", "Sandnes", "4306", "Norway");
```
- If you are adding values for all the columns of the table, you do not need to specify the column names in the SQL query.
## UNION, UNION ALL
- `UNION` selects only distinct values by default. To allow duplicate values, use `UNION ALL`
## `MINUS` (for Oracle), `EXCEPT` (for MS SQL Server)
- Return all rows in the first `SELECT` statement that are not returned by the second `SELECT` statement.
## `START WITH CONNECT BY PRIOR ORDER SIBLINGS BY`
- `START WITH` specifies the root row(s) of the hierarchy. `START WITH` 다음에 조건을 명시할 수도 있다.
- `CONNECT BY` specifies the relationship between parent rows and child rows of the hierarchy. It always joins a table to itself, not to another table.
- `PRIOR` should occur exactly once in each `CONNECT BY` expression. `PRIOR` can occur on either the left-hand side or the right-hand side of the expression, but not on both.
- If you want to order rows of siblings of the same parent, then use the `ORDER SIBLINGS BY` clause.
```sql
SELECT last_name, employee_id, manager_id, LEVEL
FROM employees
START WITH employee_id = 100
CONNECT BY PRIOR employee_id = manager_id
ORDER SIBLINGS BY last_name;
```
- The `NOCYCLE` parameter in the `CONNECT BY` condition causes Oracle to return the rows in spite of the loop. The `CONNECT_BY_ISCYCLE` pseudocolumn shows you which rows contain the cycle.
```sql
SELECT last_name "Employee", CONNECT_BY_ISCYCLE "Cycle",
LEVEL, SYS_CONNECT_BY_PATH(last_name, '/') "Path"
FROM employees
WHERE level <= 3 AND department_id = 80
START WITH last_name = 'King'
CONNECT BY NOCYCLE PRIOR employee_id = manager_id AND LEVEL <= 4;
```
## CAST()
```sql
SELECT "There are a total of " + CAST(COUNT(occupation) AS CHAR) + " " + LOWER(occupation) + "s."
FROM occupations
GROUP BY occupation
ORDER BY COUNT(occupation), LOWER(occupation);
```
## PIVOT, UNPIVOT
```sql
SELECT 반정보, 과목, 점수
FROM dbo.성적	UNPIVOT (점수	FOR 과목 IN(국어, 수학, 영어)) AS UNPVT
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
```sql
SELECT COALESCE(col1, col2*10, 100)
FROM table
```
## `IS NULL`, `IS NOT NULL`
- Source: https://en.wikipedia.org/wiki/Null_(SQL)
- A null should not be confused with a value of 0. A null value indicates a lack of a value, which is not the same thing as a value of zero. For example, consider the question "How many books does Adam own?" The answer may be "zero" (we know that he owns none) or "null" (we do not know how many he owns). In a database table, the column reporting this answer would start out with no value (marked by Null), and it would not be updated with the value "zero" until we have ascertained that Adam owns no books.
- SQL null is a state, not a value.
## `CASE WHEN THEN ELSE END`
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
## `IN()`
```sql
SELECT name, addr
FROM usertbl
WHERE addr IN("서울", "경기", "충청");
```
## `GROUP BY`
### `GROUP BY ROLLUP()`
```sql
SELECT DEPARTMENT_ID, JOB_ID, SUM(SALARY)
FROM EMPLOYEES
WHERE DEPARTMENT_ID > 80
GROUP BY ROLLUP(DEPARTMENT_ID, JOB_ID)
ORDER BY DEPARTMENT_ID;
```
### `GROUP BY CUBE()`
### `GROUP BY GROUPING SETS()`
```sql
SELECT DEPARTMENT_ID, JOB_ID, SUM(SALARY)
FROM EMPLOYEES
WHERE DEPARTMENT_ID > 80
GROUP BY GROUPING SETS((DEPARTMENT_ID, JOB_ID), ());
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
## `RANK() OVER(ORDER BY)`
- 중복 값들에 대해서 동일 순위로 표시하고, 중복 순위 다음 값에 대해서는 중복 개수만큼 떨어진 순위로 출력.
```sql
SELECT emp_no, emp_nm, sal,
	RANK() OVER(ORDER BY salary DESC) ranking
FROM employee;
```
## `DENSE_RANK() OVER(ORDER BY)`
- 중복 값들에 대해서 동일 순위로 표시하고, 중복 순위 다음 값에 대해서는 중복 값 개수와 상관없이 순차적인 순위 값을 출력.
## `ROW_NUMBER() OVER(ORDER BY)`
- 중복 값들에 대해서도 순차적인 순위를 표시하도록 출력.
## `NTILE() OVER ()`
- 뒤에 함께 적어주는 숫자 만큼 등분.



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
- Correlated Subquery
	- Source: https://myjamong.tistory.com/176
	- 내부 Subquery에서 외부테이블의 값을 참조할 때 사용됩니다.
	- Subquery와는 다르게  Inner Query 부터 Outer Query 순서대로 실행되는 것이 아니라 Outer Query에서 읽어온 행을 갖고 Inner쿼리를 실행하는 것을 반복하여 결과를 출력해줍니다.
	- Outer Query와 Inner Query에서 같은 테이블을 참조한다면 Outer Query의 테이블에 Alias를 사용해서 구분해줍니다.