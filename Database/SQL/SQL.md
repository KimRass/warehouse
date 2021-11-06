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
	- `ORDER BY`가 정렬을 하는 시점은 모든 실행이 끝난 후 데이터를 출력하기 바로 직전이다.

# Error Messages
## Column <<COLUMN1>> is invalid in the ORDER BY clause because it is not contained in either an aggregate function or the GROUP BY clause.

# Operator Precedence Rules
- Parentheses -> Arithmetic Operators(`*`, `/` -> `+`, `-`) -> Concatenation Operator(`||`) -> Comparison Operators(`=`, `!=`, `<`, `<=`, `>`, `>=`) -> `IS`(`IS NULL`, `IS NOT NULL`, `IS EMPTY`, `IS NOT EMPTY`) -> (`BETWEEN`, `LIKE`, `IN()`( -> Logical Operatiors(`NOT` -> `AND` -> `OR`)

# DDL(Data Definition Language)
- Oracle database implicitly commits the current transaction before and after every DDL statement.
- `CREATE`, `ALTER`, `DROP`, `TRUNCATE`,  `RENAME`
## `CREATE`
### `CREATE TABLE`
#### `CREATE TABLE (NOT NULL, DEFAULT, UNIQUE, CHECK, PRIMARY KEY, FOREIGN KEY REFERENCES)`
- `NOT NULL`: Ensures that a column cannot have NULL value.
- `DEFAULT`: Provides a default value for a column when none is specified.
- `UNIQUE`: Ensures that all values in a column are different.
- `CHECK`: Ensures that all the values in a column satisfies certain conditions.
- `PRIMARY KEY`: Uniquely identifies each record in a database table.
- `FOREIGN KEY REFERENCES`: Uniquely identifies a record in any of the given database table.
```sql
CREATE TABLE orders(
	order_id INT NOT NULL PRIMARY KEY,
	order_no INT NOT NULL,
	person_id INT FOREIGN KEY REFERENCES persons(person_id));
```
- For defining a primary key on multiple columns, the statement should be like;
	```sql
	CREATE TABLE orders(
		order_id INT NOT NULL,
		order_no INT NOT NULL,
		person_id INT,
		PRIMARY KEY(order_id),
		FOREIGN KEY(person_id) REFERENCES persons(person_id));
	```
##### `CREATE TABLE (FOREIGN KEY REFERENCES ON DELETE CASCADE)`
- Source: http://www.dba-oracle.com/t_foreign_key_on_delete_cascade.htm
- When you create a foreign key constraint, Oracle default to `ON DELETE RESTRICT` to ensure that a parent rows cannot be deleted while a child row still exists. However, you can also implement `ON DELETE CASCADE` to delete all child rows when a parent row is deleted.
- Using `ON DELETE CASCADE` and `ON DELETE RESTRICT` is used when a strict one-to-many relationship exists such that any "orphan" row violates the integrity of the data.
### `CREATE VIEW AS SELECT FROM`
### `CREATE ROLE`
- Source: https://www.programmerinterview.com/database-sql/database-roles/
- A database role is a collection of any number of permissions/privileges that can be assigned to one or more users. A database role also is also given a name for that collection of privileges.
- The majority of today’s RDBMS’s come with predefined roles that can be assigned to any user. But, a database user can also create his/her own role if he or she has the CREATE ROLE privilege.
### `CREATE INDEX ON`
```sql
CREATE INDEX <<Index_name>>
ON <<Table>>(<<Column1>>, <<Column2>>, ...);
```
## `ALTER`
### `ALTER TABLE`
#### `ALTER TABLE ADD`
```sql
ALTER TABLE users
	ADD birth_date CHAR(6) NULL;
```
##### `ALTER TABLE ADD FIRST`
##### `ALTER TABLE ADD AFTER`
##### `ALTER TABLE ADD PRIMARY KEY`
##### ALTER TABLE ADD()
#### `ALTER TABLE ALTER COLUMN` (MS SQL Server), `ALTER TABLE MODIFY` (Oracle), `ALTER TABLE MODIFY COLUMN` (MySQL)
- Change the data type of a column in a table.
```sql
ALTER TABLE users
ALTER COLUMN user_id VARCHAR(16) NOT NULL;
```
```sql
ALTER TABLE ex_table
MODIFY COLUMN sFifth VARCHAR(55);
```
##### `ALTER TABLE MODIFY CONSTRAINT`
```sql
ALTER TABLE users
MODIFY user_id VARCHAR(16) CONSTRAINT
#### `ALTER TABLE CHANGE`, `ALTER TABLE CHANGE COLUMN`
- Change column name and the data type.
```sql
ALTER TABLE users
	CHANGE userid user_id VARCHAR(16) NOT NULL;
```
```sql
ALTER TABLE ex_table CHANGE COLUMN nSecond sSecond VARCHAR(22);
```
#### `ALTER TABLE DROP COLUMN`
```sql
ALTER TABLE users
	DROP COLUMN user_age;
```
#### `ALTER TABLE ADD COLUMN`
```sql
ALTER TABLE Customers
ADD Email VARCHAR(255);
```
#### `ALTER TABLE ADD CONSTRAINT`
##### `ALTER TABLE ADD CONSTRAINT PRIMARY KEY`
##### `ALTER TABLE ADD CONSTRAINT FOREIGN KEY REFERENCES`
#### `ALTER TABLE DROP CONSTRAINT`
#### `ALTER TABLE DROP FOREIGN KEY`
## `DROP`
### `DROP TABLE`
- With the help of `DROP` command we can drop (delete) the whole structure in one go i.e, it removes the named elements of the schema. By using this command the existence of the whole table is finished or say lost.
- Here we can’t restore the table by using the `ROLLBACK` command.
#### `DROP TABLE IF EXISTS`
### `DROP VIEW`
## `TRUNCATE`
### `TRUNCATE TABLE`
- By using this command the existence of all the rows of the table is lost. It is comparatively faster than `DELETE` command as it deletes all the rows fastly.
- Here we can’t restore the tuples of the table by using the `ROLLBACK` command.
## `RENAME`
### `RENAME TABLE TO`
```sql
RENAME TABLE old_table1 TO new_table1, old_table2 TO new_table2, old_table3 TO new_table3;
```

# DCL(Data Control Language)
## `GRANT ON TO`, `REVOKE ON TO`
```sql
GRANT <<Privileges>>
ON <<Table>>
TO <<User>>
```
- <<Privileges>>: `CONNECT`, `RESOURCE`, `DBA`, ...
- `CONNECT`: DB 접속 권한.
- `RESOURCE`: 테이블 등 생성 권한.
```sql
## `WITH GRANT OPTION`

# TCL(Transaction Control Language)
## `COMMIT`
- Source: https://www.geeksforgeeks.org/difference-between-commit-and-rollback-in-sql/
- `COMMIT` is used to permanently save the changes done in the transaction in tables/databases. The database cannot regain its previous state after the execution of it.
## `SAVEPOINT`
- Use the `SAVEPOINT` statement to identify a point in a transaction to which you can later roll back.
```sql
SAVEPOINT <<Savepoint Name>>
```
## `ROLLBACK`
- `ROLLBACK` is used to undo the transactions that have not been saved in database. The command is only be used to undo changes since the last `COMMIT`.
### `ROLLBACK TO`
```sql
ROLLBACK TO <<Savepoint Name>>
```

# `INFORMATION_SCHEMA`
## `INFORMATION_SCHEMA.TABLES`
```sql
SELECT *
FROM INFORMATION_SCHEMA.TABLES
WHERE TABLE_NAME = 'users';
```
## `INFORMATION_SCHEMA.COLUMNS`
```sql
SELECT COLUMN_NAME
FROM INFORMATION_SCHEMA.COLUMNS
WHERE TABLE_NAME = 'users';
```
## `INFORMATION_SCHEMA.TABLE_CONSTRAINTS`
```sql
SELCT *
FROM INFORMATION_SCHEMA.TALBE_CONTSTRAINTS
WHERE TABLE_NAME = `users`;
```

# `DUAL` (Oracle)
- Oracle provides you with the DUAL table which is a special table that belongs to the schema of the user SYS but it is accessible to all users.
- The `DUAL` table has one column named `DUMMY` whose data type is `VARCHAR2()` and contains one row with a value `X`.
- By using the `DUAL` table, you can execute queries that contain functions that do not involve any table

# DML(Data Manipulation Language)
- `INSERT`, `UPDATE`, `DELETE`, `SELECT`
## `INSERT`
### `INSERT INTO VALUES()`
```sql
INSERT INTO customers 
(customername, address, city, postalcode, country)
VALUES("Hekkan Burger", "Gateveien 15", "Sandnes", "4306", "Norway");
```
- If you are adding values for all the columns of the table, you do not need to specify the column names in the SQL query.
### `SELECT INSERT()`
```sql
SELECT INSERT("abcdefghi", 3, 2, "####");
```
## `UPDATE`
### `UPDATE SET FROM WHERE`
```
UPDATE emp
SET ename = 'jojo'
WHERE emp_no = 100;
```
## `DELETE`
### `DELETE FROM WHERE`
- Here we can use the `ROLLBACK` command to restore the tuple.
```sql
DELETE FROM emp
WHERE emp_no = 100;
```
## `SELECT TOP FROM` (MS SQL Server), `SELECT FROM LIMIT` (MySQL), `SELECT FROM FETCH FIRST ROWS ONLY` (Oracle)
```sql
SELECT TOP 1 months*salary, COUNT(employee_id)
FROM employee
GROUP BY months*salary
ORDER BY months*salary DESC;
```
```sql
SELECT *
FROM Customers
FETCH FIRST 3 ROWS ONLY;
```
### `SELECT TOP PERCENT FROM` (MS SQL Server), `SELECT FROM FETCH FIRST PERCENT ROWS ONLY` (Oracle)
## `SELECT ROWNUM FROM`
```sql
SELECT ROWNUM, pr.product_name, pr.standard_cost
FROM products AS pr
```
- `ROWNUM`과 `ORDER BY`를 같이 사용할 경우 매겨놓은 순번이 섞여버리는 현상이 발생합니다.
- 이 때, Inline view에서 먼저 정렬을 하고 순번을 매기는 방법으로 정렬된 데이터에 순번을 매길 수 있습니다.
## `ROW_NUMBER()`
- Inline view를 사용하지 않고도 어떤 값의 순서대로 순번을 매길 수 있습니다. 
## `SELECT FROM WHERE EXISTS()`
- Source: https://www.w3schools.com/sql/sql_exists.asp
- The `EXISTS` operator is used to test for the existence of any record in a subquery.
- The `EXISTS` operator returns `TRUE` if the subquery returns one or more records.
```sql
SELECT *
FROM customers
WHERE EXISTS(
	SELECT *
	FROM orders
	WHERE orders.c_id = customers.c_id);
```
- 주문한 적이 있는 고객 정보를 조회하는 query.
```sql
SELECT SupplierName
FROM Suppliers
WHERE EXISTS (
	SELECT ProductName
	FROM Products
	WHERE Products.SupplierID = Suppliers.supplierID AND Price < 20);
```
- The above SQL statement returns `TRUE` and lists the suppliers with a product price less than 20.

## UNION, UNION ALL
- `UNION` selects only distinct values by default. To allow duplicate values, use `UNION ALL`
## `MINUS` (Oracle), `EXCEPT` (MS SQL Server)
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

# Logical Functions
## `COALESCE()` (MySQL)
```sql
SELECT COALESCE(col1, col2*10, 100)
FROM table
```
## `ISNULL()` (MS SQL Server)
## `NVL()`/`NVL2() (Oracle)
```sql
NVL(<<Column>>, <<Value if NULL>>)
NVL2(<<Column>>, <<Value if not NULL>>, <<Value if NULL>>)
```
## `IFNULL()` (MySQL)
```sql
SELECT animal_type, IFNULL(name, "No name"), sex_upon_intake
FROM animal_ins
ORDER BY animal_id
```
## `NULLIF()` (Oracle, MS SQL Server)
- Return `NULL` if two expressions are equal, otherwise return the first expression.
## `IS NULL`, `IS NOT NULL`
- Source: https://www.w3schools.com/sql/sql_null_values.asp
- It is not possible to test for NULL values with comparison operators, such as `=`, `<`, ....
## `CASE WHEN THEN ELSE END`
- Search for conditions sequentially.
```sql
SELECT CASE WHEN (a >= b + c OR b >= c + a OR c >= a + b) THEN "Not A Triangle" WHEN (a = b AND b = c) THEN "Equilateral" WHEN (a = b OR b = c OR c = a) THEN "Isosceles" ELSE "Scalene" END
FROM triangles;
```
## `ANY()`
- Return `TRUE` if any of the subquery values meet the condition.
```sql
SELECT name
FROM usertbl
WHERE height > ANY(
	SELECT height
	FROM usertbl
	WHERE addr = '서울');
```
## `ALL()`
- Return `TRUE` if all of the subquery values meet the condition.
- Used with `SELECT`, `WHERE` and `HAVING` statements.
## `IN()`
- `NULL`은 `IN()` 연산자 안에 있어도 아무런 의미를 갖지 않습니다.
```sql
SELECT name, addr
FROM usertbl
WHERE addr IN("서울", "경기", "충청");
```

# Numeric Functions
## CEILING(), FLOOR()
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
## `REPLACE()`
```sql
SELECT CAST(CEILING(AVG(CAST(salary AS FLOAT)) - AVG(CAST(REPLACE(CAST(salary AS FLOAT), "0", "") AS FLOAT))) AS INT)
FROM employees;
```
## `LEFT()`, `RIGHT()`
```sql
SELECT DISTINCT city
FROM station
WHERE RIGHT(city, 1) IN ("a", "e", "i", "o", "u");
```
## `SUBSTRING()`
## LOWER(), UPPER(), INITCAP()
## LPAD(), RPAD()
```sql
SELECT LPAD("이것이", 5, "##")
```
## `LTRIM()`, `RTRIM()`
## `TRIM()`
### `SELECT TRIM(BOTH) FROM`, `SELECT TRIM(LEADING) FROM`, `SELECT TRIM(TRAILING) FROM`
```sql
SELECT TRIM("   좌우측공백삭제   ";
SELECT TRIM(BOTH "ㅋ" FROM "ㅋㅋㅋ좌우측문자삭제ㅋㅋㅋ");
SELECT TRIM(LEADING "ㅋ" FROM "ㅋㅋㅋ좌측문자삭제ㅋㅋㅋ");
SELECT TRIM(TRAILING "ㅋ" FROM "ㅋㅋㅋ우측문자삭제ㅋㅋㅋ");
```
## `CONCAT()`
```sql
SELECT CONCAT(CASE hdc_mbr.mbr_mst.mbr_sex WHEN '0' THEN '남자' WHEN '`' THEN '여자' END, '/', hdc_mbr.mbr_mst.mbr_birth)
```
## CONCAT_WS()
```sql
SELECT CONCAT_WS("/", "2020", "01", "12");
```
## `SELECT FROM WHERE LIKE`
```sql
SELECT DISTINCT city
FROM station
WHERE city LIKE "%a" OR city LIke "%e" OR city LIKE "%i" OR city LIKE "%o" OR city LIKE "%u";
```
- `%` represents zero, one, or multiple characters.
- `_` represents one, single character.
## `SELECT TO_CHAR() FROM`
- Used to convert a number or date to a string.
```sql
TO_CHAR(mbrmst.mbr_leaving_expected_date, "YYYY/MM/DD") AS mbr_leaving_expected_dt
```

# Date Functions
## `HOUR()`
```sql
SELECT HOUR(datetime) AS HOUR, COUNT(*)
FROM animal_outs
GROUP BY HOUR
HAVING HOUR BETWEEN 9 AND 19
ORDER BY HOUR;
```
- HOUR()는 일반조건이므로 (COUNT()와 달리) HAIVNG과 함께 쓸 수 없다.
```
## `SYSDATE` (Oracle), `SYSDATE()` (MySQL)
## `EXTRACT('YEAR' FROM)`, `EXTRACT('MONTH' FROM)`, `EXTRACT('DAY' FROM)`
## `DATEADD()`
## `DATEDIFF(YEAR)`, `DATEDIFF(MONTH)`, `DATEDIFF(DAY)`
```sql
DATEDIFF(DAY, A.start_date, MIN(B.end_date))
```

# Window Functions
- `SUM()`, `MAX()`, `MIN()`, `RANK()`
## `OVER(PARTITION BY ORDER BY ROWS)`, `OVER(PARTITION BY ORDER BY RANGE)`
- `ROWS` is used to specify which rows to include in the aggregation.
- `RANGE` can be used to let Oracle determine which rows lie within the range.
- `UNBOUNDED PRECEDING`: Window의 첫 위치가 첫 번째 행.
- `UNBOUNDED FOLLOWING`: Window의 마지막 위치가 마지막 행.
- `CURRENT ROW`: Window의 첫 위치가 현재 행.
```sql
SELECT month, SUM(tot_sales) AS monthly_sales,
	AVG(SUM(tot_sales)) OVER(ORDER BY month RANGE BETWEEN 1 PRECEDING AND 1 FOLLOWING) AS rolling_avg
FROM orders
```

# Rank Functions
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

# Aggregate Functions
## `MIN()`, `MAX()`, `AVG()`, `SUM()`, `STDDEV()`, `VARIAN()`
## `COUNT()`
- `COUNT(*)`, `COUNT(1)`, ...: Return the number of rows in the table.
- `COUNT(<<Column>>)`: Return the number of non-NULL values in the column.
- `COUNT(DISTINCT <<Column>>)`: Return the number of distinct non-NULL values in the column.

# Group Functions
## `SELECT FROM GROUP BY`
- `GROUP BY` treats `NULL` as valid values.
- All `NULL` values are grouped into one value or bucket.
### `SELECT FROM GROUP BY ROLLUP()`
```sql
SELECT DEPARTMENT_ID, JOB_ID, SUM(SALARY)
FROM EMPLOYEES
WHERE DEPARTMENT_ID > 80
GROUP BY ROLLUP(DEPARTMENT_ID, JOB_ID)
ORDER BY DEPARTMENT_ID;
```
### `SELECT FROM GROUP BY CUBE()`
- Generate subtotals for all combinations of the dimensions specified.
```sql
SELECT dname, job, SUM(sal)
FROM test18
GROUP BY CUBE(dname, job);
```
### `SELECT FROM GROUP BY GROUPING SETS()`
- `GROUP BY GROUPING SETS(<<Column>>)`
	- `GROUP BY <<Column>>`과 동일.
- `GROUP BY GROUPING SETS((<<Column1>>, <<Column2>>, ...))`
	- 인자로 주어진 컬럼들의 조합별로 값 집계.
- `GROUP BY GROUPING SETS(())`
	- 전체에 대한 집계.
- 인자가 여러 개인 경우 위 3가지 경우의 각 결과를 `UNION ALL`한 것과 같음.
```sql
SELECT DEPARTMENT_ID, JOB_ID, SUM(SALARY)
FROM EMPLOYEES
WHERE DEPARTMENT_ID > 80
GROUP BY GROUPING SETS((DEPARTMENT_ID, JOB_ID), ());
```

# Subquery
- Source: https://www.geeksforgeeks.org/sql-subquery/
- A subquery is a query within another query. The outer query is called as main query and inner query is called as subquery.
- The subquery generally executes first, and its output is used to complete the query condition for the main or outer query.
- Subquery must be enclosed in parentheses.
- Subqueries are on the right side of the comparison operator.
- `ORDER BY` cannot be used in a Subquery. `GROUP BY` can be used to perform same function as `ORDER BY`.
- Use single-row operators with single-row subqueries. Use multiple-row operators with multiple-row subqueries.
	- Source: https://www.tutorialspoint.com/What-are-single-row-and-multiple-row-subqueries
	- Single-Row Subquery: A single-row subquery is used when the outer query's results are based on a single, unknown value. Although this query type is formally called "single-row," the name implies that the query returns multiple columns-but only one row of results. However, a single-row subquery can return only one row of results consisting of only one column to the outer query.
	- `=`, `!=`, `<`, `<=`, `>`, `>=`
	```sql
	SELECT first_name, salary, department_id
	FROM employees
	WHERE salary = (SELECT MIN(salary)
		FROM employees);
	```
	- Multiple-Row Subquery: Multiple-row subqueries are nested queries that can return more than one row of results to the parent query
	- `IN()`, `ALL()`, `ANY()`, `EXISTS()`
- A subquery can be placed in `WHERE` clause(Subquery), `HAVING` clause, `FROM` clause(Inline View), `SELECT` clause(Scala Subquery).
## Inline View
- Subquery in the `FROM` clause of a `SELECT` statement.
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
## Scala Subquery
- 반드시 1개의 행과 열만 반환.
## Correlated Subquery(= 상관 서브 쿼리)
- Source: https://myjamong.tistory.com/176
- 내부 Subquery에서 외부테이블의 값을 참조할 때 사용됩니다.
- Subquery와는 다르게  Inner Query 부터 Outer Query 순서대로 실행되는 것이 아니라 Outer Query에서 읽어온 행을 갖고 Inner쿼리를 실행하는 것을 반복하여 결과를 출력해줍니다.
- Outer Query와 Inner Query에서 같은 테이블을 참조한다면 Outer Query의 테이블에 Alias를 사용해서 구분해줍니다.