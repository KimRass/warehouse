# Case Sensitivity
- Source: https://seeq.atlassian.net/wiki/spaces/KB/pages/443088907/SQL+Column+Names+and+Case+Sensitivity
## Oracle
- Oracle stores unquoted column names in uppercase.
## MS SQL Server
- The returned column name has the case that was used in the `SELECT` statement.

# Table or Column Names
## Oracle
- Be no longer than 30 characters.
- Begin with an alphabetical character.
- Contain only alphabetical characters, numbers, or one of the following special characters: `#`, `$`, `_`

# Data Types
## String
- `CHAR(n)`: A fixed length string.
	- Blank-Padded Comparison Semantics: ***If the two values have different lengths, then Oracle first adds blanks to the end of the shorter one so their lengths are equal.*** Oracle then compares the values character by character up to the first character that differs. The value with the greater character in the first differing position is considered greater. If two values have no differing characters, then they are considered equal. This rule means that two values are equal if they differ only in the number of trailing blanks.
- `VARCHAR(n)`: A variable length string.
	- Nonpadded Comparison Semantics: Oracle compares two values character by character up to the first character that differs. The value with the greater character in that position is considered greater. If two values of different length are identical up to the end of the shorter one, then the longer value is considered greater. ***If two values of equal length have no differing characters, then the values are considered equal.***
- `VARCHAR2(n)`
	- There is no difference between `VAHRCHAR` and `VARCHAR2` in Oracle. However, it is advised not to use `VARCHAR` for storing data as it is reserved for future use for storing some other type of variable. Hence, ***always use `VARCHAR2` in place of `VARCHAR`.***
## Numeric
- `BOOL`: Zero is considered as false, nonzero values are considered as true.
- `INT`
- `FLOAT(n)`
## Date and Time
- A later date is considered greater than an earlier one.
- `DATE`: A date. Format: `YYYY-MM-DD`.
- `TIME`: A time. Format: `hh:mm:ss`.
- `DATETIME`: A date and time combination. Format: `YYYY-MM-DD hh:mm:ss`.

# Execution Order
- `FROM` -> `WHERE` -> `GROUP BY` -> `HAVING` -> `SELECT` -> `ORDER BY` -> `LIMIT`, `TOP` or `FETCH`

# Error Messages
## `Column <<Column>> is invalid in the ORDER BY clause because it is not contained in either an aggregate function or the GROUP BY clause.`

# Operators
## Arithmetic Operators(`*`, `/` -> `+`, `-`)
- ***Any calculation performed on the NULL value returns NULL.***
## Concatenation Operator(`||`) (Oracle, PostgreSQL), `+` (MS SQL Server), `CONCAT()` (All)
- The result of concatenating two character strings is another character string.
## `BETWEEN AND`
- The `BETWEEN AND` operator selects values within a given range. The values can be numbers, text, or dates.
- The `BETWEEN AND` operator is inclusive: begin and end values are included.
- 앞에 오는 숫자가 뒤에 오는 숫자보다 작아야 합니다.
## Operator Precedence Rules
- ***Parentheses -> Arithmetic Operators(`*`, `/` -> `+`, `-`) -> Concatenation Operator(`||`) -> Comparison Operators(`=`, `!=`, `<`, `<=`, `>`, `>=`) -> `IS`(`IS NULL`, `IS NOT NULL`, `IS EMPTY`, `IS NOT EMPTY`), `LIKE`, `IN()` -> `BETWEEN` -> Logical Operatiors(`NOT` -> `AND` -> `OR`)***

# `INFORMATION_SCHEMA` (MS SQL Server)
## `INFORMATION_SCHEMA.TABLES`
- Pseudocolumns: `table_catalog`, `table_schema`, `table_name`, `table_type`
```sql
SELECT *
FROM INFORMATION_SCHEMA.TABLES;
```
## `INFORMATION_SCHEMA.COLUMNS`
```sql
SELECT COLUMN_NAME
FROM INFORMATION_SCHEMA.COLUMNS
[WHERE TABLE_NAME = <<Table>>];
```
## `INFORMATION_SCHEMA.TABLE_CONSTRAINTS`
```sql
SELCT *
FROM INFORMATION_SCHEMA.TALBE_CONTSTRAINTS
[WHERE TABLE_NAME = <<Table>>];
```

# `DUAL` (Oracle)
- Oracle provides you with the DUAL table which is a special table that belongs to the schema of the user SYS but it is accessible to all users.
- The `DUAL` table has one column named `DUMMY` whose data type is `VARCHAR2()` and contains one row with a value `X`.
- By using the `DUAL` table, you can execute queries that contain functions that do not involve any table
- MS SQL Server doesn't need `FROM` clause.

# DDL(Data Definition Language)
- ***Oracle database implicitly commits the current transaction before and after every DDL statement.***
- ***MS SQL Server는 Auto commit을 수행하지 않습니다.***
- `CREATE`, `ALTER`, `DROP`, `TRUNCATE`,  `RENAME`
## `CREATE`
### `CREATE TABLE ([NOT NULL | DEFAULT | UNIQUE | CHECK | PRIMARY KEY | FOREIGN KEY REFERENCES])`
- `NOT NULL`: Ensures that a column cannot have NULL value.
- `DEFAULT`: Provides a default value for a column when none is specified.
- `UNIQUE`: Ensures that all values in a column are different.
- `CHECK`: Ensures that all the values in a column satisfies certain conditions.
- `PRIMARY KEY`: Uniquely identifies each record in a database table.
- `FOREIGN KEY REFERENCES`: Uniquely identifies a record in any of the given database table.
```sql
CREATE TABLE <<Table>>(
	<<Column1>> <<Data Type1>> [<<Constraint1>>],
	<<Column2>> <<Data Type2>> [<<Constraint2>>],
	...)
```
- For defining a primary key on multiple columns, the statement should be like;
	```sql
	CREATE TABLE <<Table>>(
		PRIMARY KEY(<<Column1>>, <<Column2>>, ...),
		...)
	```
#### `CREATE TABLE (FOREIGN KEY REFERENCES [ONE DELETE RESTRICT | ON DELETE CASCADE])`
- Source: http://www.dba-oracle.com/t_foreign_key_on_delete_cascade.htm
- ***When you create a foreign key constraint, Oracle default to `ON DELETE RESTRICT` to ensure that a parent rows cannot be deleted while a child row still exists. However, you can also implement `ON DELETE CASCADE` to delete all child rows when a parent row is deleted.***
### `CREATE VIEW AS SELECT FROM`
### `CREATE ROLE`
- Source: https://www.programmerinterview.com/database-sql/database-roles/
- ***A database role is a collection of any number of permissions/privileges that can be assigned to one or more users.*** A database role is also given a name for that collection of privileges.
- The majority of today’s RDBMS’s come with predefined roles that can be assigned to any user. But, a database user can also create his/her own role if he or she has the `CREATE ROLE` privilege.
### `CREATE INDEX ON`
```sql
CREATE INDEX <<Index_name>>
ON <<Table>>(<<Column1>>, <<Column2>>, ...);
```
### `CREATE TRIGGER`
### `CREATE PROCEDURE`
## `ALTER`
### `ALTER TABLE`
#### `ALTER TABLE ADD [FIRST | AFTER]`
```sql
ALTER TABLE <<Table>>
ADD <<Column>> <<Data Type>> [<<Constraint>>];
```
#### `ALTER TABLE MODIFY` (Oracle), `ALTER TABLE ALTER COLUMN` (MS SQL Server), `ALTER TABLE MODIFY COLUMN` (MySQL)
- Change the data type of a column in a table.
```sql
ALTER TABLE <<Table>>
ALTER COLUMN | MODIFY | MODIFY COLUMN <<Column>> <<Data Type>> [<<Constraint>>];
```
```sql
ALTER TABLE <<Table>>
ALTER COLUMN | MODIFY | MODIFY COLUMN(
	<<Column1>> <<Data Type1>> [<<Constraint1>>],
	<<Column2>> <<Data TYpe2>> [<<Constraint2>>],
	...);
```
#### `ALTER TABLE DROP COLUMN` (Oracle)
```sql
ALTER TABLE <<Table>>
DROP COLUMN <<Column>>;
```
#### `ALTER TABLE RENAME TO` (Oracle)
```sql
ALTER TABLE <<Table Before>>
RENAME TO <<Table After>>
```
#### `ALTER TABLE RENAME COLUMN TO` (Oracle)
```sql
ALTER TABLE <<Table>>
RENAME COLUMN <<Column Before>> TO <<Column After>>
```
## `DROP`
### `DROP TABLE`
- With the help of `DROP` command we can drop (delete) the whole structure in one go. The existence of the whole table is finished.
- ***The `DROP TABLE` statement does not result in space being released back to the tablespace for use by other objects.***
#### `DROP TABLE IF EXISTS`
### `DROP VIEW`
## `TRUNCATE`
### `TRUNCATE TABLE`
- It removes all the rows exist in the table.
- ***It returns the freed space to the tablespace.***
- It is comparatively faster than `DELETE` command.
## `RENAME`
### `RENAME TABLE TO`
```sql
RENAME TABLE old_table1 TO new_table1, old_table2 TO new_table2, old_table3 TO new_table3;
```

# DCL(Data Control Language)
## `GRANT ON TO`, `REVOKE ON TO`
```sql
GRANT <<Privilege1>>, <<Privilege2>>, ...
ON <<Table>>
TO <<User>>
```
- `<<Privilege>>`: `CONNECT`, `RESOURCE`, `DBA`, ...
	- `CONNECT`: DB 접속 권한.
	- `RESOURCE`: 테이블 등 생성 권한.
## `WITH GRANT OPTION`

# TCL(Transaction Control Language)
## `COMMIT`
- Source: https://www.geeksforgeeks.org/difference-between-commit-and-rollback-in-sql/
- `COMMIT` is used to permanently save the changes done in the transaction in tables/databases. The database cannot regain its previous state after the execution of it.
- `COMMIT` 되지 않은 데이터는 나 자신은 볼 수 있지만 다른 사용자는 볼 수 없으며 변경할 수도 없습니다.
## `SAVEPOINT`
```sql
SAVEPOINT <<Savepoint Name>>
```
- Use the `SAVEPOINT` statement to identify a point in a transaction to which you can later roll back.
- If you give two savepoints the same name, the earlier savepoint is erased.
## `ROLLBACK`
- `ROLLBACK` is used to undo the transactions that have not been saved in database. The command is only be used to undo changes since the last `COMMIT`.
- Rolling back to a savepoint erases any savepoints marked after that savepoint. However, the savepoint to which you rollback is not erased.
### `ROLLBACK TO`
```sql
ROLLBACK TO <<Savepoint Name>>
```

# DML(Data Manipulation Language)
- `INSERT`(Create), `SELECT`(Read), `UPDATE`(Update), `DELETE`(Delete)
- `MERGE` (Oracle)
## `INSERT`
### `INSERT INTO VALUES()`
```sql
INSERT INTO <<Table>>(<<Column1>>, <<Column2>>, ...) VALUES(<<Value1>>, <<Value2>>, ...);
```
- If you are adding values for all the columns of the table, you do not need to specify the column names in the SQL query.
### `INSERT INTO SELECT FROM`
```sql
INSERT INTO <<Target Table>>
SELECT
FROM <<Source Table>>
```
- Source: https://www.w3schools.com/sql/sql_insert_into_select.asp
- The `INSERT INTO SELECT FROM` statement copies data from `<<Source Table>>` and inserts it into `<<Target Table>>`.
- The `INSERT INTO SELECT FROM` statement requires that the data types in `<<Source Table>>` and `<<Tartget Table>>` match.
- The existing records in the `<<Target Table>>` are unaffected.
#### `INSERT ALL INTO SELECT FROM`
- When using an unconditional `INSERT ALL` statement, each row produced by the driving query results in a new row in each of the tables listed in the `INTO` clauses.
```sql
INSERT ALL
  INTO pivot_dest(id, day, val) VALUES(id, 'mon', mon_val)
  INTO pivot_dest(id, day, val) VALUES(id, 'tue', tue_val)
  INTO pivot_dest(id, day, val) VALUES(id, 'wed', wed_val)
  INTO pivot_dest(id, day, val) VALUES(id, 'thu', thu_val)
  INTO pivot_dest(id, day, val) VALUES(id, 'fri', fri_val)
SELECT *
FROM   pivot_source;
```
#### `INSERT FIRST WHEN THEN INTO ELSE INTO SELECT FROM`
```sql
INSERT FIRST
  WHEN id <= 3 THEN INTO dest_tab1(id, description) VALUES(id, description)
  WHEN id <= 5 THEN INTO dest_tab2(id, description) VALUES(id, description)
  ELSE INTO dest_tab3(id, description) VALUES(id, description)
SELECT id, description
FROM   source_tab;
```
- Using `INSERT FIRST` makes the multitable insert work like a `CASE` expression, so the conditions are tested until the first match is found, and no further conditions are tested.
## `SELECT`
### `SELECT TOP FROM` (MS SQL Server), `SELECT FROM LIMIT` (MySQL), `SELECT FROM FETCH FIRST ROWS ONLY` (Oracle)
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
### `SELECT FROM ORDER BY`
- ***`ORDER BY`가 정렬을 하는 시점은 모든 실행이 끝난 후 데이터를 출력하기 바로 직전이다.***
- `ORDER BY <<Number>>`
	- `<Number>>` stands for the column based on the number of columns defined in the `SELECT` clause.
- ***Oracle: NULL values are larger than any non-NULL values.***
- ***MS SQL Server: NULL values are smaller than any non-NULL values.***
#### `SELECT FROM ORDER BY NULLS FIRST | NULLS LAST`
### `SELECT TOP PERCENT FROM` (MS SQL Server), `SELECT FROM FETCH FIRST PERCENT ROWS ONLY` (Oracle)
### `SELECT ROWNUM FROM`, `SELECT FROM WHERE ROWNUM`
- For each row returned by a query, the `ROWNUM` pseudocolumn returns a number indicating the order in which Oracle selects the row from a table or set of joined rows. The first row selected has a `ROWNUM` of 1, the second has 2, and so on.
```sql
SELECT ROWNUM, pr.product_name, pr.standard_cost
FROM products;
```
- ***Execution order: `ROWNUM` -> `ORDER BY`***
#### `SELECT FROM ORDER BY HAVING`
- ***`HAVING`은 `GROUP BY`문의 조건절이므로 `GROUP BY` 함수 없이 사용되면 `SELECT`문에서 어떤 행도 반환하지 않습니다.***
### `SELECT FROM WHERE EXISTS()`
- Source: https://www.w3schools.com/sql/sql_exists.asp
- The `EXISTS` operator is used to test for the existence of any record in a subquery.
- The `EXISTS` operator returns TRUE if the subquery returns one or more records.
```sql
SELECT *
FROM customers
WHERE EXISTS(
	SELECT *
	FROM orders
	WHERE orders.c_id = customers.c_id);
```
- The above statement returns TRUE and lists the customers who have ever ordered.
## `UPDATE`
### `UPDATE SET WHERE`
```
UPDATE emp
SET ename = 'jojo'
WHERE emp_no = 100;
```
## `DELETE`
### `DELETE [FROM] [WHERE]`
- Here we can use the `ROLLBACK` command to restore the tuple.
```sql
DELETE [FROM] emp
WHERE emp_no = 100;
```
## `MERGE` (Oracle)
- Source: https://www.oracletutorial.com/oracle-basics/oracle-merge/
- The `MERGE` statement selects data from one or more source tables and updates or inserts it into a target table. The `MERGE` statement allows you to specify a condition to determine whether to update data from or insert data into the target table.
- Because the `MERGE` is a deterministic statement, you cannot update the same row of the target table multiple times in the same `MERGE` statement.
## `MERGE INTO USING ON WHEN MATCHED THEN UPDATE SET WHERE [DELETE WHERE] WHEN NOT MATCHED THEN INSERT () VALUES() WHERE`
```sql
MERGE INTO <<Target Table>>
USING <<Source Table>> 
ON <<Search Condition>>
WHEN MATCHED THEN
	UPDATE SET <<Column1>> = <<Value1>>, <<Column2>> = <<Value2>>, ...
	WHERE <<Update Condition>>
	[DELETE WHERE <<Delete Condition>>]
WHEN NOT MATCHED THEN
	INSERT (<<Column1>>, <<Column2>>, ...)
	VALUE(<<Value1>>, <<Value2>>, ...)
	WHERE <<Insert Condition>>;
```
- `<<Search Condition>>`
	- For each row in the target table, Oracle evaluates the `<<Search Condition>>`
		- If the result is true, then Oracle updates the row with the corresponding data from the source table.
		- In case the result is false for any rows, then Oracle inserts the corresponding row from the source table into the target table.
- `<<Delete Condition>>`
	- You can add an optional `DELETE WHERE` clause to the `MATCHED` clause to clean up after a merge operation. The `DELETE` clause deletes only the rows in the target table that match both `ON` and `DELETE WHERE` clauses.
	- Source: https://oracle-base.com/articles/10g/merge-enhancements-10g
	- Only those rows in the destination table that match both the ON clause and the `DELETE WHERE` are deleted. If you add a `WHERE` clause to the update in the matched clause, we can think of this as additional match criteria for the delete, as only rows that are touched by the update are available for the `DELETE` clause to remove. Depending on which table the `DELETE WHERE` references, it can target the rows prior or post update.

# Hierarchical Queries
## `START WITH CONNECT BY PRIOR ORDER SIBLINGS BY`
- `START WITH` specifies the root row(s) of the hierarchy. `START WITH` 다음에 조건을 명시할 수도 있다.
- `CONNECT BY` specifies the relationship between parent rows and child rows of the hierarchy. It always joins a table to itself, not to another table.
- `PRIOR` should occur exactly once in each `CONNECT BY` expression. `PRIOR` can occur on either the left-hand side or the right-hand side of the expression, but not on both.
- If the query contains a `WHERE` clause without a join, then Oracle eliminates all rows from the hierarchy that do not satisfy the condition of the `WHERE` clause. ***Oracle evaluates this condition for each row individually, rather than removing all the children of a row that does not satisfy the condition.***
- For each row returned by a hierarchical query, the `LEVEL` pseudocolumn returns 1 for a root row, 2 for a child of a root, and so on
- The `CONNECT_BY_ISLEAF` pseudocolumn returns 1 if the current row is a leaf of the tree defined by the `CONNECT BY` condition, 0 otherwise.
```sql
CONNECT_BY_ROOT last_name AS boss
```
- The `CONNECT_BY_ROOT` pseudocolumn returns the root.
```sql
SELECT last_name, employee_id, manager_id, LEVEL
FROM employees
START WITH employee_id = 100
CONNECT BY PRIOR employee_id = manager_id
ORDER SIBLINGS BY last_name;
```
- If you want to order rows of siblings of the same parent, then use the `ORDER SIBLINGS BY` clause.
```sql
SELECT last_name "Employee", CONNECT_BY_ISCYCLE "Cycle",
LEVEL, SYS_CONNECT_BY_PATH(last_name, '/') "Path"
FROM employees
WHERE level <= 3 AND department_id = 80
START WITH last_name = 'King'
CONNECT BY NOCYCLE PRIOR employee_id = manager_id AND LEVEL <= 4;
```
- The `NOCYCLE` parameter in the `CONNECT BY` condition causes Oracle to return the rows in spite of the loop.
- The `CONNECT_BY_ISCYCLE` pseudocolumn shows you which rows contain the cycle.
```sql
SYS_CONNECT_BY_PATH(<<Column>>, <<Character>>)
```
- The `SYS_CONNECT_BY_PATH` function returns the path of a `<<Column>>` value from root to node, with column values separated by `<<Character>>` for each row returned by `CONNECT BY` condition.

# NULL Functions
- Source: https://en.wikipedia.org/wiki/Null_(SQL)
- A null value indicates a lack of a value, which is not the same thing as a value of zero. For example, consider the question "How many books does Adam own?" The answer may be "zero" (we know that he owns none) or "null" (we do not know how many he owns). In a database table, the column reporting this answer would start out with no value (marked by NULL), and it would not be updated with the value "zero" until we have ascertained that Adam owns no books.
- SQL null is a state, not a value.
- NULL means that the value is unknown.
## `NVL()` (Oracle)
```sql
NVL(<<Column to Test>>, <<Value if NULL>>)
```
- Return `<<Column to Test>>` if not NULL otherwise return `<<Value if NULL>>` if NULL.
## `NVL2()` (Oracle)
```sql
NVL2(<<Column to Test>>, <<Value if not NULL>>, <<Value if NULL>>)
```
- Return `<<Value if not NULL>>` if <<Column to Test>> is not NULL otherwise return `<<Value if NULL>>` if `<<Column to Test>>` is NULL.
## `COALESCE()` (Oracle, MySQL)
```sql
COALESCE(<<Expression1>>, <<Expression2>>, ...)
```
- Return the first non-NULL expression in the list. If all expressions evaluate to null, return NULL
## `NULLIF()` (Oracle, MS SQL Server)
```sql
NULLIF(<<Expression1>>, <<Expression2>>)
```
- Return NULL if `<<Expression1>>` and `<<Expression2>>` are equal and return `<<Expression1>>` if `<<Expression1>>` and `<<Expression2>>` are not equal.
## `IFNULL()` (MySQL)
```sql
SELECT animal_type, IFNULL(name, "No name"), sex_upon_intake
FROM animal_ins
ORDER BY animal_id
```
## `ISNULL()` (MS SQL Server)
## `IS NULL`, `IS NOT NULL`
- Source: https://www.w3schools.com/sql/sql_null_values.asp
- It is not possible to test for NULL values with comparison operators(`=`, `!=`, `<`, `<=`, `>`, `>=`).
- `NULL IS NULL` is TRUE and `NULL IS NOT NULL` is FALSE.
- Because NULL represents a lack of data, a NULL cannot be equal or unequal to any value or to another NULL. However, Oracle considers two NULLs to be equal when evaluating a `DECODE()` function.(So `NULL IN(NULL, ...)` is FALSE)

# Logical Functions
## `CASE WHEN THEN ELSE END`
- Search for conditions sequentially.
```sql
SELECT CASE WHEN (a >= b + c OR b >= c + a OR c >= a + b) THEN "Not A Triangle" WHEN (a = b AND b = c) THEN "Equilateral" WHEN (a = b OR b = c OR c = a) THEN "Isosceles" ELSE "Scalene" END
FROM triangles;
```
## `DECODE()`
```sql
DECODE(<<Expression>>, <<Value to Be Compared1>>, <<Value If Equal to1>>, <<Value to Be Compared2>>, <<Value If Equal to2>>, ..., <<Default>>)
```
- `<<Default>>`: (Optional) It is used to specify the default value to be returned if no matches are found.
## `ANY()`
- Return TRUE if any of the subquery values meet the condition.
```sql
SELECT name
FROM usertbl
WHERE height > ANY(
	SELECT height
	FROM usertbl
	WHERE addr = '서울');
```
## `ALL()`
- Return TRUE if all of the subquery values meet the condition.
- Used with `SELECT`, `WHERE` and `HAVING` statements.
## `IN()`, `NOT IN()`
```sql
SELECT name, addr
FROM usertbl
WHERE addr IN("서울", "경기", "충청");
```

# Numeric Functions
## `CEIL()`(= `CEILING()`), `FLOOR()`
- Returns the smallest(largest) integer value that is bigger(smaller) than or equal to a number.
## `ROUND()`
```sql
ROUND(<<Target Number>>, [<<Decimal Place>>])
```
- If no <<Decimal Place>> is defined (Oracle) or equal to zero (MS SQL Server), then <<Target Number>> is rounded to zero places.
- If the <<Decimal Place>> specified is negative, then <<Target Number>> is rounded off to the left of the decimal point.
- If the <<Decimal Place>> specified is positive, then <<Target Number>> is rounded up to <<Decimal Place>> decimal place(s).
## `ABS()`
## `FORMAT()` (MS SQL Server)
```sql
SELECT FORMAT(123456.123456, 4);
```
- 출력할 소수점 이하 자릿수 지정. 
# `BETWEEN AND`
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
## `SUBSTR()`(= `SUBSTRING()`)
```sql
SUBSTR(<<String>>, <<Start Position>>, [<<Length>>])
```
- `<<String>>`: The first position in the `<<String>>` is always 1, the last -1.
- `<<Length>>`: If this parameter is omitted, the `SUBSTR` function will return the entire `<<String>>`.
## `LOWER()`, `UPPER()`, `INITCAP()`
```sql
UPPER('eabc')
```
## `LPAD()`, `RPAD()`
- The `LPAD()` function left-pads a string with another string, to a certain length.
```sql
LPAD(<<Original String>>, <<Target Length>>, [<<String to Pad>>])
```
- `<<Target Length>>`: The length of the string after it has been left-padded. Note that if the `<<Target Length>>` is less than the length of the `<<Original String>>`, then `LPAD()` function will shorten down the `<<Original String>>` to the `<<Target Length>>` without doing any padding.
- `<<String to Pad>>` is equal to `' '` if omitted.
## `LTRIM()`, `RTRIM()`
## `TRIM()`
### `SELECT TRIM([[LEADING | TRAILING | BOTH] FROM]) FROM`
```sql
TRIM('   tech   ')
```
```sql
TRIM(' ' FROM '   tech   ')
```
```sql
TRIM(LEADING '0' FROM '000123')
```
```sql
TRIM(TRAILING '1' FROM 'Tech1')
```
```sql
TRIM(BOTH '1' FROM '123Tech111')
```
## `CONCAT()` (All)
```sql
SELECT CONCAT(CASE hdc_mbr.mbr_mst.mbr_sex WHEN '0' THEN '남자' WHEN '`' THEN '여자' END, '/', hdc_mbr.mbr_mst.mbr_birth)
```
## `CONCAT_WS()`
```sql
CONCAT_WS(<<Separator>>, <<Expression1>>, <<Expression2>>, ...)
```
- The `CONCAT_WS()` function adds `<<Expression1>>`, `<<Expression2>>`, ... together with a `<<Separator>>`.
## `SELECT FROM WHERE LIKE`
```sql
SELECT DISTINCT city
FROM station
WHERE city LIKE "%a" OR city LIke "%e" OR city LIKE "%i" OR city LIKE "%o" OR city LIKE "%u";
```
- A wildcard character is used to substitute one or more characters in a string.
	- `%`: Represents zero, one, or multiple characters.
	- `_`: Represents one, single character.
	- `-`: Represents any single character within the specified range
## `SELECT FROM WHERE LIKE ESCAPE`
```sql
SELECT *
FROM tbl
WHERE name LIKE '%@_%` ESCAPE '@';
```
## `SELECT TO_CHAR() FROM`
- Used to convert a number or date to a string.
```sql
TO_CHAR(mbrmst.mbr_leaving_expected_date, "YYYY/MM/DD") AS mbr_leaving_expected_dt
```

# Date Functions
## `DATE_FORMAT()` (MySQL)
## `HOUR()`
```sql
SELECT HOUR(datetime) AS hour, COUNT(*)
FROM animal_outs
GROUP BY hour
HAVING hour BETWEEN 9 AND 19
ORDER BY hour;
```
- HOUR()는 일반조건이므로 (COUNT()와 달리) HAIVNG과 함께 쓸 수 없다.
```
## `SYSDATE` (Oracle), `CURDATE()` (MySQL), `GETDATE()` (MS SQL Server)
## `EXTRACT([YEAR | MONTH | DAY | HOUR | MINUTE | SECOND] FROM)`
- Returns a numeric value according to the parameter.
```sql
EXTRACT(MONTH FROM order_date)
```
## `CALENDAR_YEAR()`, `CALENDAR_QUARTER()`, `CALENDAR_MONTH()`, `DAY_IN_MONTH()`, `HOUR_IN_DAY()` (Salesforce Object Query Language)
## `DATE()` (MS SQL Server)
## `DATEADD([YEAR | MONTH | DAY | HOUR | MINUTE | SECOND])`
```sql
DATEADD(<<Interval>>, <<Number>>, <<Date>>)
```
- <<Number>>: Required. The number of interval to add to <<Date>>.
- <<Date>>:	Required. The date that will be modified.
## `DATEDIFF([YEAR | MONTH | DAY])`
```sql
DATEDIFF(YEAR, 0, GETDATE())
```
- `0` means the default date of `1900-01-01 00:00:00.000`
## `LAST_YEAR`, `THIS_YEAR`, `NEXT_YEAR`, `LAST_MONTH`, `THIS_MONTH`, `NEXT_MONTH`, `YESTERDAY`, `TODAY`, `TOMORROW` (Salesforce Object Query Language)

# Data Type Conversion Functions
## `CAST(AS)`
```sql
CAST(COUNT(occupation) AS CHAR)
```
## `STR()`
## `YEAR`, `MONTH`, `DAY`, `HOUR`, `MINUTE`, `SECOND`
## `TO_CHAR()`, `TO_DATE()` (Oracle)
```sql
TO_CHAR(1340.64, '9999.9')
TO_CHAR(sysdate, 'Month DD, YYYY')
```

# Window Functions
- Source: https://docs.oracle.com/cd/E17952_01/mysql-8.0-en/window-functions-usage.html
- Window functions are aggregates based on a set of rows, similar to an aggregate function like a `GROUP BY`, but in this case this aggregation of rows moves or slides across a number of rows so we have a sort of sliding window, thus the name window functions.
- Window operations do not collapse groups of query rows to a single output row. Instead, they produce a result for each row.
- ***`GROUP BY`와 함께 사용할 수 없습니다.***
## `OVER()`
- An empty `OVER` clause treats the entire set of query rows as a single partition. The window function thus produces a global sum, but does so for each row.
### `OVER([PARTITION BY])`
- `OVER(PARTITION BY)` clause partitions rows by a column, producing a sum per partition. The function produces this sum for each partition row.
#### `OVER([PARTITION BY] [ORDER BY])`
- An `ORDER BY` clause indicates how to sort rows in each partition. If `ORDER BY` is omitted, partition rows are unordered.
- An `ORDER BY` in a window definition applies within individual partitions. To sort the result set as a whole, include an `ORDER BY` at the query top level.
#### `OVER([PARTITION BY] [ORDER BY] [ROWS BETWEEN AND])`, `OVER([PARTITION BY] [ORDER BY] [RANGE BETWEEN AND])`
- Source: https://learnsql.com/blog/range-clause/
- The `RANGE` and the `ROW` clauses have the same purpose: to specify the starting and ending points within the partition, with the goal of limiting rows. However, each clause does it differently. The `ROW` clause does it by specifying a fixed number of rows that precede or follow the current row. The `RANGE` clause, on the other hand, limits the rows logically; it specifies the range of values in relation to the value of the current row.
- The default window frame without the `ORDER BY` is the whole partition. But when you use the `ORDER BY`, the default window frame is `RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW`(= `UNBOUNDED PRECEDING`).
- `BETWEEN <<Start Point>> AND <<End Point>>`
	- `<<Start Point>>`: `UNBOUNDED PRECEDING`, `n PRECEDING`, `CURRENT ROW`
	- `<<End Point>>`: `UNBOUNDED FOLLOWING`, `m FOLLOWING`, `CURRENT ROW`
- `<<Start Point>>`
	- In this case, `<<End Point>>` is same as `CURRENT ROW`(`<<End Point>>` is omitted)
```sql
SELECT month, SUM(tot_sales) AS monthly_sales,
	AVG(SUM(tot_sales)) OVER(ORDER BY month RANGE BETWEEN 1 PRECEDING AND 1 FOLLOWING) AS rolling_avg
FROM orders
```
## Aggregate Window Functions
### `MIN()`, `MAX()`
### `AVG()`, `SUM()`
- ***NULL values are ignored.***
### `COUNT()`
- ***`COUNT(*)`, `COUNT(1)`, `COUNT(2)`, ...: Return the number of rows in the table.***
- ***`COUNT(<<Column>>)`: Return the number of non-NULL values in the column.***
- ***`COUNT(DISTINCT <<Column>>)`: Return the number of distinct non-NULL values in the column.***
## Rank Window Functions
- Source: https://codingsight.com/similarities-and-differences-among-rank-dense_rank-and-row_number-functions/, https://www.sqlshack.com/overview-of-sql-rank-functions/
### `RANK()`
- If there is a tie between N previous records for the value in the `ORDER BY` column, the `RANK()` function skips the next N-1 positions before incrementing the counter.(e.g., 1, 2, 2, 4, 4, 4, 7)
```sql
SELECT emp_no, emp_nm, sal,
	RANK() OVER(ORDER BY salary DESC) ranking
FROM employee;
```
### `DENSE_RANK()`
- `DENSE_RANK()` function does not skip any ranks if there is a tie between the ranks of the preceding records.(e.g., 1, 2, 2, 3, 3, 3, 4)
### `ROW_NUMBER()`
- Give a unique sequential number for each row in the specified data. It gives the rank one for the first row and then increments the value by one for each row. We get different ranks for the row having similar values as well.(e.g., 1, 2, 3, 4, 5, 6, 7)
- `ROW_NUMBER()`을 사용하면 Inline view를 사용하지 않고도 어떤 값의 순서대로 순번을 매길 수 있습니다.
### `NTILE()`
- Source: https://www.sqltutorial.org/sql-window-functions/sql-ntile/
- `NTILE()` is a window function that allows you to break the result set into a specified number of approximately equal groups, or buckets. It assigns each group a bucket number starting from one. For each row in a group, the `NTILE()` function assigns a bucket number representing the group to which the row belongs.
- The `ORDER BY` clause specifies the order of rows in each partition to which the `NTILE()` is applied.
- Notice that if the number of rows is not divisible by buckets, the `NTILE()` function results in groups of two sizes with the difference by one. The larger groups always come before the smaller group in the order specified by the `ORDER BY` clause.
## Value Window Functions 
### `LAG()`, `LEAD()`
- The `LAG` function is used to access data from a previous row.
- The `LEAD` function is used to return data from rows further down the result set.
### `FIRST_VALUE() [RESPECT | IGNORE NULLS]`, `LAST_VALUE() [RESPECT | IGNORE NULLS]`
- `RESPECT NULLS | IGNORE NULLS`: It is an optional parameter which is used to specify whether to include or ignore the NULL values in the calculation. The default value is `RESPECT NULLS`.
```sql
SELECT DISTINCT FIRST_VALUE(marks)
OVER(ORDER BY marks DESC RANGE BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS `Highest`
FROM students;
```

# Group Functions
## `SELECT FROM GROUP BY`
- `GROUP BY` treats NULL as valid values.
- All NULL values are grouped into one value or bucket.
### `SELECT FROM GROUP BY ROLLUP()`
- ***`GROUP BY ROLLUP(<<Column1>>, <<Column2>>, ...)` is same as `GROUP BY GROUPING SETS(<<Column1>>, (<<Column1>>, <<Column2>>), ..., ())`***
- ***계층 구조로 계층 간 정렬이 가능합니다.***
### `SELECT FROM GROUP BY CUBE()`
- Generate subtotals for all combinations of the dimensions specified.
- ***`GROUP BY CUBE(<<Column1>>, <<Column2>>, ...)` is same as `GROUP BY GROUPING SETS(<<Column1>>, <<Column2>>, ..., (<<Column1>>, <<Column2>>, ...), ())`***
### `SELECT FROM GROUP BY GROUPING SETS()`
- `GROUP BY GROUPING SETS((<<Column1>>, <<Column2>>, ...))`
	- `GROUP BY <<Column1>>, <<Column2>>, ...`과 동일합니다.
- `GROUP BY GROUPING SETS(())`
	- 전체에 대한 집계.
- 인자가 여러 개인 경우 위 3가지 경우의 각 결과를 `UNION ALL`한 것과 같음.

# Set Operators
## `UNION`, `UNION ALL`
### `SELECT FROM UNION(UNION ALL) SELECT FROM`
- The `UNION` operator eliminates duplicate selected rows. The sort is implicit in order to remove duplicates since.
## `INTERSECT`, `INTERSECT ALL`
### `SELECT FROM INTERSECT(INTERSECT ALL) SELECT FROM`
- The `INTERSECT` operator returns only those rows returned by both queries.
- ***The `INTERSECT` operator removes duplicate rows from the final result set.***
- The `INTERSECT ALL` operator does not remove duplicate rows from the final result set, but if a row appears X times in the first query and Y times in the second, it will appear min(X, Y) times in the result set.
## `MINUS` (Oracle), `EXCEPT`, `EXCEPT ALL` (MS SQL Server)
### `SELECT FROM MINUS SELECT FROM`, `SELECT FROM EXCEPT(EXCEPT ALL) SELECT FROM`
- Return all rows in the first `SELECT` statement that are not returned by the second `SELECT` statement.
- ***The `MINUS`(`EXCEPT`) operator removes duplicate rows from the final result set.***
- The `EXCEPT ALL` operator does not remove duplicates, but if a row appears X times in the first query and Y times in the second, it will appear max(X - Y, 0) times in the result set.

# `PIVOT`, `UNPIVOT`
```sql
SELECT 반정보, 과목, 점수
FROM dbo.성적	UNPIVOT (점수	FOR 과목 IN(국어, 수학, 영어)) AS UNPVT
```
- `PIVOT` transforms rows to columns.
- `UNPIVOT` transforms columns to rows.

# Join
## `INNER JOIN`(= `JOIN`), `LEFT OUTER JOIN`, `LEFT OUTER JOIN`, `FULL OUTER JOIN`
- `FULL OUTER JOIN` is same as `LEFT OUTER JOIN UNION ALL RIGHT OUTER JOIN`.
## `CROSS JOIN`
- In Mathematics, given two sets `A` and `B`, the Cartesian product of `AxB` is the set of all ordered pair `(a, b)`, which `a` belongs to `A` and `b` belongs to `B`.
- Join key가 없을 때 발생합니다.
- The `CROSS JOIN` produces a result set which is the number of rows in the first table multiplied by the number of rows in the second table if no `WHERE` clause is used along with `CROSS JOIN`.This kind of result is called as Cartesian product.
- If `WHERE` clause is used with `CROSS JOIN`, it functions like an `INNER JOIN`.
```sql
SELECT foods.item_name,foods.item_unit, company.company_name,company.company_city 
FROM foods CROSS JOIN company;
```
## `NATURAL JOIN`
- A `NATURAL JOIN` is a join operation that creates an implicit join clause for you based on the common columns in the two tables being joined. Common columns are columns that have the same name in both tables.
- ***`<<Table>>.<<Column>>`과 같은 형태로 사용할 수 없습니다.***
### `NATURAL INNER JOIN`(= `NATURAL JOIN`), `NATURAL LEFT OUTER JOIN`, `NATURAL RIGHT OUTER JOIN`
- 
```sql
SELECT *
FROM countries NATURAL INNER JOIN cities;
```
```sql
SELECT *
FROM countries INNER JOIN cities
	USING(country, country_iso_code)
```
- The above two statements are the same.
### Join with `+` Operator
```sql
FROM A LEFT OUTER JOIN B ON A.<<Column1>> = B.<<Column2>>
```
```sql
FROM A, B
WHERE A.<<Column1>> = B.<<Column2>>(+)
```
- The above two statements are the same.
```sql
FROM A RIGHT OUTER JOIN B ON A.<<Column1>> = B.<<Column2>>
```
```sql
FROM A, B
WHERE A.<<Column1>>(+) = B.<<Column2>>
```
- The above two statements are the same.
## Equi Join
- Source: https://www.w3resource.com/sql/joins/perform-an-equi-join.php
- Equi join performs a join against equality or matching column(s) values of the associated tables. ***An equal sign(`=`) is used as comparison operator in the where clause to refer equality.***
- Salesforce Object Query Language에서는 Equi join만 사용 가능합니다.
## Non-Equi Join
- Source: https://learnsql.com/blog/illustrated-guide-sql-non-equi-join/
- `!=`, `<`, `<=`, `>`, `>=`, `BETWEEN AND`

# Subquery
- Source: https://www.geeksforgeeks.org/sql-subquery/
- A subquery is a query within another query. The outer query is called as main query and inner query is called as subquery.
- The subquery generally executes first, and its output is used to complete the query condition for the main or outer query.
- Subquery must be enclosed in parentheses.
- Subqueries are on the right side of the comparison operator.
- ***`ORDER BY` cannot be used in a Subquery. `GROUP BY` can be used to perform same function as `ORDER BY`.***
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
- A subquery can be placed in `SELECT` clause(Scala Subquery), `FROM` clause(Inline View), `WHERE` clause(Subquery), `HAVING` clause.
## Inline View
- Inline view is a subquery in the `FROM` clause of a `SELECT` statement.
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
- Scala subquery should be a single-row subquery.
## Correlated Subquery(= 상관 서브 쿼리)
- Source: https://myjamong.tistory.com/176
- ***내부 Subquery에서 외부테이블의 값을 참조할 때 사용됩니다.***
- Subquery와는 다르게  Inner Query 부터 Outer Query 순서대로 실행되는 것이 아니라 Outer Query에서 읽어온 행을 갖고 Inner쿼리를 실행하는 것을 반복하여 결과를 출력해줍니다.
- Outer Query와 Inner Query에서 같은 테이블을 참조한다면 Outer Query의 테이블에 Alias를 사용해서 구분해줍니다.