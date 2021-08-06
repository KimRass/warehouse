## FLUSH PRIVILEGES
```sql
DELETE FROM mysql.user WHERE user="6363";
DELETE FROM mysql.db WHERE user="6363";
FLUSH PRIVILEGES;
```
```sql
GRANT ALL PRIVILEGES ON masterdata.* TO "6363"@"%";
FLUSH PRIVILEGES;
```
- WITH GRANT OPTION : 권한 위임 가능
```sql
SHOW GRANTS FOR "6363"@"%";
```
#### CREATE
##### CREATE USER
```
CREATE USER
	"6363"@"%"
IDENTIFIED BY
	"6363";
```
##### SELECT + WITH ROLLUP
```sql
SELECT num, groupname, SUM(price*amount) AS "비용"
FROM buytbl
GROUP BY num, groupname
WITH ROLLUP;
```
##### SELECT + LAST_INSERT_ID();
- AUTO_INCREMENT 사용 시 사용
##### SELECT + @
```sql
SELECT @변수 이름;
```
### INSERT
#### INSERT INTO
```sql
INSERT INTO testtbl VALUES (1, "홍길동", 25);
```
```sql
INSERT INTO testtbl(age, id) VALUES (25, NULL);
```
- 값을 NULL로 주면 AUTO_INCREMENT에 따라 값 지정
#### INSERT IGNORE INTO
- 에러가 발생해도 무시하고 진행
#### INSERT INTO + SELECT
```sql
INSERT INTO testtbl1;
SELECT emp_no, first_name, last_name
FROM testtbl2;
```
#### CREATE TABLE + SELECT : copy table
```sql
CREATE TABLE testtbl2
(SELECT emp_no, first_name, last_name
FROM testtbl1);
```
#### CREATE TABLE + LIKE : copy table structure
```sql
CREATE TABLE membertbl LIKE usertbl;
```
#### CREATE INDEX
```sql
CREATE INDEX idx_indexTBL_firstname ON indexTBL(first_name);
```
#### CREATE VIEW
```sql
CREATE VIEW uv_membertbl
SELECT membername, memberaddress FROM membertbl;
```
### ALTER
#### ALTER TABLE + AUTO_INCREMENT + SET
```sql
ALTER TABLE testtbl AUTO_INCREMENT=100;
SET @@auto_increment_increment=3;
INSERT INTO testtbl VALUES (NULL, "홍길동", 25);
```
- AUTO_INCREMENT 값부터 3씩 건너뛰어 id 지정
### SHOW
```sql
SHOW DATABASES;
```
```sql
SHOW TABLES;
```
```sql
SHOW TABLE STATUS;
```
### DESCRIBE
```sql
DESCRIBE table;
```
### DELIMITTER
```sql
DELIMITTER //
BEGIN
END //
```
### USE
```sql
USE database;
```
# data type
## number
- TINYINT , SMALLINT, MEDIUMINT, INT, BIGINT
- FLOAT, DOUBLE
- 부호 없는 정수 지정 시 UNSIGNED를 붙임.
## string
- CHAR(n) : 항상 n자리
-  VARCHAR(n) : 최대 n자리
- TINYTEXT, TEXT, MEDIUMTEXT, LONGTEXT : 최대 4GB
## date and time
- DATE : "YYYY-MM-DD" 형식
- TIME : "HH:MM:SS" 형식
- DATETIME : "YYYY-MM-DD HH:MM:SS" 형식
### PREPARE
#### PREPARE + EXECUTE + USING
```sql
SET @var1=3;

PREPARE query1
FROM "SELECT name, height FROM usertbl ORDER BY height LIMI ?;

EXECUTE query1 USING @var1;
```
### CAST(CONVERT)
```sql
SELECT COUNT(mobile) AS "핸드폰을 사용하는 사람의 수" FROM usertbl;
```
```sql
SELECT CAST(COUNT(mobile) AS SIGNED INTEGER) AS "핸드폰을 사용하는 사람의 수" FROM usertbl;
```
```sql
SELECT CONVERT(COUNT(mobile), SIGNED INTEGER) AS "핸드폰을 사용하는 사람의 수" FROM usertbl;
```
#### CAST + CONCAT
```sql
SELECT CONCAT(CAST(price AS CHAR(10)), " X ", CAST(amount AS CHAR(4)), " = ") AS "단가 X 수량", price*amount AS "구매액"
FROM buytbl;
```
### IF
```sql
SELECT IF (100>200, "참이다", "거짓이다");
```
### IFNULL
```sql
SELECT IFNULL(value1, value2);
```
- returns value2 if value1 is NULL and returns value1 if value1 is not NULL.
### NULLIF
```sql
SELECT NULLIF(value1, value2);
```
- returns value2 if value1=value2 and returns value1 if value1!=value2
#### REPEAT
```sql
SELECT REPEAT("문자반복", 3);
```
#### REPLACE
```sql
SELECT REPLACE("이것이 데이터베이스다", "이것이", "저것이");
```
#### REVERSE
```sql
SELECT REVERSE("문자뒤집기");
```
#### SUBSTRING
```sql
SELECT SUBSTRING("대한민국만세", 3, 2);
```
#### SUBSTRING_INDEX
```sql
SELECT SUBSTRING_INDEX("cafe.naver.com", ".", 2); #cafe.naver
SELECT SUBSTRING_INDEX("cafe.naver.com", ".", -2); #naver.com
```
#### ADDDATE, SUBDATE, ADDTIME, SUBTIME
```sql
ADDDATE("2020-01-01", INTERVAL 31 DAY);
```
```sql
ADDTIME("2020-01-01 12:31:59", "1:10:21");
```
#### CURDATE, CURTIME, CURRENT_TIME, NOW
#### DATEDIFF
#### DAYOFWEEK, DAYOFYEAR
#### MONTHNAME
#### TIME_TO_SEC
#### SLEEP
# pivot table
```sql
SELECT username,
    SUM(IF(season="봄", amount, 0)) AS "봄",
    SUM(IF(season="여름", amount, 0)) AS "여름",
    SUM(IF(season="가을", amount, 0)) AS "가을",
    SUM(IF(season="겨울", amount, 0)) AS "겨울",
    SUM(amount) AS "합계"
FROM pivottest GROUP BY username;
```
# JSON
## JSON_OBJECT
```sql
SELECT JASON_OBJECT("이름", name, "키", height) AS "JSON 값"
FROM usertbl
WHERE height>=180;
```
# MySQL Command Line Client
## show databases;
--------------------------------------------------
# ORDER
- ORDER BY > LIMIT
- GROUP BY > HAVING
- WHERE > GROUP BY
## SET
```sql
SET @hour := -1;

SELECT (@hour := @hour + 1) as "hour",
(SELECT COUNT(*) FROM animal_outs WHERE @hour = HOUR(datetime)) as "count"
FROM animal_outs
WHERE @hour < 23;
```
```sql
SET @nameis="가수 이름 =>";

SELECT @nameis, name FROM usertbl WHERE height>180;
```
------------------------------------------------------------------------------
## SELECT FROM
### CASE WHEN END
```sql
SELECT
CASE hdc_mbr.mbr_mst.mbr_sys_ty
WHEN 'C000004001' THEN '입주예정자'
WHEN 'C000004003' THEN '입주자(대표)'
WHEN 'C000004004' THEN '입주자(세대원)'
END AS mbr_sys_ty
```
### CONCAT
```sql
SELECT
CONCAT(CASE hdc_mbr.mbr_mst.mbr_sex
WHEN "0" THEN "남자"
WHEN "`" THEN "여자"
END, "/", hdc_mbr.mbr_mst.mbr_birth)
```
### WHERE
### COLUMN_NAME
```sql
SELECT COLUMN_NAME
FROM INFORMATION_SCHEMA.COLUMNS;
```
### IN
```sql
SELECT name, addr
FROM usertbl
WHERE addr IN ("서울", "경기", "충청");
```
### BETWEEN AND
```sql
SELECT name, height
FROM sqlDB.usertbl
WHERE height BETWEEN 180 AND 183;
```
### LIKE
```sql
SELECT animal_ins.animal_id, animal_ins.animal_type, animal_ins.name
FROM animal_ins
INNER JOIN animal_outs
ON animal_ins.animal_id = animal_outs.animal_id
WHERE animal_ins.sex_upon_intake LIKE "Intact%"
AND (animal_outs.sex_upon_outcome LIKE "Spayed%"
OR animal_outs.sex_upon_outcome LIKE "Neutered%")
ORDER BY animal_ins.animal_id;
```
### ANY()
```sql
SELECT name
FROM usertbl
WHERE height > ANY (SELECT height FROM usertbl WHERE addr="서울");
```
## LIKE
```sql
SELECT name, height
FROM usertbl
WHERE name LIKE "김%" OR "_종신";
```
## ORDER BY
### ASC, DESC
```sql
SELECT animal_id, name, datetime
FROM animal_ins
ORDER BY name ASC, datetime DESC;
```
## LIMIT
```sql
SELECT name
FROM animal_ins
ORDER BY datetime
LIMIT 1;
```
## COUNT()
### DISTINCT
```sql
SELECT COUNT(DISTINCT name)
FROM animal_ins;
```
## GROUP BY
### HAVING
- HOUR()는 일반조건이므로 (COUNT()와 달리) HAIVNG과 함께 쓸 수 없다.
```sql
SELECT animal_type, COUNT(animal_id)
FROM animal_ins
GROUP BY animal_type
HAVING animal_type in ("Cat", "Dog")
ORDER BY animal_type;
```
## HOUR()
```sql
SELECT HOUR(datetime) AS HOUR, COUNT(*)
FROM animal_outs
GROUP BY HOUR
HAVING HOUR BETWEEN 9 AND 19
ORDER BY HOUR;
```
### AS
## MIN(), MAX()
## IS NULL, IS NOT NULL
```sql
SELECT animal_id
FROM animal_ins
WHERE name IS NULL;
```
## IFNULL()
```sql
SELECT animal_type,
IFNULL(name, "No name"),
sex_upon_intake
FROM animal_ins
ORDER BY animal_id
```
## INNER JOIN, LEFT OUTER JOIN
### ON
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
## UPPER()
```sql
SELECT animal_id, name
FROM animal_ins
WHERE animal_type = "Dog"
AND UPPER(name) LIKE "%EL%"
ORDER BY name;
```
## CASE WHEN THEN ELSE END
```sql
SELECT animal_id,
name,
CASE WHEN (sex_upon_intake LIKE "%Neutered%"
OR sex_upon_intake LIKE "%Spayed%")
THEN "O"
ELSE "X"
END AS "중성화 여부"
FROM animal_ins
ORDER BY animal_id;
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
## LEFT(), RIGHT()
```sql
SELECT LEFT("abcdefghi", 3);
```
## LCASE(), UCASE()
```sql
SELECT LCASE("abcdEFGHI");
```
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
## ABS : 절대값
## CEILING : 올림
## FLOOR : 버림
## ROUND : 반올림
## CONV : 진수 변환
## DEGREES : 육십분법으로 변환
## RADIANS : 호도법으로 변환
## MOD, % : 나머지
## POW : 거듭제곱
## SQRT : 제곱근
## RAND : 0 이상 1 미만의 수 반환
## SIGN : 양수, 0, 음수 판별
## TRUNCATE : 소수점 이하 자리수 지정
## INSERT INTO
### VALUES
```sql
INSERT INTO customers 
(customername, address, city, postalcode, country)
VALUES ("Hekkan Burger", "Gateveien 15", "Sandnes", "4306", "Norway");
```
- If you are adding values for all the columns of the table, you do not need to specify the column names in the SQL query.
## UPDATE
### SET
```sql
UPDATE customers
SET city = "Oslo", country = "Norway"
WHERE customerid = 32;
```
## DELETE FROM
```sql
DELETE FROM sqldb
WHERE fname = "김";
```
## CREATE DATABASE
## DROP DATABASE
```sql
DROP DATABASE IF EXISTS database;
```
## CREATE TABLE
```sql
CREATE TABLE sqldb.usertbl
(id AUTO_INCREMENT PRIMARY KEY,
name VARCHAR(10) NOT NULL,
birthyear INT NOT NULL,
addr CHAR(2) NOT NULL,
mdate DATE);
```
## DROP TABLE
## TRUNCATE TABLE
- Delete the data inside a table, but not the table itself.
## PRIMARY KEY
- The `PRIMARY KEY` constraint uniquely identifies each record in a table. Primary keys must contain UNIQUE values, and cannot contain NULL values. A table can have only ONE primary key; and in the table, this primary key can consist of single or multiple columns (fields).
## FOREIGN KEY
- The `FOREIGN KEY` constraint is used to prevent actions that would destroy links between tables. A `FOREIGN KEY` is a field (or collection of fields) in one table, that refers to the `PRIMARY KEY` in another table. The table with the foreign key is called the child table, and the table with the primary key is called the referenced or parent table.
- The `FOREIGN KEY` constraint prevents invalid data from being inserted into the foreign key column, because it has to be one of the values contained in the parent table.
### REFERENCES
```sql
CREATE TABLE orders
(orderid INT NOT NULL,
ordernumber INT NOT NULL,
personid INT,
PRIMARY KEY (orderid),
FOREIGN KEY (personid) REFERENCES persons(personid));
```
