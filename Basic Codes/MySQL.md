# MySQL
## Command Line Client
### set character-set
```
[client]
default-character-set = utf8

[mysqld]
character-set-client-handshake = FALSE
init_connect="SET collation_connection = utf8_general_ci"
init_connect="SET NAMES utf8"
character-set-server = utf8

[mysql]
default-character-set = utf8

[mysqldump]
default-character-set = utf8
```

## Workbench
### set local_infile
```
set global local_infile = 1;
```
### import csv
```
load data infile 'C:Program Files/MySQL/MySQL Server 8.0/uploads/base_info.csv'
into table masterdata.base_info
fields terminated by ','
lines terminated by '\n'
```
### SQL syntax
```
SELECT host, user, authentication_string FROM mysql.user;
SELECT host, user FROM mysql.db;
```
```
DELETE FROM mysql.user WHERE user="6363"
DELETE FROM mysql.db WHERE user="6363"
FLUSH PRIVILEGES;
```
```
GRANT ALL PRIVILEGES ON masterdata.* TO "6363"@"%";
FLUSH PRIVILEGES;
```
#### CREATE
##### CREATE USER
```
CREATE USER "6363"@localhost IDENTIFIED BY "6363";
```
```
CREATE USER "6363"@"%" IDENTIFIED BY "6363";
```
#### SELECT
##### SELECT + WHERE
```sql
SELECT *
FROM transac, matching, base_info, money_value
WHERE transac.Name = matching.Name AND transac.Area = matching.Area AND matching.Id = base_info.Id AND transac.TransacYM = money_value.TransacYM;
```
##### SELECT + WHERE + BETWEEN
```sql
SELECT name, height
FROM sqlDB.usertbl
WHERE height BETWEEN 180 AND 183;
```
##### SELECT + WHERE + IN
```sql
SELECT name, addr
FROM usertbl
WHERE addr IN ("서울", "경기", "충청");
```
##### SELECT + WHERE + LIKE
```sql
SELECT name, height
FROM usertbl
WHERE name LIKE "김%" OR "_종신";
```
#### SELECT + WHERE + ANY
```sql
SELECT name
FROM usertbl
WHERE height > ANY (SELECT height FROM usertbl WHERE addr="서울");
```
##### SELECT + WHERE + MIN, MAX
```sql
SELECT height
FROM usertbl
WHERE height = (SELECT MIN(height) FROM usertbl);
OR height = (SELECT MAX(height) FROM usertbl);
```
#### SELECT + GROUP BY
```sql
SELECT AVG(amount) AS "평균 구매 수량"
FROM buytbl GROUP BY userid;
```
##### SELECT + ORDER BY
```sql
SELECT name, mdate
FROM usertbl
ORDER BY mdate DESC name ASC;
```
###### SELECT + ORDER BY + COUNT, SUM, AVG, MAX, MIN
```sql
SELECT userid AS "사용자 ID", SUM(price*amount) AS "총 구매액"
FROM buytbl
GROUP BY userid
ORDER BY SUM(price*amount) DESC;
```
##### SELECT + DISTINCT
```sql
SELECT DISTINCT addr FROM usertbl;
```
##### SELECT + LIMIT
```sql
SELECT * FROM usertbl LIMIT 11, 5;
```
##### SELECT + WITH ROLLUP
```sql
SELECT num, groupname, SUM(price*amount) AS "비용"
FROM buytbl
GROUP BY num, groupname
WITH ROLLUP;
```
##### SELECT + HAVING
```sql
SELECT userid AS "사용자 ID", SUM(price*amount) AS "총 구매액"
FROM buytbl
GROUP BY userid
HAVING SUM(price*amount) > 100000;
```
##### SELECT + LAST_INSERT_ID();
- AUTO_INCREMENT 사용 시 사용
##### SELECT + @
```sql
SELECT @변수 이름;
```
#### DROP
##### DROP DATABASE
```sql
DROP DATABASE IF EXISTS database;
```
##### DROP TABLE   
```sql
DROP TABLE tbl;
```
- 내용 및 구조 삭제
--------------------------------------------------------------------------------------------------------------------------------
# set path
```
SETX PATH "%PATH%;C:\Program Files\MySQL\MySQL Server 8.0\bin"
```
### load employees.sql
```
cd C:\Program Files\MySQL\MySQL Server 8.0\bin\employees
mysql -u root -p
source employees.sql;
```
### 시스템 변수 확인
```
SHOW VARIABLES LIKE "max%";
```
### modify max_allowed_packet
- cd %programdata% -> cd MySQL -> cd MySQL Server 8.0 -> notepad my.ini -> max_allowed_packet 수정 -> 재부팅
### export data
* check off "Dump Stored Procedures and Functions", "Dump Events", "Dump Triggers"
* check off "Create Dump in a Single Transaction (self-contained file only)", "Export to Self-Contrained File"
* check off "Include Create Schema"
### import/restore data
* check off "Import from Self-Contained File"
* select "Default Target Schema"
### MySQL Connector/ODBC
* "제어판" -> "관리 도구" -> "ODBC Data Sources (32-bit)"/"ODBC 데이터 원본(64비트)" -> "시스템 DSN" -> "추가(D)..." -> "MySQL ODBC 8.0 Unicode Driver" -> "TCP /IP Server" : 127.0.0.1 -> "Test"
### ASP.NET  Web Forms
* "도구 상자" -> "데이터" -> "SqlDataSource" -> "데이터 소스 구성..." -> "새 연결(C)..." -> "Microsoft ODBC 데이터 소스"
### create and save model
- "File" -> "New Model" -> "Add Diagram" -> "Place a New Table" -> put "Table Name", column info
- "Place a Relationship Using Existing Columns" -> click forein key -> click primary key
- "File" -> "Save Model"
### open model
- "File" -> "Open Model"
- "Database" -> "Foward Engineer..." -> "Stored Connection:" : "Local instance MySQL80" -> check off "Export MySQL Table Objects"
### client connections
- "Client Connections" -> "Kill Connections(s)"
### create statement
- choose a table -> "Send to SQL Editor" -> "Create Statement"
### .txt로 특정 폴더에 저장
- 명령 프롬프트 -> cd %programdata% -> cd MySQL -> cd MySQL Server 8.0 -> notepad my.ini -> secure-file-priv=C:\temp 추가 -> 재부팅
-------------------------------------------------------------------------------------------------------------------

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
### UPDATE
#### UPDATE + SET
```sql
UPDATE usertbl
SET lname="홍"
WHERE fname"길동";
```
```sql
UPDATE buytbl
SET price=price*1.5;
```
### DELETE
#### DELETE FROM
```sql
DELETE FROM sqldb WHERE fname="김";
```
- 연산 부하가 큼
## DDL(Data Definition Language) : CREATE, ALTER, DROP, TRUNCATE, RENAME
### CREATE
#### CREATE SHCEMA | DATABASE
```sql
CREATE SCHEMA sqldb;
CREATE DATABASE sqldb; /* 위 코드와 동일 */
```
#### CREATE TABLE
```sql
CREATE TABLE sqldb.usertbl
(id AUTO_INCREMENT PRIMARY KEY,
name VARCHAR(10) NOT NULL,
birthyear INT NOT NULL,
addr CHAR(2) NOT NULL,
mdate DATE);
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
### TRUNCATE
```sql
TRUNCATE TABLE testtbl;
```
- table의 내용만 삭제
## DCL(Data Control Language) : GRANT, REVOKE
## TCL(Transaction Control Language) : COMMIT, ROLLBACK, SAVEPOINT
## 기타
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
### SET
```sql
SET @nameis="가수 이름 =>";

SELECT @nameis, name FROM usertbl WHERE height>180;
```
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
### CASE
```sql
SELECT CASE 10
    WHEN 1 THEN "일"
    WHEN 5 THEN "오"
    WHEN 10 THEN "십"
    ELSE "모름"
END;
```
### CONCAT_WS
```sql
SELECT CONCAT_WS("/", "2020", "01", "12");
```
### FORMAT
```sql
SELECT FORMAT(123456.123456, 4);
```
- 출력할 소수점 이하 자릿수 지정. 
## 문자열 관련 함수
#### INSERT
```sql
SELECT INSERT("abcdefghi", 3, 2, "####");
```
#### LEFT, RIGHT
```sql
SELECT LEFT("abcdefghi", 3);
```
#### LCASE, UCASE(LOWER, UPPER)
```sql
SELECT LCASE("abcdEFGHI");
```
#### LPAD, RPAD
```sql
SELECT LPAD("이것이", 5, "##")
```
#### LTRIM, RTRIM
```sql
SELECT LTRIM("    좌측공백제거");
```
#### TRIM
```sql
SELECT TRIM("   좌우측공백삭제   ";
SELECT TRIM(BOTH "ㅋ" FROM "ㅋㅋㅋ좌우측문자삭제ㅋㅋㅋ");
SELECT TRIM(LEADING "ㅋ" FROM "ㅋㅋㅋ좌측문자삭제ㅋㅋㅋ");
SELECT TRIM(TRAILING "ㅋ" FROM "ㅋㅋㅋ우측문자삭제ㅋㅋㅋ");
```
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
## 숫자 관련 함수
#### ABS : 절대값
#### CEILING : 올림
#### FLOOR : 버림
#### ROUND : 반올림
#### CONV : 진수 변환
#### DEGREES : 육십분법으로 변환
#### RADIANS : 호도법으로 변환
#### MOD, % : 나머지
#### POW : 거듭제곱
#### SQRT : 제곱근
#### RAND : 0 이상 1 미만의 수 반환
#### SIGN : 양수, 0, 음수 판별
#### TRUNCATE : 소수점 이하 자리수 지정
## 날짜, 시간 관련 함수
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
