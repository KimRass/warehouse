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