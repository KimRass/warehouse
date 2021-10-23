Written by KimRass
# Key
- Source: https://www.guru99.com/dbms-keys.html
## Super Key
- A super key is a group of single or multiple keys which identifies rows in a table. A Super key may have additional attributes that are not needed for unique identification.
- uniqueness but not minimality
## Candidate Key
- Candidate key is a set of attributes that uniquely identify tuples in a table. Candidate Key is a super key with no repeated attributes.
- The Primary key should be selected from the candidate keys. Every table must have at least a single candidate key. A table can have multiple candidate keys but only a single primary key.
- It must contain unique values.
- Candidate key in SQL may have multiple attributes.
- Must not contain null values.
- It should contain minimum fields to ensure uniqueness.
- Uniquely identify each record in a table.
## Primary Key
- Primary key is a column or group of columns in a table that uniquely identify every row in that table.
- Two rows can’t have the same primary key value.
- It must for every row to have a primary key value.
- The primary key field cannot be null.
- The value in a primary key column can never be modified or updated if any foreign key refers to that primary key.
- uniqueness and minimality
## Alternate Key
- A table can have multiple choices for a primary key but only one can be set as the primary key. All the keys which are not primary key are called an Alternate Key.
## Foreign Key
- Foreign key is a column that creates a relationship between two tables. The purpose of foreign keys is to maintain data integrity and allow navigation between two different instances of an entity.
- Foreign Key is used to prevent actions that would destroy links between tables. It is a field (or collection of fields) in one table, that refers to the primary key in another table. The table with the foreign key is called the child table, and the table with the primary key is called the referenced or parent table.
- Foreign key prevents invalid data from being inserted into the foreign key column, because it has to be one of the values contained in the parent table.
## Compound Key
## Composite Key
- Composite key is a combination of two or more columns that uniquely identify rows in a table. The combination of columns guarantees uniqueness, though individual uniqueness is not guaranteed.
## Surrogate Key
- Surrogate key is an artificial key which aims to uniquely identify each record is called a surrogate key. This kind of partial key in DBMS is unique because it is created when you don’t have any natural primary key.
- Surrogate key in DBMS is usually an integer. A surrogate key is a value generated right before the record is inserted into a table.

# View
- Source: https://en.wikipedia.org/wiki/View_(SQL)
- Unlike ordinary base tables in a relational database, a view does not form part of the physical schema: as a result set, it is a virtual table computed or collated dynamically from data in the database when access to that view is requested. Changes applied to the data in a relevant underlying table are reflected in the data shown in subsequent invocations of the view.
- Views can represent a subset of the data contained in a table. Consequently, a view can limit the degree of exposure of the underlying tables to the outer world: a given user may have permission to query the view, while denied access to the rest of the base table.
- Views can join and simplify multiple tables into a single virtual table.
- Views can act as aggregated tables, where the database engine aggregates data (sum, average, etc.) and presents the calculated results as part of the data.
- Views can hide the complexity of data. For example, a view could appear as Sales2000 or Sales2001, transparently partitioning the actual underlying table.
- Views take very little space to store; the database contains only the definition of a view, not a copy of all the data that it presents.

# Data Modeling
- Source: https://www.ibm.com/cloud/learn/data-modeling
- Data Requirements Collection -> Conceptual Data Models -> Logical Data Models -> Physical Data Models
## Data Requirements
- Source: https://www.freetutes.com/systemanalysis/sa7-data-modeling-data-requirements.html
- Here the database designer interviews database users. By this process they are able to understand their data requirements. Results of this process are clearly documented.
기능
인터페이스
보안
성능
## Conceptual Data Modeling
- Conceptual models offer a big-picture view of what the system will contain, how it will be organized, and which business rules are involved.
- They are usually created as part of the process of gathering initial project requirements.
## Logical Data Modeling
- Logical models are less abstract and provide greater detail about the concepts and relationships in the domain under consideration. One of several formal data modeling notation systems is followed. These indicate data attributes, such as data types and their corresponding lengths, and show the relationships among entities. Logical data models don’t specify any technical system requirements.
### Database Normalization
- Source: https://en.wikipedia.org/wiki/Database_normalization
- Database normalization is the process of structuring a database, usually a relational database, in accordance with a series of so-called normal forms in order to reduce data redundancy and improve data integrity.
- Objectives:
	- To free the collection of relations from undesirable insertion, update and deletion dependencies.
	- To reduce the need for restructuring the collection of relations, as new types of data are introduced, and thus increase the life span of application programs.
- When an attempt is made to modify (update, insert into, or delete from) a relation, the following undesirable side-effects may arise in relations that have not been sufficiently normalized:
	- Update anomaly: For example, a change of address for a particular employee may need to be applied to multiple records (one for each skill). If the update is only partially successful – the employee's address is updated on some records but not others – then the relation is left in an inconsistent state. Specifically, the relation provides conflicting answers to the question of what this particular employee's address is.
	- Insertion anomaly: There are circumstances in which certain facts cannot be recorded at all. For example, the details of any faculty member who teaches at least one course can be recorded, but a newly hired faculty member who has not yet been assigned to teach any courses cannot be recorded, except by setting the Course Code to null.
	- Deletion anomaly: Under certain circumstances, deletion of data representing certain facts necessitates deletion of data representing completely different facts. For example, if a faculty member temporarily ceases to be assigned to any courses, the last of the records on which that faculty member appears must be deleted, effectively also deleting the faculty member, unless the Course Code field is set to null.
- Normalization is a database design technique, which is used to design a relational database table up to higher normal form.[9] The process is progressive, and a higher level of database normalization cannot be achieved unless the previous levels have been satisfied.
#### Satisfying 1NF(First Normal Form)
- Only atomic columns.
- To satisfy first normal form, each column of a table must have a single value. Columns which contain sets of values or nested records are not allowed.
#### Satisfying 2NF(Second Normal Form)
- No partial dependencies.
- To conform to 2NF and remove duplicities, every non candidate key attribute must depend on the whole candidate key, not just part of it.
- For example, all of the attributes that are not part of the candidate key depend on `Title`, but only `Price` also depends on `Format`. To normalize this table, make `Title` a (simple) candidate key (the primary key) so that every non candidate-key attribute depends on the whole candidate key, and remove `Price` into a separate table so that its dependency on `Format` can be preserved.
#### Satisfying 3NF(Third Normal Form)
- No transitive dependencies.
- Primary key가 아닌 Column에 Dependent하는 Columns를 제거합니다.
#### 4NF
- 서로 독립적인 관계는 테이블을 쪼갠다.
#### 5NF
## Physical Data Modeling
- Physical models provide a schema for how the data will be physically stored within a database. As such, they’re the least abstract of all.
### Denormalization
- Source: https://www.geeksforgeeks.org/denormalization-in-databases/
- Denormalization is a database optimization technique in which we add redundant data to one or more tables. This can help us avoid costly joins in a relational database. Note that denormalization does not mean not doing normalization. It is an optimization technique that is applied after doing normalization.
- In a traditional normalized database, we store data in separate logical tables and attempt to minimize redundant data. We may strive to have only one copy of each piece of data in database. For example, in a normalized database, we might have a Courses table and a Teachers table. Each entry in Courses would store the teacherID for a Course but not the teacherName. When we need to retrieve a list of all Courses with the Teacher’s name, we would do a join between these two tables. In some ways, this is great; if a teacher changes his or her name, we only have to update the name in one place. The drawback is that if tables are large, we may spend an unnecessarily long time doing joins on tables. Denormalization, then, strikes a different compromise. Under denormalization, we decide that we’re okay with some redundancy and some extra effort to update the database in order to get the efficiency advantages of fewer joins.
- Pros:
	- Retrieving data is faster since we do fewer joins
	- Queries to retrieve can be simpler(and therefore less likely to have bugs), since we need to look at fewer tables.
- Cons:
	- Updates and inserts are more expensive.
	- Denormalization can make update and insert code harder to write.
	- Data may be inconsistent . Which is the “correct” value for a piece of data?
	- Data redundancy necessitates more storage.

# Wireframe & Storyboard
## Wireframe
- Source: https://balsamiq.com/learn/articles/what-are-wireframes/
- ![image]( https://balsamiq.com/assets/learn/articles/all-controls-split-r.png)
- A wireframe is a schematic or blueprint that is useful for helping you, your programmers and designers think and communicate about the structure of the software or website you're building.
- Doing this work now, before any code is written and before the visual design is finalized, will save you lots of time and painful adjustment work later.
- Wireframes make it clear that no code has been written yet. If your customer or stakeholder received some screens that looked like screenshots of the final app, instead of a wireframe, they might assume that all the code behind those screenshots had already been written. This is most often not the case. Wireframes don't have this danger.
- Source: http://www.differencebetween.net/technology/difference-between-wireframe-and-storyboard/
- Wireframe is a page schematic, a sketch of your website before any kind of development or design element goes into it. It is basically a visual representation of the layout of your website without the fancy elements such as colors, fonts, shading, or just any other design element that makes your website visually appealing and interactive.
- A wireframe is a linear representation of a website or web page structure, kind of a mock up screen of what the actual thing will look like.
## Storyboard
- 요청하는 고객이 작성해야만 하며 기획자가 대신 작성해주지 못함.
- 버전 이력 관리가 필요.
- 사용자 화면과 관리자 화면이 모두 있어야 함.
- Source: http://www.differencebetween.net/technology/difference-between-wireframe-and-storyboard/
- Storyboard is kind of advanced wireframing created on a piece of paper using a pencil or using a graphics program on a computer.
- A storyboard is a more detailed representation, a high level outline including descriptions of what happens as user goes further inside the application. Storyboards are more dynamic in structure in terms of grouping and ordering.

# Customer Segmentation(Market Segmentation)
- Source: https://openviewpartners.com/blog/customer-segmentation/#.YU6EJZpByHs
- the division of potential customers in a given market into discrete groups.

# ERD(Entity Relationship Diagram)
## Entity
## Relationship
- One to One or One to Many or Many to Many
### Identifying Relationship
- 외래키를 기본키로 사용하는 관계를 식별 관계.
- B테이블은 A테이블에 종속적이 되어서 A의 값이 없으면 B의 값은 무의미해짐.
- 실선
### Non-Identifying Relationship
- When the primary key of the parent must not become primary key of the child.
- A의 값이 없더라도 B의 값은 독자적으로 의미를 가짐.
- 부모 테이블의 PK가 없더라도 유일하게 레코드 구분 가능할 때.
- 한 사람이 한 뉴스에 여러 개의 댓글을 달 수 있으므로 회원번호만으로는 특정 댓글 식별 불가. 따라서 비식별 관계.
- 점선

# Partition
- Source: https://en.wikipedia.org/wiki/Partition_(database)
## Partitioning Methods
### Horizontal Partitioning
- Involves putting different rows into different tables. For example, customers with ZIP codes less than 50000 are stored in CustomersEast, while customers with ZIP codes greater than or equal to 50000 are stored in CustomersWest. The two partition tables are then CustomersEast and CustomersWest, while a view with a union might be created over both of them to provide a complete view of all customers.
### Vertical Partitioning
- Involves creating tables with fewer columns and using additional tables to store the remaining columns. Generally, this practice is known as normalization. However, vertical partitioning extends further and partitions columns even when already normalized.
- Distinct physical machines might be used to realize vertical partitioning: Storing infrequently used or very wide columns, taking up a significant amount of memory, on a different machine, for example, is a method of vertical partitioning.
- A common form of vertical partitioning is to split static data from dynamic data, since the former is faster to access than the latter, particularly for a table where the dynamic data is not used as often as the static.
- Creating a view across the two newly created tables restores the original table with a performance penalty, but accessing the static data alone will show higher performance.
## Partitioning Criteria
- Range partitioning: selects a partition by determining if the partitioning key is within a certain range.
- List partitioning: a partition is assigned a list of values. If the partitioning key has one of these values, the partition is chosen.
- Composite partitioning
- Round-robin partitioning
- Hash partitioning

# Static Data & Dynamic Data
- Source: https://blog.zoominfo.com/dynamic-data/
## Static Data
- Refers to a fixed data set—or, data that remains the same after it’s collected.
- Source: https://en.wikipedia.org/wiki/Dynamic_data
- Data that is infrequently accessed and not likely to be modified.
## Dynamic Data
- Continually changes after it’s recorded in order to maintain its integrity.
- Source: https://en.wikipedia.org/wiki/Dynamic_data
- Dynamic data may be updated at any time, with periods of inactivity in between.

1. 회원 테이블에서 현재 위치를 저장하는 컬럼을 만든다.
2. 서버B의 거래처 테이블에 CRUD가 발생할 때마다 서버A에도 동기화
3. 월 단위로 Partitioning한다.
4. 조인을 여러 번 해야 함 -> 고객-배송을 관계를 맺음.
5. 여러 통계를 계산하여 담은 테이블을 별도로 만든다.
6. 결제금액을 미리 계산해서 주문 테이블의 컬럼으로 만든다.
7. 내용이 용량 너무 커서 16K 자주 조회해야 함 -> 포스트 내용만을 담은 1:1 테이블을 하나 따로 만든다
8. 자주 조회되는 컬럼, 아닌 컬럼으로 테이블 나눈다

# Case Types
## Camel Case
- e.g., camelCaseVar.
## Pascal Case
- e.g., CamelCaseVar.
## Snake Case
- e.g., camel_case_var.

# CRUD Matrix
# Transaction
- Source: https://en.wikipedia.org/wiki/Database_transaction
- A database transaction symbolizes a unit of work performed within a database management system (or similar system) against a database, and treated in a coherent and reliable way independent of other transactions.
- In a database management system, a transaction is a single unit of logic or work, sometimes made up of multiple operations. Any logical calculation done in a consistent mode in a database is known as a transaction. One example is a transfer from one bank account to another: the complete transaction requires subtracting the amount to be transferred from one account and adding that same amount to the other.
## ACID(Atomicity, Consistency, Isolation, Durability)
- Source: https://en.wikipedia.org/wiki/ACID
- ACID is a set of properties of database transactions intended to guarantee data validity despite errors, power failures, and other mishaps.
- In the context of databases, a sequence of database operations that satisfies the ACID properties (which can be perceived as a single logical operation on the data) is called a transaction. For example, a transfer of funds from one bank account to another, even involving multiple changes such as debiting one account and crediting another, is a single transaction.
- Atomicity
	- Transactions are often composed of multiple statements. Atomicity guarantees that each transaction is treated as a single "unit", which either succeeds completely, or fails completely: if any of the statements constituting a transaction fails to complete, the entire transaction fails and the database is left unchanged. An atomic system must guarantee atomicity in each and every situation, including power failures, errors and crashes. A guarantee of atomicity prevents updates to the database occurring only partially, which can cause greater problems than rejecting the whole series outright. As a consequence, the transaction cannot be observed to be in progress by another database client. At one moment in time, it has not yet happened, and at the next it has already occurred in whole (or nothing happened if the transaction was cancelled in progress).
- Consistency(Correctness)
	- Consistency ensures that a transaction can only bring the database from one valid state to another, maintaining database invariants: any data written to the database must be valid according to all defined rules, including constraints, cascades, triggers, and any combination thereof. This prevents database corruption by an illegal transaction, but does not guarantee that a transaction is correct.
- Isolation
	- Transactions are often executed concurrently (e.g., multiple transactions reading and writing to a table at the same time). Isolation ensures that concurrent execution of transactions leaves the database in the same state that would have been obtained if the transactions were executed sequentially.
- Durability
	- Durability guarantees that once a transaction has been committed, it will remain committed even in the case of a system failure (e.g., power outage or crash). This usually means that completed transactions (or their effects) are recorded in non-volatile memory.
	
# Database Index
- Source: https://en.wikipedia.org/wiki/Database_index
- A database index is a data structure that improves the speed of data retrieval operations on a database table at the cost of additional writes and storage space to maintain the index data structure. Indexes are used to quickly locate data without having to search every row in a database table every time a database table is accessed. Indexes can be created using one or more columns of a database table, providing the basis for both rapid random lookups and efficient access of ordered records.
- An index is a copy of selected columns of data, from a table, that is designed to enable very efficient search. An index normally includes a "key" or direct link to the original row of data from which it was copied, to allow the complete row to be retrieved efficiently. Some databases extend the power of indexing by letting developers create indexes on column values that have been transformed by functions or expressions. For example, an index could be created on upper(last_name), which would only store the upper-case versions of the last_name field in the index. Another option sometimes supported is the use of partial indices, where index entries are created only for those records that satisfy some conditional expression. A further aspect of flexibility is to permit indexing on user-defined functions, as well as expressions formed from an assortment of built-in functions.
- Source: https://www.guru99.com/clustered-vs-non-clustered-index.html
- Clustered index is a type of index that sorts the data rows in the table on their key values whereas the non-clustered index stores the data at one location and indices at another location.
- Clustered index stores data pages in the leaf nodes of the index while Non-clustered index method never stores data pages in the leaf nodes of the index.
- Clustered index doesn’t require additional disk space whereas the non-clustered index requires additional disk space.
- Clustered index offers faster data accessing, on the other hand, non-clustered index is slower.
## Clustered Index
- In the Database, there is only one clustered index per table.
- A clustered index defines the order in which data is stored in the table which can be sorted in only one way.
- In an RDBMS, usually, the primary key allows you to create a clustered index based on that specific column.
인덱스 자체의 리프 페이지가 곧 데이터이다. 즉 테이블 자체가 인덱스이다. (따로 인덱스 페이지를 만들지 않는다.)
데이터 입력, 수정, 삭제 시 항상 정렬 상태를 유지한다.
비 클러스형 인덱스보다 검색 속도는 더 빠르다. 하지만 데이터의 입력. 수정, 삭제는 느리다.
## Non-Clusterd Index
- A Non-clustered index stores the data at one location and indices at another location. The index contains pointers to the location of that data.
- A single table can have many non-clustered indexes as an index in the non-clustered index is stored in different places.
- For example, a book can have more than one index, one at the beginning which displays the contents of a book unit wise while the second index shows the index of terms in alphabetical order.

# Cardinality
- Source: https://orangematter.solarwinds.com/2021/10/01/what-is-cardinality-in-a-database/
- The number of distinct values in a table column relative to the number of rows in the table.

# Database Trigger
- Source: https://en.wikipedia.org/wiki/Database_trigger
- A database trigger is procedural code that is automatically executed in response to certain events on a particular table or view in a database. The trigger is mostly used for maintaining the integrity of the information on the database. For example, when a new record (representing a new worker) is added to the employees table, new records should also be created in the tables of the taxes, vacations and salaries. Triggers can also be used to log historical data, for example to keep track of employees' previous salaries.

# Super Type

# 재귀적 구조
- 조직도, 쇼핑몰 카테고리, 게시판 댓글

# BOM

# 표준화

# Domain

# 포워드
- Check 테이블, 기본키, 외래키, 코멘트

HeidiSQL, DBeaver, DataGrip
utf8 -> utf8_general_ci

CREATE Index
ADD CONSTRAINT, DROP CONSTRAINT
INFORMATION_SCHEMA