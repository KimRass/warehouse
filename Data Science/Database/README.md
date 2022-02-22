Written by KimRass
# Constraints
- Source: https://www.tutorialspoint.com/sql/sql-constraints.htm
- Constraints are the rules enforced on the data columns of a table. These are used to limit the type of data that can go into a table. This ensures the accuracy and reliability of the data in the database.
- Constraints could be either on a column level or a table level. The column level constraints are applied only to one column, whereas the table level constraints are applied to the whole table.

# Key
- Source: https://www.guru99.com/dbms-keys.html
## Super Key
- A super key is a group of single or multiple keys which identifies rows in a table. A Super key may have additional attributes that are not needed for unique identification.
- uniqueness but not minimality
## Candidate Key(= 후보 식별자)
- Candidate key is a set of attributes that uniquely identify tuples in a table. Candidate Key is a super key with no repeated attributes.
- The Primary key should be selected from the candidate keys. Every table must have at least a single candidate key. A table can have multiple candidate keys but only a single primary key.
- It must contain unique values.
- Candidate key in SQL may have multiple attributes.
- Must not contain NULL values.
- It should contain minimum fields to ensure uniqueness.
- Uniquely identify each record in a table.
## Primary Key(= 주식별자)
- ***A primary key is a specific choice of a minimal set of attributes(= columns) that uniquely specify a tuple(= row) in a table.(Uniqueness and minimality)***
- It must for every row to have a primary key value.
- ***The value in a primary key column can never be modified or updated if any foreign key refers to that primary key.***
## Alternate Key(= 보조 식별자)
- A table can have multiple choices for a primary key but only one can be set as the primary key. All the keys which are not primary key are called an Alternate Key.
## Foreign Key(= 외부 식별자 <-> 내부 식별자)
- Foreign key is a column that creates a relationship between two tables. The purpose of foreign keys is to maintain data integrity and allow navigation between two different instances of an entity.
- Foreign Key is used to prevent actions that would destroy links between tables. It is a field (or collection of fields) in one table, that refers to the primary key in another table. The table with the foreign key is called the child table, and the table with the primary key is called the referenced or parent table.
- Foreign key prevents invalid data from being inserted into the foreign key column, because it has to be one of the values contained in the parent table.
## Compound Key
## Composite Key(= 복합 식별자 <-> 복합 식별자)
- Composite key is a combination of two or more columns that uniquely identify rows in a table. The combination of columns guarantees uniqueness, though individual uniqueness is not guaranteed.
## Surrogate Key(= 인조 식별자)
- Surrogate key is an artificial key which aims to uniquely identify each record is called a surrogate key. This kind of partial key in DBMS is unique because it is created when you don’t have any natural primary key.
- Surrogate key in DBMS is usually an integer. A surrogate key is a value generated right before the record is inserted into a table.

# Database Index
- Source: https://en.wikipedia.org/wiki/Database_index
- A database index is a data structure that improves the speed of data retrieval operations on a database table at the cost of additional writes and storage space to maintain the index data structure. Indexes are used to quickly locate data without having to search every row in a database table every time a database table is accessed. Indexes can be created using one or more columns of a database table, providing the basis for both rapid random lookups and efficient access of ordered records.
- An index is a copy of selected columns of data, from a table, that is designed to enable very efficient search. An index normally includes a "key" or direct link to the original row of data from which it was copied, to allow the complete row to be retrieved efficiently. Some databases extend the power of indexing by letting developers create indexes on column values that have been transformed by functions or expressions. For example, an index could be created on upper(last_name), which would only store the upper-case versions of the last_name field in the index. Another option sometimes supported is the use of partial indices, where index entries are created only for those records that satisfy some conditional expression. A further aspect of flexibility is to permit indexing on user-defined functions, as well as expressions formed from an assortment of built-in functions.
- Source: https://www.guru99.com/clustered-vs-non-clustered-index.html
- Clustered index is a type of index that sorts the data rows in the table on their key values whereas the non-clustered index stores the data at one location and indices at another location.
- Clustered index stores data pages in the leaf nodes of the index while Non-clustered index method never stores data pages in the leaf nodes of the index.
- Clustered index doesn’t require additional disk space whereas the non-clustered index requires additional disk space.
- Clustered index offers faster data accessing, on the other hand, non-clustered index is slower.
- ***만들 수 있는 Index의 수는 제한이 없으나 너무 많이 만들면 오히려 성능 저하가 발생합니다.***
- 순차 인덱스, 결합 인덱스, 해시 인덱스, 비트맵, 클러스터
## Clustered Index
- In the database, there is only one clustered index per table.
- A clustered index defines the order in which data is stored in the table which can be sorted in only one way.
- In an RDBMS, usually, the primary key allows you to create a clustered index based on that specific column.
인덱스 자체의 리프 페이지가 곧 데이터이다. 즉 테이블 자체가 인덱스이다. (따로 인덱스 페이지를 만들지 않는다.)
데이터 입력, 수정, 삭제 시 항상 정렬 상태를 유지한다.
비 클러스형 인덱스보다 검색 속도는 더 빠르다. 하지만 데이터의 입력. 수정, 삭제는 느리다.
## Non-Clusterd Index
- A Non-clustered index stores the data at one location and indices at another location. The index contains pointers to the location of that data.
- A single table can have many non-clustered indexes as an index in the non-clustered index is stored in different places.
- For example, a book can have more than one index, one at the beginning which displays the contents of a book unit wise while the second index shows the index of terms in alphabetical order.

# View
- Source: https://en.wikipedia.org/wiki/View_(SQL)
- Unlike ordinary base tables in a relational database, ***a view does not form part of the physical schema: as a result set, it is a virtual table computed or collated dynamically from data in the database when access to that view is requested.(물리성)*** Changes applied to the data in a relevant underlying table are reflected in the data shown in subsequent invocations of the view.
- Views can represent a subset of the data contained in a table. Consequently, a view can limit the degree of exposure of the underlying tables to the outer world: ***a given user may have permission to query the view, while denied access to the rest of the base table.(보안성)***
- ***Views can join and simplify multiple tables into a single virtual table.(편리성)***
- Views can act as aggregated tables, where the database engine aggregates data (sum, average, etc.) and presents the calculated results as part of the data.
- Views can hide the complexity of data. For example, a view could appear as Sales2000 or Sales2001, transparently partitioning the actual underlying table.
- Views take very little space to store; the database contains only the definition of a view, not a copy of all the data that it presents.
- ***독립성: 테이블 구조가 변경되어도 View를 사용하는 응용 프로그램은 변경하지 않아도 된다.***

# Data Modeling
- ***추상화(Abstraction): 현실 세계를 간략하게 추상화하여 표현합니다.***
- ***단순화(Simplicity): 누구나 쉽게 이해할 수 있도록 표현합니다.***
- ***명확성(Clarity): 명확하게 의미가 해석되어야 합니다.***
- Source: https://www.ibm.com/cloud/learn/data-modeling
- Data Requirements Collecting -> Conceptual Data Modeling -> Logical Data Modeling -> Physical Data Modeling
## Data Requirements Collecting
- Source: https://www.freetutes.com/systemanalysis/sa7-data-modeling-data-requirements.html
- Here the database designer interviews database users. By this process they are able to understand their data requirements. Results of this process are clearly documented.
## Conceptual Data Modeling
- Conceptual models offer a big-picture view of what the system will contain, how it will be organized, and which business rules are involved.
- They are usually created as part of the process of gathering initial project requirements.
- 사용자 관점에서 Data requirements를 식별합니다.
## Logical Data Modeling
- Logical models are less abstract and provide greater detail about the concepts and relationships in the domain under consideration. One of several formal data modeling notation systems is followed. ***These indicate data attributes, such as data types and their corresponding lengths, and show the relationships among entities.*** Logical data models don’t specify any technical system requirements.
### ***Database Normalization***
- Source: https://en.wikipedia.org/wiki/Database_normalization
- Database normalization is the process of structuring a database, usually a relational database, in accordance with a series of so-called normal forms in order to reduce data redundancy and improve data integrity.
- Objectives:
	- ***To free the collection of relations from undesirable update, insertion and deletion dependencies.***
	- ***To reduce the need for restructuring the collection of relations, as new types of data are introduced, and thus increase the life span of application programs.***
- When an attempt is made to modify (update, insert into, or delete from) a relation, the following undesirable side-effects may arise in relations that have not been sufficiently normalized:
	- Update anomaly: For example, a change of address for a particular employee may need to be applied to multiple records. If the update is only partially successful – the employee's address is updated on some records but not others – then the relation is left in an inconsistent state. Specifically, the relation provides conflicting answers to the question of what this particular employee's address is.
	- Insertion anomaly: There are circumstances in which certain facts cannot be recorded at all. For example, the details of any faculty member who teaches at least one course can be recorded, but a newly hired faculty member who has not yet been assigned to teach any courses cannot be recorded, except by setting the Course Code to NULL.
	- Deletion anomaly: Under certain circumstances, deletion of data representing certain facts necessitates deletion of data representing completely different facts. For example, if a faculty member temporarily ceases to be assigned to any courses, the last of the records on which that faculty member appears must be deleted, effectively also deleting the faculty member, unless the course code field is set to NULL.
- Normalization is a database design technique, which is used to design a relational database table up to higher normal form.[9] The process is progressive, and a higher level of database normalization cannot be achieved unless the previous levels have been satisfied.
#### Satisfying 1NF(First Normal Form)
- Only atomic columns.
- ***To satisfy first normal form, each column of a table must have a single value.*** Columns which contain sets of values or nested records are not allowed.
#### Satisfying 2NF(Second Normal Form)
- No partial dependencies.
- To conform to 2NF and remove duplicities, ***every non candidate key attribute must depend on the whole candidate key, not just part of it.***
- For example, all of the attributes that are not part of the candidate key depend on `Title`, but only `Price` also depends on `Format`. To normalize this table, make `Title` a (simple) candidate key (the primary key) so that every non candidate-key attribute depends on the whole candidate key, and remove `Price` into a separate table so that its dependency on `Format` can be preserved.
#### Satisfying 3NF(Third Normal Form)
- No transitive dependencies.
- For example, `Author Nationality` is dependent on `Author`, which is dependent on `Title`. `Genre Name` is dependent on `Genre ID`, which is dependent on `Title`. 
## Physical Data Modeling
- ***Physical models provide a schema for how the data will be physically stored within a database***. As such, they’re the least abstract of all.

# Denormalization
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
## Partition
- Source: https://en.wikipedia.org/wiki/Partition_(database)
### Horizontal Partitioning
- Involves putting different rows into different tables. For example, customers with ZIP codes less than 50000 are stored in CustomersEast, while customers with ZIP codes greater than or equal to 50000 are stored in CustomersWest. The two partition tables are then CustomersEast and CustomersWest, while a view with a union might be created over both of them to provide a complete view of all customers.
### Vertical Partitioning
- Involves creating tables with fewer columns and using additional tables to store the remaining columns. Generally, this practice is known as normalization. However, vertical partitioning extends further and partitions columns even when already normalized.
- Distinct physical machines might be used to realize vertical partitioning: Storing infrequently used or very wide columns, taking up a significant amount of memory, on a different machine, for example, is a method of vertical partitioning.
- A common form of vertical partitioning is to split static data from dynamic data, since the former is faster to access than the latter, particularly for a table where the dynamic data is not used as often as the static.
- Creating a view across the two newly created tables restores the original table with a performance penalty, but accessing the static data alone will show higher performance.
### Partitioning Criteria
- Range partitioning: selects a partition by determining if the partitioning key is within a certain range.
- List partitioning: a partition is assigned a list of values. If the partitioning key has one of these values, the partition is chosen.
- Round-robin partitioning
- Hash partitioning
- Composite partitioning
### Partition Index
#### Global Partition Index
- A global partitioned index is an index on a partitioned or non-partitioned table that is partitioned independently, i.e. using a different partitioning key from the table.
- Table partition key와 Index partition key가 서로 다릅니다.
- ***Partition key에 대해 생성한 Index.***
##### Global Prefixed Index
##### Global Non-Prefixed Index
#### Local Partition Index
- A local index on a partitioned table is created where the index is partitioned in exactly the same manner as the underlying partitioned table. That is, the local index inherits the partitioning method of the table. This is known as equi-partitioning.
- The table and local index are either partitioned in exactly the same manner, or have the same partition key because the local indexes are automatically maintained, can offer higher availability.
- Table partition key와 Index partition key가 서로 같습니다.
##### Local Prefixed Index
- 인덱스 첫번째 컬럼이 인덱스 파티션 키와 같습니다.
##### Local Non-Prefixed Index
- 인덱스 첫번째 컬럼이 인덱스 파티션 키와 다릅니다.
### Non-Partition Index
## Adding Columns
### Adding Redundant Columns
- Source: https://www.slideshare.net/mnoia/denormalization-56426459
- You can add redundant columns to eliminate frequent joins.
### Adding Derived Columns
- Adding derived columns can help eliminate joins and reduce the time needed to produce aggregate values.
## Combining Tables
- If most users need to see the full set of joined data from two tables, collapsing the two tables into one can improve performance by eliminating the join.

# ERD(Entity Relationship Diagram)
- Mandatory, Optional.
- IE(Information Engineering) Notation.
## Entity
- 1개의 Entity는 2개 이상의 Instance의 집합이어야 합니다.
- 1개의 Entity는 2개 이상의 Attribute를 갖습니다.
- 1개의 Attribute는 1개의 Value를 갖습니다.
### Associative Entity(= Intersection Entity, 교차 엔티티)
- ***A relational database requires the implementation of a base relation (or base table) to resolve many-to-many relationships.*** A base relation representing this kind of entity is called, informally, an associative table.
## Attribute
### Stored Attribute(= 기본 속성)
- Stored attribute is an attribute which are physically stored in the database.
### Derived Attribute(= 파생 속성)
- ***A derived attribute is an attribute whose values are calculated from other attributes.***
### 설계 속성
## Relationship
### Degree of a Relationship(= 관게 차수)
- ***The number of participating entities in a relationship***
### Relationship Cardinality
- Determines the number of entities on one side of the relationship that can be joined to a single entity on the other side.
- ***One of the followings; One to One, One to Many or Many to Many.***
### Relationship Optionality
- Specifies if entities on one side must be joined to an entity on the other side.
### Identifying Relationship
- The primary key of the parent would become a subset of the primary key of the child.
- Reprsented by a solid line.
### Non-Identifying Relationship
- When the primary key of the parent must not become primary key of the child.
- Reprsented by a dotted line.
### Recursive Relationship
- Source: https://bookshelf.erwin.com/bookshelf/public_html/2020R1/Content/User%20Guides/erwin%20Help/Recursive_Relationships.html
- A recursive relationship is a non-identifying relationship between two entities or tables that represents the fact that, for example, one company can own another company. In this type of relationship, the parent entity or table and the child entity or table are the same.
- Types of recursive relationships:
	- Hierarchical Recursive Relationship: A parent entity or table can have any number of children, but a child can only have one parent.
	- Network Recursive Relationship: A parent entity or table can have any number of children, and a child can have any number of parents. An entity or table has a many-to-many relationship with itself. When a many-to-many network recursion problem exists, you can clarify the situation by creating an intermediate entity or table and converting the many-to-many relationship into two one-to-many relationships.
- Source: https://sqldatabasetutorials.com/sql-db/recursive-relationship/
- The ERD convention to show a recursive structure is drawn as a loop, also known as a "pig's ear".

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

# Case Types
## Camel Case
- e.g., camelCaseVar.
## Pascal Case
- e.g., CamelCaseVar.
## Snake Case
- e.g., camel_case_var.

# CRUD Matrix
- https://en.wikipedia.org/wiki/Create,_read,_update_and_delete
- The four basic operations of persistent storage.
- Create, Read, Update, Delete.
# Transaction
- Source: https://en.wikipedia.org/wiki/Database_transaction
- A database transaction symbolizes a unit of work performed within a database management system (or similar system) against a database, and treated in a coherent and reliable way independent of other transactions.
- In a database management system, a transaction is a single unit of logic or work, sometimes made up of multiple operations. Any logical calculation done in a consistent mode in a database is known as a transaction. One example is a transfer from one bank account to another: the complete transaction requires subtracting the amount to be transferred from one account and adding that same amount to the other.
## ACID(Atomicity, Consistency, Isolation, Durability)
- Source: https://en.wikipedia.org/wiki/ACID
- ACID is a set of properties of database transactions intended to guarantee data validity despite errors, power failures, and other mishaps.
- In the context of databases, a sequence of database operations that satisfies the ACID properties (which can be perceived as a single logical operation on the data) is called a transaction. For example, a transfer of funds from one bank account to another, even involving multiple changes such as debiting one account and crediting another, is a single transaction.
- Atomicity(= 원자성)
	- Transactions are often composed of multiple statements. ***Atomicity guarantees that each transaction is treated as a single "unit", which either succeeds completely, or fails completely: if any of the statements constituting a transaction fails to complete, the entire transaction fails and the database is left unchanged.*** An atomic system must guarantee atomicity in each and every situation, including power failures, errors and crashes. A guarantee of atomicity prevents updates to the database occurring only partially, which can cause greater problems than rejecting the whole series outright. As a consequence, the transaction cannot be observed to be in progress by another database client. At one moment in time, it has not yet happened, and at the next it has already occurred in whole (or nothing happened if the transaction was cancelled in progress).
- Consistency(Correctness)(= 일관성)
	- Source: https://www.bmc.com/blogs/acid-atomic-consistent-isolated-durable/
	- ***Consistency refers to maintaining data integrity constraints. A consistent transaction will not violate integrity constraints placed on the data by the database rules.*** Enforcing consistency ensures that if a database enters into an illegal state (if a violation of data integrity constraints occurs) the process will be aborted and changes rolled back to their previous, legal state.
- Isolation(= 고립성, 격리성)
	- Transactions are often executed concurrently (e.g., multiple transactions reading and writing to a table at the same time). ***Isolation ensures that concurrent execution of transactions leaves the database in the same state that would have been obtained if the transactions were executed sequentially.***
- Durability(= 영속성, 지속성)
- Durability guarantees that once a transaction has been committed, it will remain committed even in the case of a system failure (e.g., power outage or crash). ***This usually means that completed transactions (or their effects) are recorded in non-volatile memory.***

# Database Trigger
- Source: https://en.wikipedia.org/wiki/Database_trigger
- A database trigger is procedural code that is automatically executed in response to certain events on a particular table or view in a database. The trigger is mostly used for maintaining the integrity of the information on the database. For example, when a new record (representing a new worker) is added to the employees table, new records should also be created in the tables of the taxes, vacations and salaries. Triggers can also be used to log historical data, for example to keep track of employees' previous salaries.
- 생성하면 소스코드와 실행코드가 생성됩니다.
- 생성 후 자동으로 실행됩니다.
- `COMMIT`, `ROLLBACK`이 불가합니다.

# Procedure
- 생성하면 소스코드와 실행코드가 생성됩니다.
- `EXECUTE` 명령어로 실행합니다.
- `COMMIT`, `ROLLBACK`이 가능합니다.

# Supertype & Subtype
- Source: https://sqldatabasetutorials.com/sql-db/supertypes-and-subtypes/
- Often some instances of an entity have attributes and/or relationships that other instances do not have. Imagine a business which needs to track payments from customers. Customers can pay by cash, by check, or by credit card. All payments have some common attributes: payment date, payment amount, and so on. But only credit cards would have a "card number" attribute. And for credit card and check payments, we may need to know which customer made the payment, while this is not needed for cash payments. Should we create a single `PAYMENT` entity or three separate entities `CASH`, `CHECK`, and `CREDIT CARD`?
- Sometimes it makes sense to subdivide an entity into subtypes. This may be the case when a group of instances has special properties, such as attributes or relationships that exist only for that group. In this case, the entity is called a supertype and each group is called a subtype.
- A subtype:
	- Inherits all attributes of the supertype.
	- Inherits all relationships of the supertype.
	- Usually has its own attributes or relationships.
	- Is drawn within the supertype.
	- Never exists alone.
	- May have subtypes of its own.
- Subtype rules:
	- Exhaustive: Every instance of the supertype is also an instance of one of the subtypes. All subtypes are listed without omission.
	- Mutually Exclusive: Each instance of a supertype is an instance of only one possible subtype.
- At the conceptual modeling stage, it is good practice to include an `OTHER` subtype to make sure that your subtypes are exhaustive — that you are handling every instance of the supertype.
## One to One Types
- ***Supertype과 Subtype를 개별 Table로 도출합니다.***
- ***Table의 수가 많아서 Join이 많이 발생하고 관리가 어렵습니다.***
## Plus Types
- ***Supertype과 Subtype를 개별 Table로 도출합니다.***
- ***Table의 수가 많아서 Join이 많이 발생하고 관리가 어렵습니다.***
## Single Type
- ***Supertype과 Subtype를 하나의 Table로 도출합니다.***
- ***Join 성능이 좋고 관리가 편합니다.***

# Data Domain
- Source: https://en.wikipedia.org/wiki/Data_domain
- A data domain is the collection of values that a column may contain. ***The rule for determining the domain boundary may be as simple as a data type with an enumerated list of values.***
- For example, a database table that has information about people, with one record per person, might have a "marital status" column. This column might be declared as a string data type, and allowed to have one of two known code values: "M" for married, "S" for single, and NULL for records where marital status is unknown or not applicable. The data domain for the marital status column is: "M", "S".
## Cardinality
- Source: https://orangematter.solarwinds.com/2021/10/01/what-is-cardinality-in-a-database/
- The number of distinct values in a table column relative to the number of rows in the table.

# Data Standardization
- Source: https://www.getlore.io/knowledgecenter/data-standardization
- Data standardization is a data processing workflow that converts the structure of disparate datasets into a common data format. As part of the Data Preparation field, Data Standardization deals with the transformation of datasets after the data is pulled from source systems and before it's loaded into target systems.
- Data standardization enables the data consumer to analyze and use data in a consistent manner. Typically, when data is created and stored in the source system, it's structured in a particular way that is often unknown to the data consumer. Moreover, datasets that might be semantically related may be stored and represented differently, thereby making it difficult for a data consumer to aggregate or compare the datasets.

# Cursor
- https://en.wikipedia.org/wiki/Cursor_(databases)
- A database cursor is a mechanism that enables traversal over the records in a database. Cursors facilitate subsequent processing in conjunction with the traversal, such as retrieval, addition and removal of database records.
- 쿼리문에 의해서 반환되는 결과값들을 저장하는 메모리공간
- 커서를 open하고 나서 fetch가 발생하면 true 값을 반환
- To use cursors in SQL procedures, you need to do the following:
	- ***Declare*** a cursor that defines a result set.
	- ***Open*** the cursor to establish the result set.
	- ***Fetch*** the data into local variables as needed from the cursor, one row at a time.(Fetch : 커서에서 원하는 결과값을 추출하는 것)
	- ***Close*** the cursor when done.
## Explicit 커서: 사용자가 선언해서 생성 후 사용하는 SQL 커서, 주로 여러개의 행을 처리하고자 할 경우 사용.
## Implicit 커서: 오라클에서 자동으로 선언해주는 SQL 커서. 사용자는 생성 유무를 알 수 없다.

# Distributed Database
- Source: https://en.wikipedia.org/wiki/Distributed_database
- A distributed database is a database in which data is stored across different physical locations. It may be stored in multiple computers located in the same physical location (e.g., a data center); or maybe dispersed over a network of interconnected computers.
- Because distributed databases store data across multiple computers, ***distributed databases may improve performance at end-user worksites by allowing transactions to be processed on many machines, instead of being limited to one.***
- Two processes ensure that the distributed databases remain up-to-date and current: replication and duplication.
	- Replication involves using specialized software that looks for changes in the distributive database. Once the changes have been identified, the replication process makes all the databases look the same. The replication process can be complex and time-consuming, depending on the size and number of the distributed databases. This process can also require much time and computer resources.
	- Duplication, on the other hand, has less complexity. It identifies one database as a master and then duplicates that database. The duplication process is normally done at a set time after hours. This is to ensure that each distributed location has the same data. In the duplication process, users may change only the master database. This ensures that local data will not be overwritten.
- Both replication and duplication can keep the data current in all distributive locations.
- ***시스템 가용성이 높습니다.***
- ***Data integrity를 보장할 수 없습니다.***
- ***빠른 응답 속도와 통신 비용을 보장합니다.***
- ***데이터 처리 비용이 높습니다.***

# Database Lock
- Source: https://www.tutorialcup.com/interview/sql-interview-questions/db-locks.htm
- When two sessions or users of database try to update or delete the same data in a table, then there will be a concurrent update problem. In order to avoid this problem, database locks the data for the first user and allows him to update/delete the data. Once he is done with his update/delete, he `COMMIT`s or `ROLLBACK` the transaction, which will release the lock on the data. When lock on the data is released, another user can lock it for his changes.
- Therefore locking in the context of SQL is to hold the row or particular column which the user is going to update and not allowing any other session or user to insert/update/delete the data. It will not allow anyone to use the data until the lock on the data is released. Lock on the data will be released when the transaction is committed or rolled back.
- Whenever a user issues `UPDATE` or `DELETE` command, database will implicitly place the lock on the data. It does not require user to explicitly type lock on the data. Whenever the database sees `UPDATE` or `DELETE` statement, lock is automatically placed on the data.
- Reading the data when it is locked depends on the locking mechanism used. If the lock is read exclusive, then it will not allow to read locked data

# Optimizer Join
- Whenever you join a table to another table logically, the query optimizer can choose one of the three physical join iterators based on some cost based decision, these are hash join, nested loop join and merge join.
## Hash Join
- Source: https://en.wikipedia.org/wiki/Hash_join
- All variants of hash join algorithms involve building hash tables from the tuples of one or both of the joined relations, and subsequently probing those tables so that only tuples with the same hash code need to be compared for equality in equijoins.
- The classic hash join algorithm for an inner join of two relations proceeds as follows:
	- First, prepare a hash table using the contents of one relation, ideally whichever one is smaller after applying local predicates. This relation is called the build side of the join. The hash table entries are mappings from the value of the (composite) join attribute to the remaining attributes of that row (whichever ones are needed).
	- Once the hash table is built, scan the other relation (the probe side). For each row of the probe relation, find the relevant rows from the build relation by looking in the hash table.
- The first phase is usually called the "build" phase, while the second is called the "probe" phase. Similarly, the join relation on which the hash table is built is called the "build" input, whereas the other input is called the "probe" input.
- This algorithm is simple, but it requires that the smaller join relation fits into memory, which is sometimes not the case.
- Source: https://coding-factory.tistory.com/758
- 비용 기반 옵티마이저를 사용할 때만 사용될 수 있는 조인 방식.
- ***Equi join에서만 사용될 수 있습니다.***
- 해시 테이블을 생성하는 비용이 수반되므로 이 과정을 효율화하는 것이 성능 개선에 있어 가장 중요합니다.
- Build Input 해시 키 칼럼에 중복 값이 거의 없어야 효율적인 동작을 기대할 수 있습니다.
- 수행 빈도가 낮고 수행 시간이 오래 걸리는 ***대용량 테이블에 대해 유용합니다.***
- ***조인하는 컬럼의 인덱스가 존재하지 않을 경우에도 사용할 수 있습니다.***
- ***Random access로 인한 부하가 없습니다.***
## Nested Loop Join
- Source: https://en.wikipedia.org/wiki/Nested_loop_join
- A nested loop join is a naive algorithm that joins two sets by using two nested loops.
- ***Random access: This technique is only applicable when the joining column(s) of the inner side table is already sorted (indexed).***
- Outer table, Inner table, Index
- OLTP(OnLine Transaction Processing)에 유용합니다.
- 적은 데이터를 Join할 때 유리합니다.
- ***Full access join을 수행하는 두 개의 테이블에 대해서 Nested loop join을 수행하면 두 개의 테이블의 모든 행의 조합마다 join 연산을 수행하므로 성능이 떨어집니다.***
## (Sort) Merge Join
- ***Sorting한 후에 Merging하면서 정렬을 수행합니다.***
- ***Equi join과 Non-equi join 모두에서 사용할 수 있습니다.***
- ***정렬된 결과들을 통해 조인 작업이 수행되며 조인에 성공하면 추출 버퍼에 넣는 작업을 수행합니다.***

# PL/SQL
- Source: https://en.wikipedia.org/wiki/PL/SQL
- PL/SQL(***Procedural Language for SQL***) is Oracle Corporation's procedural extension for SQL and the Oracle relational database.
- Block 구조로 되어 있어 각 기능별로 모듈화가 가능합니다.
- 여러 SQL 문장을 Block으로 묶고 한 번에 Block 전부를 서버로 보내기 때문에 통신량을 줄일 수 있습니다.
- 변수, 상수 등을 선언하여 SQL 문장 간 값을 교환합니다.
- IF, LOOP 등의 절차형 언어를 사용하여 절차적인 프로그램이 가능하도록 합니다.
- DBMS 정의 에러나 사용자 정의 에러를 정의하여 사용할 수 있습니다.
- Oracle에 내장되어 있습니다.
- 응용 프로그램의 성능을 향상시킬 수 있습니다.
- ***임시 테이블로서 잠깐 사용하기 위해 PL/SQL 내부에서 테이블을 생성할 수 있습니다.***

# Relational Operation
- SELECT, PROJECT, JOIN, DIVISION

# Execution Plan
- 사용자가 SQL을 실행하여 데이터를 추출하려고 할 때 옵티마이저가 수립하는 작업절차를 뜻한다.
- SQL 해석 -> 실행계획 수립 -> 실행
## Optimizer
- 사용자가 질의한 SQL문에 대해 최적의 실행 방법(= Execution Plan)을 결정하는 역할 수행.
- 어떤 방법으로 처리하는 것이 최소 일량으로 동일한 일을 처리할 수 있을지 결정하는 것
### RBO(Rule-Based Optimizer)
- 규칙(우선 순위)을 가지고 실행계획을 생성, 실행계획을 생성하는 규칙을 이해하면 누구나 실행계획을 비교적 쉽게 예측 가능.
### CBO(Cost-Based Optimizer)
- 비용(예상되는 소요시간, 자원 사용량)이 가장 적은 실행계획을 선택하는 방식, 규칙기반 옵티마이저의 단점을 극복하기 위해서 출현.

# ETL (Extract, Transform, Load)
- Source: https://www.ibm.com/topics/etl
- ***ETL is a process that extracts, transforms, and loads data from multiple sources to a data warehouse or other unified data repository.***

# NoSQL
- Source: https://www.mongodb.com/nosql-explained
- *NoSQL databases store data in a format other than relational tables.*
- NoSQL databases allow developers to store huge amounts of unstructured data, giving them a lot of flexibility.
- Types of NoSQL Databases
## Document Databases
- A document database stores data in JSON, BSON , or XML documents (not Word documents or Google docs, of course). In a document database, documents can be nested. Particular elements can be indexed for faster querying.
- Documents can be stored and retrieved in a form that is much closer to the data objects used in applications, which means less translation is required to use the data in an application. SQL data must often be assembled and disassembled when moving back and forth between applications and storage.
- ***Document databases are popular with developers because they have the flexibility to rework their document structures as needed to suit their application, shaping their data structures as their application requirements change over time.*** This flexibility speeds development because in effect data becomes like code and is under the control of developers.
- The most widely adopted document databases are usually implemented with a scale-out architecture, providing a clear path to scalability of both data volumes and traffic.
- In order to retrieve all of the information about a user and their hobbies, a single document can be retrieved from the database. ***No joins are required, resulting in faster queries.***
- *NoSQL databases can store relationship data — they just store it differently than relational databases do.*
- Note that the way data is modeled in NoSQL databases can eliminate the need for multi-record transactions in many use cases. Consider the earlier example where we stored information about a user and their hobbies in both a relational database and a document database. *In order to ensure information about a user and their hobbies was updated together in a relational database, we'd need to use a transaction to update records in two tables. In order to do the same in a document database, we could update a single document — no multi-record transaction required.*
### MongoDB
## Key-Value Databases
- The simplest type of NoSQL database is a key-value store. Every data element in the database is stored as a key value pair consisting of an attribute name (or "key") and a value. In a sense, ***a key-value store is like a relational database with only two columns: the key or attribute name (such as state) and the value (such as Alaska).***
### Amazon DynamoDB
### Redis
## Column-Oriented Databases
- While a relational database stores data in rows and reads data row by row, a column store is organized as a set of columns. This means that when you want to run analytics on a small number of columns, you can read those columns directly without consuming memory with the unwanted data. Columns are often of the same type and benefit from more efficient compression, ***making reads even faster. Columnar databases can quickly aggregate the value of a given column (adding up the total sales for the year, for example).***
- Unfortunately there is no free lunch, which means that ***while columnar databases are great for analytics, the way in which they write data makes it very difficult for them to be strongly consistent as writes of all the columns require multiple write events on disk. Relational databases don't suffer from this problem as row data is written contiguously to disk.***
### Cassandra
### HBase
## Graph Databases
- A graph database focuses on the relationship between data elements. Each element is stored as a node (such as a person in a social media graph). The connections between elements are called links or relationships. In a graph database, connections are first-class elements of the database, stored directly.
- *A graph database is optimized to capture and search the connections between data elements, overcoming the overhead associated with JOINing multiple tables in SQL.*

# Data Warehouse (DW) & Data Mart (DM)
- Source: https://www.guru99.com/data-warehouse-vs-data-mart.html
- *Data Warehouse is focused on all departments in an organization whereas Data Mart focuses on a specific group.*
- *Comparing Data Warehouse vs Data Mart, Data Warehouse size range is 100 GB to 1 TB+ whereas Data Mart size is less than 100 GB.*
## Data Warehouse (DW)
- A Data Warehouse collects and manages data from varied sources to provide meaningful business insights. *It is a collection of data which is separate from the operational systems and supports the decision making of the company.*
## Data Mart (DM)
- It is subject-oriented, and it is designed to meet the needs of a specific group of users.

# Python Libraries
# `pymssql`, `pymysql`, `psycopg2`
## `pymssql.connect()`, `pymysql.connect()`, `psycopg2.connect()`
```python
conn = pymssql.connect(server="125.60.68.233", database="eiparkclub", user="myhomie", password="homie2021!@#", charset="utf8")
```
### `conn.cursor()`
```python
cur = conn.cursor()
```
#### `cur.excute()`
```python
cur.excute(query)
```
#### `cur.fetchall()`
```python
result = cur.fetchall() 
```
#### `cur.close()`
#### `cur.description`
```python
salary = pd.DataFrame(result, columns=[col[0] for col in cur.description])
```

# `psycopg2`
## `psycopg2.extras`
### `RealDictCursor`
```python
from psycopg2.extras import RealDictCursor
```
## `psycopg2.connect()`
### `conn.cursor()`
```python
cur = conn.cursor(cursor_factory=RealDictCursor)
```

# `traceback`
```python
import traceback
```
## `traceback.format_exec()`
```python
try:            
    cur.execute(query)            
    result = cur.fetchall()        
except Exception as e:            
    msg = traceback.format_exc()            
    msg += "\n\n Query: \n" + query            
    print(msg)  
```