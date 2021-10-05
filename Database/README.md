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
- To satisfy First normal form, each column of a table must have a single value. Columns which contain sets of values or nested records are not allowed.
## Physical Data Models
- Physical models provide a schema for how the data will be physically stored within a database. As such, they’re the least abstract of all. They offer a finalized design that can be implemented as a relational database, including associative tables that illustrate the relationships among entities as well as the primary keys and foreign keys that will be used to maintain those relationships.

# Wireframe & Storyboard
# Wireframe
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
### Non-Identifying Relationship
- When the primary key of the parent must not become primary key of the child.
- A의 값이 없더라도 B의 값은 독자적으로 의미를 가짐.