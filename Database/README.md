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
- Foreign key is a column that creates a relationship between two tables. The purpose of Foreign keys is to maintain data integrity and allow navigation between two different instances of an entity.
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
- They offer a big-picture view of what the system will contain, how it will be organized, and which business rules are involved.
- Conceptual models are usually created as part of the process of gathering initial project requirements.
## Logical Data Models
- They are less abstract and provide greater detail about the concepts and relationships in the domain under consideration. One of several formal data modeling notation systems is followed. These indicate data attributes, such as data types and their corresponding lengths, and show the relationships among entities. Logical data models don’t specify any technical system requirements.
## Physical Data Models
- They provide a schema for how the data will be physically stored within a database. As such, they’re the least abstract of all. They offer a finalized design that can be implemented as a relational database, including associative tables that illustrate the relationships among entities as well as the primary keys and foreign keys that will be used to maintain those relationships.

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
