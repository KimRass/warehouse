# Key
## PRIMARY KEY
- The `PRIMARY KEY` constraint uniquely identifies each record in a table. Primary keys must contain UNIQUE values, and cannot contain NULL values. A table can have only ONE primary key; and in the table, this primary key can consist of single or multiple columns (fields).
## FOREIGN KEY
- The `FOREIGN KEY` constraint is used to prevent actions that would destroy links between tables. A `FOREIGN KEY` is a field (or collection of fields) in one table, that refers to the `PRIMARY KEY` in another table. The table with the foreign key is called the child table, and the table with the primary key is called the referenced or parent table.
- The `FOREIGN KEY` constraint prevents invalid data from being inserted into the foreign key column, because it has to be one of the values contained in the parent table.

# View
1. 뷰는 사용자에게 접근이 허용된 자료만을 제한적으로 보여주기 위해 하나 이상의 기본 테이블로부터 유도된, 이름을 가지는 가상 테이블이다.
2. 뷰는 저장장치 내에 물리적으로 존재하지 않지만 사용자에게 있는 것처럼 간주된다.
3. 뷰는 데이터 보정작업, 처리과정 시험 등 임시적인 작업을 위한 용도로 활용된다.
4. 뷰는 조인문의 사용 최소화로 사용상의 편의성을 최대화 한다.
## 뷰(View)의 특징
1. 뷰는 기본테이블로부터 유도된 테이블이기 때문에 기본 테이블과 같은 형태의 구조를 사용하며, 조작도 기본 테이블과 거의 같다.
2. 뷰는 가상 테이블이기 때문에 물리적으로 구현되어 있지 않다.
3. 데이터의 논리적 독립성을 제공할 수 있다.
4. 필요한 데이터만 뷰로 정의해서 처리할 수 있기 때문에 관리가 용이하고 명령문이 간단해진다.
5. 뷰를 통해서만 데이터에 접근하게 하면 뷰에 나타나지 않는 데이터를 안전하게 보호하는 효율적인 기법으로 사용할 수 있다.
6. 기본 테이블의 기본키를 포함한 속성(열) 집합으로 뷰를 구성해야지만 삽입, 삭제, 갱신, 연산이 가능하다.
7. 일단 정의된 뷰는 다른 뷰의 정의에 기초가 될 수 있다.
8. 뷰가 정의된 기본 테이블이나 뷰를 삭제하면 그 테이블이나 뷰를 기초로 정의된 다른 뷰도 자동으로 삭제된다.
## 뷰(View)사용시 장 단점
장점
1. 논리적 데이터 독립성을 제공한다.
2. 동일 데이터에 대해 동시에 여러사용자의 상이한 응용이나 요구를 지원해 준다.
3. 사용자의 데이터관리를 간단하게 해준다.
4. 접근 제어를 통한 자동 보안이 제공된다.
단점
1. 독립적인 인덱스를 가질 수 없다.
2. ALTER VIEW문을 사용할 수 없다. 즉 뷰의 정의를 변경할 수 없다.
3. 뷰로 구성된 내용에 대한 삽입, 삭제, 갱신, 연산에 제약이 따른다.