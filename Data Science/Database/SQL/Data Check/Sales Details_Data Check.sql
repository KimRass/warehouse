SELECT *
FROM 사업별미수현황총괄표_작업;

SELECT *
FROM 계약일보_작업;

SELECT *
FROM 사업별미수현황총괄표_작업 사작
WHERE 코드 = 'D027';

SELECT *
FROM 계약일보_작업 계작 LEFT OUTER JOIN 사업별미수현황총괄표_작업 사작 ON 계작.코드 = 사작.코드;

SELECT CAST(계약일자 AS DATE)
FROM 계약일보_작업;

SELECT 계작.코드, 계작.현장명, 계작.분양구분, 계작.분양종류, 계작.계약일자, 계작.해약일자
FROM 계약일보_작업 계작 LEFT OUTER JOIN 사업별미수현황총괄표_작업 사작 ON 계작.코드 = 사작.코드
WHERE CAST(계약일자 AS DATE) > '2020-12-31' AND 계작.코드 = 'B1037C'

--'A694', 'C030', 'C1041C', 'C1043C', 'C1044C', 'C1044O', 'C1057C', 'C1059C', 'A665', 'B1037C', 'A710B', 'D022', 'D027', 'D1045C', 'C012', 'C040', 'C041', 'C1049C', 'C1058C', 'C1061C',  'C1062C', 'C032', 'I0009C', 'C1041C')

--## 일별 계약 수
SELECT 코드, 현장명, 분양구분, 분양종류, 계약일자, COUNT(계약일자) AS cnt
FROM 계약일보_작업
WHERE 1=1
	AND CAST(계약일자 AS DATE) > '2020-12-31'
	AND CAST(계약일자 AS DATE) < '2021-10-01'
	AND 코드 IN('C032')
--	AND 분양구분 = '일반'
--	AND 분양가 = 0
GROUP BY 코드, 현장명, 분양구분, 분양종류, 계약일자
ORDER BY 계약일자

--## 일별 해약 수
SELECT 코드, 현장명, 분양구분, 분양종류, 계약일자, COUNT(해약일자) AS cnt
FROM 계약일보_작업
WHERE 1=1
	AND CAST(해약일자 AS DATE) > '2020-12-31'
	AND CAST(해약일자 AS DATE) < '2021-10-01'
	AND 코드 IN('C032')
--	AND 분양구분 = '일반'
--	AND 분양가 = 0
GROUP BY 코드, 현장명, 분양구분, 분양종류, 계약일자
ORDER BY 계약일자

--## 계약, 해약 건
SELECT *
FROM 계약일보_작업 계작
WHERE 코드 = 'D027'
	AND 분양구분 = '조합'
	AND ((CAST(계약일자 AS DATE) > '2020-12-31' OR CAST(계약일자 AS DATE) < '2021-10-01')
		OR (CAST(해약일자 AS DATE) > '2020-12-31' OR CAST(해약일자 AS DATE) < '2021-10-01'))
ORDER BY 계약일자