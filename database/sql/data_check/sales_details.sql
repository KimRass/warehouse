SELECT *
FROM 사업별미수현황총괄표_작업;

SELECT *
FROM 계약일보_작업;

--## 일별 계약 수
SELECT 코드, 현장명, 분양구분, 분양종류, 계약일자, COUNT(계약일자) AS cnt
FROM 계약일보_작업
WHERE 1=1
	AND CAST(계약일자 AS DATE) > '2020-12-31'
	AND CAST(계약일자 AS DATE) < '2021-10-01'
	AND 코드 IN('C032')
GROUP BY 코드, 현장명, 분양구분, 분양종류, 계약일자
ORDER BY 계약일자

--## 일별 해약 수
SELECT 코드, 현장명, 분양구분, 분양종류, 계약일자, COUNT(해약일자) AS cnt
FROM 계약일보_작업
WHERE 1=1
	AND CAST(해약일자 AS DATE) > '2020-12-31'
	AND CAST(해약일자 AS DATE) < '2021-10-01'
	AND 코드 IN('C032')
GROUP BY 코드, 현장명, 분양구분, 분양종류, 계약일자
ORDER BY 계약일자