SELECT *
FROM 단계별진행_예정 단예;
--- 업데이트: 매일 4시
--- 사용 영역: `예정 (착공 8개월 전)`, `예정 #개`

SELECT *
FROM 단계별진행_착공초기 단착;
--- 업데이트: 매일 4시
--- 사용 영역: `전체 현장 수 (진행 현장)`, `착공 초기(착공 6개월 이내)`, `착공 초기 #개`

SELECT *
FROM 단계별진행_진행 단진;
--- 업데이트: 매일 4시
--- 사용 영역: `전체 현장 수 (진행 현장)`, `진행 (착공 6개월 다음날부터 준공 90일 전일까지)`, `진행 #개`

SELECT *
FROM 단계별진행_준공예정 단준;
--- 업데이트: 매일 4시
--- 사용 영역: `전체 현장 수 (진행 현장)`, `준공 예정 (준공 90일 전 이내)`, `준공 예정`

SELECT *
FROM DATAMART_COMBBZPLN_ALL DCA;
--- 업데이트: 매달 21일 4시
--- 사용 영역: `이번 달 소화 실적/계획`, `올해 누계 소화 실적/계획`, `이번 달 소화 계획`, `이번 달 소화 실적`, `계획 대비 실적` (1), `올해 누계 소화 계획`, `올해 누계 소화 실적`, `계획 대비 실적` (2), `월별 소화 추이`, `사업형태별 소화 누계`, `현장 소화(전체)`, `현장별 소화 상세`

--## 올해 현장별 소화 계획, 소화 실적
SELECT 현장코드, 현장명, ROUND(SUM(CAST(계획 AS FLOAT)), 0) AS 누계_계획, ROUND(SUM(CAST(실적 AS FLOAT)), 0) AS 누계_실적
FROM DATAMART_COMBBZPLN_ALL DCA
WHERE 날짜 > '2020-12-31' AND 날짜 < '2022-01-01' AND 실적 IS NOT NULL
GROUP BY 현장코드, 현장명
ORDER BY 현장코드;

--## 올해 월별 소화 실적 합계, 소화 계획 합계
SELECT 날짜, ROUND(SUM(CAST(실적 AS FLOAT)), 0) AS 실적, ROUND(SUM(CAST(계획 AS FLOAT)), 0) AS 계획
FROM DATAMART_COMBBZPLN_ALL DCA
WHERE 날짜 > '2020-12-31' AND 날짜 < '2022-01-01'
GROUP BY 날짜
ORDER BY 날짜;

--## 올해 월별 누적 소화 실적, 누적 소화 계획
SELECT 날짜, SUM(실적) OVER(ORDER BY 날짜) AS 누계_실적, SUM(계획) OVER(ORDER BY 날짜) AS 누계_계획
FROM (
	SELECT 날짜, ROUND(SUM(CAST(실적 AS FLOAT)), 0) AS 실적, ROUND(SUM(CAST(계획 AS FLOAT)), 0) AS 계획
	FROM DATAMART_COMBBZPLN_ALL DCA
	WHERE 날짜 > '2020-12-31' AND 날짜 < '2023-01-01'
	GROUP BY 날짜	) A;

--## 올해 소화 실적, 소화 계획
SELECT SUM(실적) AS 실적, SUM(계획) AS 계획
FROM (
	SELECT 날짜, ROUND(SUM(CAST(실적 AS FLOAT)), 0) AS 실적, ROUND(SUM(CAST(계획 AS FLOAT)), 0) AS 계획
	FROM DATAMART_COMBBZPLN_ALL DCA
	WHERE 날짜 > '2020-12-31' AND 날짜 < '2022-01-01'
	GROUP BY 날짜) A
WHERE 실적 IS NOT NULL;

--## 올해 현장별 청산율 실적
SELECT 현장코드, 현장명, CAST(청산율실적 AS FLOAT)
FROM (
	SELECT *, RANK() OVER(PARTITION BY 현장코드 ORDER BY 날짜 DESC) AS rank
	FROM DATAMART_COMBBZPLN_ALL DCA
	WHERE 실적 IS NOT NULL) AS A
WHERE rank = 1
ORDER BY 현장코드;

SELECT *
FROM 프로젝트목록 프;
--- 사용 영역: `사업형태별 소화 누계`, `현장 소화(전체)`

SELECT *
FROM 프로젝트목록_진행 프진;
--- 업데이트: 매일 4시
--- 사용 영역: 카드 지표-`건축`, `인프라`, `전체`, `국내`, `해외`, `건축`, `인프라`, `기타`, `국내 진행 현장`

--## 단계별 현장
SELECT *
FROM (
	SELECT *,
		CASE WHEN DATEADD(MONTH, -8, 착공일자) <= GETDATE() - 1 AND GETDATE() - 1 < 착공일자 THEN '예정'
			WHEN 착공일자 <= GETDATE() - 1 AND GETDATE() - 1 <= DATEADD(MONTH, 6, 착공일자) THEN '착공 초기'
			WHEN DATEADD(month, 6, 착공일자) < GETDATE() - 1 AND GETDATE() - 1 < DATEADD(DAY, -90, 준공일자) THEN '공사 중'
			WHEN DATEADD(DAY, -90, 준공일자) <= GETDATE() - 1 AND GETDATE() - 1 < 준공일자 THEN '준공 예정'
			WHEN GETDATE() - 1 <= 준공일자 AND YEAR(GETDATE() - 1) = YEAR(준공일자) THEN '완료'
			ELSE NULL END 단계
	FROM (
		SELECT 현장코드, 현장명, CAST(착공일자 AS DATE) AS 착공일자, CAST(준공일자 AS DATE) AS 준공일자, 			사업유형분류체계1
		FROM 프로젝트목록_진행 프진) A) B
WHERE 현장코드 NOT IN('C1051C') AND 단계 IN('착공 초기', '공사 중', '준공 예정')
ORDER BY 단계, 현장코드;

--SELECT B.*, 단진.현장코드
--FROM (
--	SELECT *,
--		CASE WHEN DATEADD(MONTH, -8, 착공일자) <= GETDATE() - 1 AND GETDATE() - 1 < 착공일자 THEN '예정'
--			WHEN 착공일자 <= GETDATE() - 1 AND GETDATE() - 1 <= DATEADD(MONTH, 6, 착공일자) THEN '착공 초기'
--			WHEN DATEADD(month, 6, 착공일자) < GETDATE() - 1 AND GETDATE() - 1 < DATEADD(DAY, -90, 준공일자) THEN '공사 중'
--			WHEN DATEADD(DAY, -90, 준공일자) <= GETDATE() - 1 AND GETDATE() - 1 < 준공일자 THEN '준공 예정'
--			WHEN GETDATE() - 1 <= 준공일자 AND YEAR(GETDATE() - 1) = YEAR(준공일자) THEN '완료'
--			ELSE NULL END 단계
--	FROM (
--		SELECT 현장코드, 현장명, CAST(착공일자 AS DATE) AS 착공일자, CAST(준공일자 AS DATE) AS 준공일자
--		FROM 프로젝트목록_진행 프진
--		WHERE 공사유형분류체계1 != '해외') A) B, 단계별진행_진행 단진
--WHERE B.현장코드 += 단진.현장코드 AND B.단계 IS NOT NULL
--ORDER BY B.단계, B.준공일자;

--# `Site Coordinates.csv`
--- 사용 영역: `국내 진행 현장`

SELECT *
FROM 소화_청산 소청;
--- 업데이트: 매달 21일 4시
--- 사용 영역: `이번 달 청산율`, `관리 현장 추이`-`소화`, `관리 현장 추이`-표 부분

SELECT *
FROM 프로젝트목록_현장수추이_언피벗 프현언;
--- 업데이트: 매일 4시
--- 사용 영역: `관리 현장 추이`-현장 수 부분

SELECT *
FROM 현장현황 현;
--- 업데이트: 매달 21일 4시
--- 사용 영역: 카드 지표-`소화`-`계획`, `실적`, `청산율`

--## 올해 월별 소화 실적 합계, 소화 계획 합계
SELECT 현장코드, 현장명, CAST(날짜 AS DATE) AS 날짜, ROUND(SUM(CAST(실적 AS FLOAT)), 0) AS 실적, ROUND(SUM(CAST(계획 AS FLOAT)), 0) AS 계획
FROM DATAMART_COMBBZPLN_ALL DCA
WHERE 날짜 > '2019-12-31'
	AND 실적 IS NOT NULL
GROUP BY 날짜, 현장코드, 현장명
ORDER BY 현장코드, 날짜;