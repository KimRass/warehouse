SELECT DISTINCT ym_magam
FROM DATAMART_DHDT_TOTAL DDT
ORDER BY ym_magam DESC

SELECT *
FROM DATAMART_DHDT_TOTAL DDT;
--- 업데이트: 매달 20일 4시?
--- 사용 영역: 카드 지표 전체, 카드 지표-`사업유형별 실적` 전체, `누적 매출`, `누적 매출이익율`, `누적 영업이익`, `누적 영업이익율`, `연간 손익`, `당기 누적 손익`-(`매출액`, `영업이익`), `주요 부문별 매출`, `주요 부문별 매출이익율`, `손익계산서`, `재무상태표`, `부채비율`, `차입금의존도`, `유동비율`, `이자보상배율`
-- 매출: LEFT(cd_trial, 2) = '41'
-- 매출원가: LEFT(cd_trial, 2) = '46'
-- 판관비: LEFT(cd_trial, 2) = '61'
-- 영업이익: 매출 - 매출원가 - 판관비
-- 매출이익: 매출 - 매출원가
-- 부채: LEFT(cd_trial, 1) = '2'
-- 자산: LEFT(cd_trial, 1) = '1'
-- 자본: 자산 - 부채
-- 부채비율: 부채/자본
-- 차입금: cd_trial IN('22101101', '22201101', '22101102', '22201201', '22301101', '22301201', '27101101', '27101102', '27101104', '27101201', '27301101', '27301201', '22101100', '22201100', '27101100')
-- 차입금의존도: 차입금/자산
-- 유동자산: LEFT(cd_trial, 2) IN('11', '12', '14')
-- 유동부채: LEFT(cd_trial, 2) IN('21', '22', '23')
-- 유동비율: 유동자산/유동부채
-- 이자비용: LEFT(cd_trial, 6) = '716011'
-- 이자보상배율: 영업이익/이자비용
-- 영업외수익: (71103101 <= CAST(cd_trial AS INT) AND CAST(cd_trial AS INT) <= 71109901)
-- 영업외비용: (71603101 <= CAST(cd_trial AS INT) AND CAST(cd_trial AS INT) <= 71609901)
-- 금융수익: (71101100 <= CAST(cd_trial AS INT) AND CAST(cd_trial AS INT) <= 71102201)
-- 금융비용: (71601100 <= CAST(cd_trial AS INT) AND CAST(cd_trial AS INT) <= 71602201)
-- 기타영업외손익: (71907008 <= CAST(cd_trial AS INT) AND CAST(cd_trial AS INT) <= 71907609)
-- 세전이익: 영업이익 + 영업외수익 - 영업외비용 + 금융수익 - 금융비용 + 기타영업외손익

--## 연도별 매출액, 매출원가, 판관비, 영업이익
SELECT yr, sales, sales_cost, fee, sales - sales_cost - fee AS profit
FROM (
	SELECT yr, ROUND(SUM(c1)/100000000, 0) AS sales, ROUND(SUM(c2)/100000000, 0) AS sales_cost, ROUND(SUM(c3)/100000000, 0) AS fee
	FROM (
		SELECT LEFT(ym_magam, 4) AS yr, CASE WHEN LEFT(cd_trial, 2) = '41' THEN am_account_wol END AS c1, CASE WHEN LEFT(cd_trial, 2) = '46' THEN am_account_wol END AS c2, CASE WHEN LEFT(cd_trial, 2) = '61' THEN am_account_wol END AS c3
		FROM DATAMART_DHDT_TOTAL DDT
		WHERE cd_dept_acnt = 'HD00') A
	GROUP BY yr) B;

--## 연월별 연 단위 누적 매출
SELECT ym_magam, SUM(am_account)
FROM DATAMART_DHDT_TOTAL DDT
WHERE cd_dept_acnt = 'HD00'
	AND LEFT(cd_trial, 2) = '41'
GROUP BY ym_magam
ORDER BY ym_magam

--## 연도별 자산, 부채, 자본, 부채비율
SELECT *, prop - debt AS cap, ROUND(debt/(prop - debt)*100, 1) AS debt_ratio
FROM (
	SELECT ym_magam, ROUND(SUM(prop)/100000000, 0) AS prop, ROUND(SUM(debt)/100000000, 0) AS debt
	FROM (
		SELECT ym_magam, MAX(ym_magam) OVER(PARTITION BY LEFT(ym_magam, 4)) AS max_yr, CASE WHEN LEFT(cd_trial, 1) = '1' THEN am_account END AS prop, CASE WHEN LEFT(cd_trial, 1) = '2' THEN am_account END AS debt
		FROM DATAMART_DHDT_TOTAL DDT
		WHERE cd_dept_acnt = 'HD00') A
	WHERE ym_magam = max_yr
	GROUP BY ym_magam) B	
ORDER BY CAST(ym_magam AS INT);

SELECT *
FROM DATAMART_DHDT_ACNT_GROUP DDAG
WHERE cd_dept_acnt LIKE '%L%';

SELECT *
FROM DATAMART_DHDT_ACNT_GROUP DDAG;
--- 업데이트: 매달 20일 4시?
--- 사용 영역:  카드 지표-`사업유형별 실적` 전체, `주요 부문별 매출`, `주요 부문별 매출이익율`

SELECT *
FROM DATAMART_DHDT_TOTAL_GROUP DDTG;
--- 업데이트: 매달 20일 4시?
--- 사용 영역: `주요 부문별 매출`, `주요 부문별 매출이익율`

SELECT DDT.ym_magam, MAX(DDT.ym_magam) OVER(PARTITION BY LEFT(DDT.ym_magam, 4)) AS max_yr, DDTG.ds_group, ROUND(SUM(am_account_wol)/100000000, 0) AS sales
FROM DATAMART_DHDT_TOTAL DDT LEFT OUTER JOIN DATAMART_DHDT_ACNT_GROUP DDAG ON DDT.cd_corp = DDAG.cd_corp AND DDT.cd_dept_acnt = DDAG.cd_dept_acnt LEFT OUTER JOIN DATAMART_DHDT_TOTAL_GROUP DDTG ON DDAG.cd_group = DDTG.cd_group
WHERE LEFT(DDT.cd_trial, 2) = '41' AND ds_group != '기타' AND LEFT(ym_magam, 4) = '2021' AND ds_group = '인프라'
GROUP BY DDT.ym_magam, DDTG.ds_group

--## 그룹, 연도별 매출
SELECT ds_group, LEFT(max_yr, 4) AS yr, SUM(sales) AS value
FROM (
	SELECT DDT.ym_magam, MAX(DDT.ym_magam) OVER(PARTITION BY LEFT(DDT.ym_magam, 4)) AS max_yr, DDTG.ds_group, ROUND(SUM(am_account_wol)/100000000, 0) AS sales
	FROM DATAMART_DHDT_TOTAL DDT
		LEFT OUTER JOIN DATAMART_DHDT_ACNT_GROUP DDAG ON DDT.cd_corp = DDAG.cd_corp AND DDT.cd_dept_acnt = DDAG.cd_dept_acnt
		LEFT OUTER JOIN DATAMART_DHDT_TOTAL_GROUP DDTG ON DDAG.cd_group = DDTG.cd_group
	WHERE ds_group != '기타'
		AND LEFT(DDT.cd_trial, 2) = '41'
	GROUP BY DDT.ym_magam, DDTG.ds_group) A
GROUP BY ds_group, max_yr
ORDER BY ds_group, LEFT(max_yr, 4);

SELECT *
FROM DATAMART_DIFV_PL DDP;
--- 업데이트: 매달 20일 4시?
--- 사용 영역: `당기 누적 손익`-(`매출액 계획`, `영업이익 계획`)

--## 올해 월별 누적 매출액 계획
SELECT MONTH, ROUND(SUM(pl) OVER(ORDER BY CAST(MONTH AS INT) ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)/100000000, 0)
FROM (
	SELECT MONTH, SUM(am_plan) AS pl
	FROM DATAMART_DIFV_PL DDP
	WHERE ds_item = '매출액' AND YEAR = 2021
	GROUP BY MONTH) AS A
ORDER BY CAST(MONTH AS INT);

--## 올해 월별 누적 영업이익 계획
SELECT MONTH, ROUND(SUM(pl) OVER(ORDER BY CAST(MONTH AS INT) ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)/100000000, 0)
FROM (
	SELECT MONTH, SUM(am_plan) AS pl
	FROM DATAMART_DIFV_PL DDP
	WHERE ds_item = '영업이익' AND YEAR = 2021
	GROUP BY MONTH) AS A
ORDER BY CAST(MONTH AS INT);

--# 엑셀 파일-전사
--- 21.07~21.09 데이터만 있음. 이전 데이터는`DATAMART_DHDT_TOTAL`에 있음. 블렌딩 기능 사용
--- 사용 영역: 카드 지표(`경영 현황`),`누적 매출`,`누적 매출이익율`,`누적 영업이익`,`누적 영업이익율`,`연간 손익`,`당기 누적 손익`,`손익계산서`,`재무상태표`,`부채비율`,`차입금의존도`,`유동비율`,`이자보상배율`

--# 엑셀 파일-부문
--- DM과 블렌딩 x 엑셀만 단독으로 사용
--- 사용 영역: 카드 지표(`사업유형별 실적`),`주요 부문별 매출`,`주요 부문별 매출이익율`
