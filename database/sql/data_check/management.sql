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

--## 연 단위 월 누적 매출 등
SELECT A.ym_magam, 매출, 매출원가, 판관비, 매출 - 매출원가 - 판관비 AS 영업이익, 매출 - 매출원가 AS 매출이익, 부채, 자산, 자산 - 부채 AS 자본, 부채/(자산 - 부채) AS 부채비율, 차입금, 차입금/자산 AS 차입금의존도, 유동자산, 유동부채, 유동자산/유동부채 AS 유동비율, 이자비용, (매출 - 매출원가 - 판관비)/이자비용 AS 이자보상배율, 영업외수익, 금융수익, 금융비용, 기타영업외손익, 매출 - 매출원가 - 판관비 + 영업외수익 - 영업외비용 + 금융수익 - 금융비용 + 기타영업외손익 AS 세전이익
FROM (
	(SELECT ym_magam, SUM(am_account) AS 매출
	FROM DATAMART_DHDT_TOTAL DDT
	WHERE cd_dept_acnt = 'HD00'
		AND LEFT(cd_trial, 2) = '41'
	GROUP BY ym_magam) A FULL OUTER JOIN (
		SELECT ym_magam, SUM(am_account) AS 매출원가
		FROM DATAMART_DHDT_TOTAL DDT
		WHERE cd_dept_acnt = 'HD00'
			AND LEFT(cd_trial, 2) = '46'
		GROUP BY ym_magam) B
	ON A.ym_magam = B.ym_magam FULL OUTER JOIN (
		SELECT ym_magam, SUM(am_account) AS 판관비
		FROM DATAMART_DHDT_TOTAL DDT
		WHERE cd_dept_acnt = 'HD00'
			AND LEFT(cd_trial, 2) = '61'
		GROUP BY ym_magam) C
	ON A.ym_magam = C.ym_magam FULL OUTER JOIN (
		SELECT ym_magam, SUM(am_account) AS 부채
		FROM DATAMART_DHDT_TOTAL DDT
		WHERE cd_dept_acnt = 'HD00'
			AND LEFT(cd_trial, 1) = '2'
		GROUP BY ym_magam) D
	ON A.ym_magam = D.ym_magam FULL OUTER JOIN (
		SELECT ym_magam, SUM(am_account) AS 자산
		FROM DATAMART_DHDT_TOTAL DDT
		WHERE cd_dept_acnt = 'HD00'
			AND LEFT(cd_trial, 1) = '1'
		GROUP BY ym_magam) E
	ON A.ym_magam = E.ym_magam FULL OUTER JOIN (
		SELECT ym_magam, SUM(am_account) AS 차입금
		FROM DATAMART_DHDT_TOTAL DDT
		WHERE cd_dept_acnt = 'HD00'
			AND cd_trial IN('22101101', '22201101', '22101102', '22201201', '22301101', '22301201', '27101101', '27101102', '27101104', '27101201', '27301101', '27301201', '22101100', '22201100', '27101100')
		GROUP BY ym_magam) F
	ON A.ym_magam = F.ym_magam FULL OUTER JOIN (
		SELECT ym_magam, SUM(am_account) AS 유동자산
		FROM DATAMART_DHDT_TOTAL DDT
		WHERE cd_dept_acnt = 'HD00'
			AND LEFT(cd_trial, 2) IN('11', '12', '14')
		GROUP BY ym_magam) G
	ON A.ym_magam = G.ym_magam FULL OUTER JOIN (
		SELECT ym_magam, SUM(am_account) AS 유동부채
		FROM DATAMART_DHDT_TOTAL DDT
		WHERE cd_dept_acnt = 'HD00'
			AND LEFT(cd_trial, 2) IN('21', '22', '23')
		GROUP BY ym_magam) H
	ON A.ym_magam = H.ym_magam FULL OUTER JOIN (
		SELECT ym_magam, SUM(am_account) AS 이자비용
		FROM DATAMART_DHDT_TOTAL DDT
		WHERE cd_dept_acnt = 'HD00'
			AND LEFT(cd_trial, 6) = '716011'
		GROUP BY ym_magam) I
	ON A.ym_magam = I.ym_magam FULL OUTER JOIN (
		SELECT ym_magam, SUM(am_account) AS 영업외수익
		FROM DATAMART_DHDT_TOTAL DDT
		WHERE cd_dept_acnt = 'HD00'
			AND (71103101 <= CAST(cd_trial AS INT) AND CAST(cd_trial AS INT) <= 71109901)
		GROUP BY ym_magam) J
	ON A.ym_magam = J.ym_magam FULL OUTER JOIN (
		SELECT ym_magam, SUM(am_account) AS 영업외비용
		FROM DATAMART_DHDT_TOTAL DDT
		WHERE cd_dept_acnt = 'HD00'
			AND (71603101 <= CAST(cd_trial AS INT) AND CAST(cd_trial AS INT) <= 71609901)
		GROUP BY ym_magam) K
	ON A.ym_magam = K.ym_magam FULL OUTER JOIN (
		SELECT ym_magam, SUM(am_account) AS 금융수익
		FROM DATAMART_DHDT_TOTAL DDT
		WHERE cd_dept_acnt = 'HD00'
			AND (71101100 <= CAST(cd_trial AS INT) AND CAST(cd_trial AS INT) <= 71102201)
		GROUP BY ym_magam) L
	ON A.ym_magam = L.ym_magam FULL OUTER JOIN (
		SELECT ym_magam, SUM(am_account) AS 금융비용
		FROM DATAMART_DHDT_TOTAL DDT
		WHERE cd_dept_acnt = 'HD00'
			AND (71601100 <= CAST(cd_trial AS INT) AND CAST(cd_trial AS INT) <= 71602201)
		GROUP BY ym_magam) M
	ON A.ym_magam = M.ym_magam FULL OUTER JOIN (
		SELECT ym_magam, SUM(am_account) AS 기타영업외손익
		FROM DATAMART_DHDT_TOTAL DDT
		WHERE cd_dept_acnt = 'HD00'
			AND (71907008 <= CAST(cd_trial AS INT) AND CAST(cd_trial AS INT) <= 71907609)
		GROUP BY ym_magam) N
	ON A.ym_magam = N.ym_magam
)
ORDER BY A.ym_magam;

SELECT *
FROM DATAMART_DHDT_ACNT_GROUP DDAG;
--- 업데이트: 매달 20일 4시?
--- 사용 영역:  카드 지표-`사업유형별 실적` 전체, `주요 부문별 매출`, `주요 부문별 매출이익율`

SELECT *
FROM DATAMART_DHDT_TOTAL_GROUP DDTG;
--- 업데이트: 매달 20일 4시?
--- 사용 영역: `주요 부문별 매출`, `주요 부문별 매출이익율`

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

--## 연도, 부문별 매출
SELECT DDTG.ds_group, DDT.ym_magam, SUM(am_account)
FROM DATAMART_DHDT_TOTAL DDT, DATAMART_DHDT_ACNT_GROUP DDAG, DATAMART_DHDT_TOTAL_GROUP DDTG 
WHERE DDT.cd_corp = DDAG.cd_corp
	AND DDT.cd_dept_acnt = DDAG.cd_dept_acnt
	AND DDAG.cd_group = DDTG.cd_group
	AND LEFT(cd_trial, 2) = '41'
	AND DDTG.ds_group NOT IN('기타')
	AND RIGHT(ym_magam, 2) = 12
GROUP BY DDTG.ds_group, DDT.ym_magam
ORDER BY DDTG.ds_group, DDT.ym_magam

--## 연 단위 월 누적 매출 등
SELECT A.ym_magam, 매출, 매출원가, 판관비, 매출 - 매출원가 - 판관비 AS 영업이익, 매출 - 매출원가 AS 매출이익, 부채, 자산, 자산 - 부채 AS 자본, 부채/(자산 - 부채) AS 부채비율, 차입금, 차입금/자산 AS 차입금의존도, 유동자산, 유동부채, 유동자산/유동부채 AS 유동비율, 이자비용, (매출 - 매출원가 - 판관비)/이자비용 AS 이자보상배율, 영업외수익, 금융수익, 금융비용, 기타영업외손익, 매출 - 매출원가 - 판관비 + 영업외수익 - 영업외비용 + 금융수익 - 금융비용 + 기타영업외손익 AS 세전이익
FROM (
	(SELECT ym_magam, SUM(am_account) AS 매출
	FROM DATAMART_DHDT_TOTAL DDT
	WHERE cd_dept_acnt = 'HD00'
		AND LEFT(cd_trial, 2) = '41'
	GROUP BY ym_magam) A FULL OUTER JOIN (
		SELECT ym_magam, SUM(am_account) AS 매출원가
		FROM DATAMART_DHDT_TOTAL DDT
		WHERE cd_dept_acnt = 'HD00'
			AND LEFT(cd_trial, 2) = '46'
		GROUP BY ym_magam) B
	ON A.ym_magam = B.ym_magam FULL OUTER JOIN (
		SELECT ym_magam, SUM(am_account) AS 판관비
		FROM DATAMART_DHDT_TOTAL DDT
		WHERE cd_dept_acnt = 'HD00'
			AND LEFT(cd_trial, 2) = '61'
		GROUP BY ym_magam) C
	ON A.ym_magam = C.ym_magam FULL OUTER JOIN (
		SELECT ym_magam, SUM(am_account) AS 부채
		FROM DATAMART_DHDT_TOTAL DDT
		WHERE cd_dept_acnt = 'HD00'
			AND LEFT(cd_trial, 1) = '2'
		GROUP BY ym_magam) D
	ON A.ym_magam = D.ym_magam FULL OUTER JOIN (
		SELECT ym_magam, SUM(am_account) AS 자산
		FROM DATAMART_DHDT_TOTAL DDT
		WHERE cd_dept_acnt = 'HD00'
			AND LEFT(cd_trial, 1) = '1'
		GROUP BY ym_magam) E
	ON A.ym_magam = E.ym_magam FULL OUTER JOIN (
		SELECT ym_magam, SUM(am_account) AS 차입금
		FROM DATAMART_DHDT_TOTAL DDT
		WHERE cd_dept_acnt = 'HD00'
			AND cd_trial IN('22101101', '22201101', '22101102', '22201201', '22301101', '22301201', '27101101', '27101102', '27101104', '27101201', '27301101', '27301201', '22101100', '22201100', '27101100')
		GROUP BY ym_magam) F
	ON A.ym_magam = F.ym_magam FULL OUTER JOIN (
		SELECT ym_magam, SUM(am_account) AS 유동자산
		FROM DATAMART_DHDT_TOTAL DDT
		WHERE cd_dept_acnt = 'HD00'
			AND LEFT(cd_trial, 2) IN('11', '12', '14')
		GROUP BY ym_magam) G
	ON A.ym_magam = G.ym_magam FULL OUTER JOIN (
		SELECT ym_magam, SUM(am_account) AS 유동부채
		FROM DATAMART_DHDT_TOTAL DDT
		WHERE cd_dept_acnt = 'HD00'
			AND LEFT(cd_trial, 2) IN('21', '22', '23')
		GROUP BY ym_magam) H
	ON A.ym_magam = H.ym_magam FULL OUTER JOIN (
		SELECT ym_magam, SUM(am_account) AS 이자비용
		FROM DATAMART_DHDT_TOTAL DDT
		WHERE cd_dept_acnt = 'HD00'
			AND LEFT(cd_trial, 6) = '716011'
		GROUP BY ym_magam) I
	ON A.ym_magam = I.ym_magam FULL OUTER JOIN (
		SELECT ym_magam, SUM(am_account) AS 영업외수익
		FROM DATAMART_DHDT_TOTAL DDT
		WHERE cd_dept_acnt = 'HD00'
			AND (71103101 <= CAST(cd_trial AS INT) AND CAST(cd_trial AS INT) <= 71109901)
		GROUP BY ym_magam) J
	ON A.ym_magam = J.ym_magam FULL OUTER JOIN (
		SELECT ym_magam, SUM(am_account) AS 영업외비용
		FROM DATAMART_DHDT_TOTAL DDT
		WHERE cd_dept_acnt = 'HD00'
			AND (71603101 <= CAST(cd_trial AS INT) AND CAST(cd_trial AS INT) <= 71609901)
		GROUP BY ym_magam) K
	ON A.ym_magam = K.ym_magam FULL OUTER JOIN (
		SELECT ym_magam, SUM(am_account) AS 금융수익
		FROM DATAMART_DHDT_TOTAL DDT
		WHERE cd_dept_acnt = 'HD00'
			AND (71101100 <= CAST(cd_trial AS INT) AND CAST(cd_trial AS INT) <= 71102201)
		GROUP BY ym_magam) L
	ON A.ym_magam = L.ym_magam FULL OUTER JOIN (
		SELECT ym_magam, SUM(am_account) AS 금융비용
		FROM DATAMART_DHDT_TOTAL DDT
		WHERE cd_dept_acnt = 'HD00'
			AND (71601100 <= CAST(cd_trial AS INT) AND CAST(cd_trial AS INT) <= 71602201)
		GROUP BY ym_magam) M
	ON A.ym_magam = M.ym_magam FULL OUTER JOIN (
		SELECT ym_magam, SUM(am_account) AS 기타영업외손익
		FROM DATAMART_DHDT_TOTAL DDT
		WHERE cd_dept_acnt = 'HD00'
			AND (71907008 <= CAST(cd_trial AS INT) AND CAST(cd_trial AS INT) <= 71907609)
		GROUP BY ym_magam) N
	ON A.ym_magam = N.ym_magam
)
ORDER BY A.ym_magam;