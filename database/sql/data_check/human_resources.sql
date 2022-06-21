SELECT *
FROM DATAMART_DAAV_BASEINFO_DETAIL DDBD;
--- 업데이트: 매일 0시 20분.
--- 사용 영역: 카드 지표 전체, `인원 현황`, `평균 연봉`, `보직자 현황`, `전문역량`
--- `dt_birth`: 끝 1자리: 1900년대생 남자 1 여자 2, 2000년대생 남자 3 여자 4.

SELECT *
FROM DATAMART_DAAV_BASEINFO_MONTH DDBM;
--- 업데이트: 매일 1시
--- 레코드 추가: 매달 1일에 전달의 말일 기준으로 추가.
--- 사용 영역: `월별 인건비·인원 추이`, `인당 생산성·인건비당 생산성`, `채용구분별 인원`, `근무지별 인원`, `임금피크·정년퇴직 (예측)`
--- `use_month`의 가장 큰 값 기준으로 12개월 또는 10년치 표현.
--- `use_month`: 2011년 3월부터 2019년 9월까지 3개월 단위로만 값 존재. 2019년 10월부터는 1개월 단위로 존재. `use_month`별로 전직원 레코드 포함.
--- `am_incomesum_month`: 임직원별 인건비

--## 채용구분, 연월별 인원 수
SELECT use_month, ds_emptype_bi, COUNT(*)
FROM DATAMART_DAAV_BASEINFO_MONTH DDBM
WHERE ty_count = 'Y'
GROUP BY use_month, ds_emptype_bi
ORDER BY use_month, ds_emptype_bi;

SELECT *
FROM DATAMART_DAAV_BASEINFO_YEAR DDBY
ORDER BY use_month DESC;
--- 업데이트: 매일 1시
--- 레코드 추가: 매년 4월 1일, 7월 1일, 10월 1일, 1월 1일에 전달의 말일 기준으로 추가.
--- 사용 영역: `채용구분별 인건비`, `근무지별 인건비`
--- `use_month`: 2011년 이후 매년 3, 6, 9, 12월. `use_month`별로 레코드 1개.
--- `am_regular`: 정규직 인건비
--- `am_contract`: 계약직 인건비

--## 연도별 인건비
SELECT use_month, am_total
FROM DATAMART_DAAV_BASEINFO_YEAR DDBY
WHERE RIGHT(use_month, 2) IN('12')
ORDER BY use_month DESC;

--## 연도, 근무지별 인건비
SELECT LEFT(use_month, 4), ds_tydept_bi, SUM(am_incomesum_month)
FROM DATAMART_DAAV_BASEINFO_MONTH DDBM
GROUP BY ds_tydept_bi, LEFT(use_month, 4)
ORDER BY LEFT(use_month, 4) DESC, ds_tydept_bi

SELECT *
FROM DATAMART_DAAV_BASEINFO_MONTH_FUTURE DDBMF;
--- 업데이트: 매일 1시
--- 레코드 추가: 매년 2월 1일에 5년 후 1월 31일 기준으로 추가.
--- 사용 영역: `임금피크·정년퇴직`
--- `use_month`: 2022년 이후 매년 1월.

--## 정년퇴직
SELECT use_month, COUNT(*) AS cnt
FROM datamart_daav_baseinfo_month
WHERE (RIGHT(use_month, 2) = 12 OR use_month = CONCAT(YEAR(GETDATE() - 1), MONTH(GETDATE() - 1) - 1)) AND ds_emptype_bi = '정규직' AND ds_position_bi != '임원' AND DATEDIFF(YEAR, CAST('19' + LEFT(dt_birth, 6) AS DATE), CAST(LEFT(use_month, 4) + '-' + RIGHT(use_month, 2) + '-' + '01' AS DATE)) BETWEEN 57 AND 60
GROUP BY use_month
UNION ALL
SELECT use_month, COUNT(*) AS cnt
FROM datamart_daav_baseinfo_month_future
WHERE RIGHT(use_month, 2) = '01' AND ds_emptype_bi = '정규직' AND ds_position_bi != '임원' AND DATEDIFF(YEAR, CAST('19' + LEFT(dt_birth, 6) AS DATE), CAST(CONCAT_WS('-', LEFT(use_month, 4), RIGHT(use_month, 2), '01') AS DATE)) BETWEEN 57 AND 60
GROUP BY use_month;

SELECT *
FROM DATAMART_DAAV_LICENSE DDL;
--- 업데이트: 매일 0시 20분.
--- 사용 영역: `전문역량`-`자격증`

SELECT *
FROM DATAMART_DAAV_SCHOOLCAREER DDS;
--- 업데이트: 매일 0시 20분.
--- 사용 영역: `전문역량`-`학력`

--## 학력
SELECT A.ds_degree, COUNT(*) 
FROM (
	SELECT DDS.id_sabun, ROW_NUMBER() OVER(PARTITION BY DDS.id_sabun ORDER BY DDS.cd_degree DESC) AS max_degree, DDS.ds_degree
	FROM DATAMART_DAAV_SCHOOLCAREER DDS LEFT OUTER JOIN DATAMART_DAAV_BASEINFO_DETAIL DDBD ON DDS.cd_corp = DDBD.cd_corp AND DDS.id_sabun = DDBD.id_sabun
	WHERE DDS.cd_degree IS NOT NULL AND DDBD.ds_retire != '퇴직' AND DDBD.DS_EMPTYPE_BI = '정규직') A
WHERE A.max_degree = 1
GROUP BY A.ds_degree;

SELECT *
FROM DATAMART_DAAV_ORDER DDO;
--- 업데이트: 매일 0시 20분.
--- 사용 영역: `정규직 입퇴사 추이`, `계약직 입퇴사 추이`, `정규직 입퇴사 추이 (10개년)`
--- 미래 발령은 제외됨.

--## 입퇴사 수
SELECT YEAR(dt_order), ds_emptype_bi, ds_order1, COUNT(*) AS CNT
FROM (
	SELECT *
	FROM (
		SELECT CAST(dt_order AS DATE) AS dt_order, id_sabun, ds_hname, ds_emptype_bi, ds_order1, ds_order2, ds_join_bi, ds_retire_bi
		FROM DATAMART_DAAV_ORDER DDO) A
	WHERE (ds_order1 = '입사'
		OR ds_order1 = '퇴사')
		AND dt_order > '1999-12-31'
		AND (ds_retire_bi != ''
			OR ds_join_bi != '')
) A
GROUP BY YEAR(dt_order), ds_emptype_bi, ds_order1
ORDER BY YEAR(dt_order), ds_emptype_bi, ds_order1

--## 2021년 현장 man-months
-- 2021.01.01: CAST(DATEADD(YEAR, DATEDIFF(YEAR, 0, GETDATE()), 0) AS DATE),
-- 2021.12.31: CAST(DATEADD(YEAR, DATEDIFF(YEAR, 0, GETDATE()) + 1, -1) AS DATE)
SELECT *, DATEDIFF(DAY, dt_start, dt_end) AS days
FROM (
	SELECT id_sabun, ds_hname, ds_order1, ds_dept,
		CAST(CASE WHEN dt_order >= '2021-01-01' THEN dt_order ELSE '2021-01-01' END AS DATE) AS dt_start,
		CAST(CASE WHEN dt_orderend <= '2022-01-01' THEN dt_orderend ELSE '2022-01-01' END AS DATE) AS dt_end
	FROM
		(SELECT id_sabun, ds_hname, ds_order1, ds_dept, CAST(dt_order AS DATE) AS dt_order, CAST(dt_orderend AS DATE) AS dt_orderend
		FROM DATAMART_DAAV_ORDER DDO
		WHERE cd_corp = 'A101' AND ds_tydept_bi = '현장' and ds_order1 != '퇴사' and ds_order1 != '유급휴직' and ds_order1 != '무급휴직') A) B
WHERE dt_start <= dt_end
ORDER BY id_sabun

SELECT id_sabun, ds_hname, ds_dept, SUM(days)
FROM (
	SELECT *, DATEDIFF(DAY, dt_start, dt_end) AS days
	FROM (
		SELECT id_sabun, ds_hname, ds_order1, ds_dept,
			CAST(CASE WHEN dt_order >= '2021-01-01' THEN dt_order ELSE '2021-01-01' END AS DATE) AS dt_start,
			CAST(CASE WHEN dt_orderend <= '2022-01-01' THEN dt_orderend ELSE '2022-01-01' END AS DATE) AS dt_end
		FROM
			(SELECT id_sabun, ds_hname, ds_order1, ds_dept, CAST(dt_order AS DATE) AS dt_order, CAST(dt_orderend AS DATE) AS dt_orderend
			FROM DATAMART_DAAV_ORDER DDO
			WHERE cd_corp = 'A101' AND ds_tydept_bi = '현장' and ds_order1 != '퇴사' and ds_order1 != '유급휴직' and ds_order1 != '무급휴직') A) B
	WHERE dt_start <= dt_end) C
GROUP BY id_sabun, ds_hname, ds_dept
ORDER BY id_sabun

SELECT SUM(interv)
FROM (
    SELECT DATEDIFF(DAY, dt_start, dt_end) AS interv
    FROM ( 
        SELECT
        	CAST(CASE WHEN dt_order >= '2021-01-01' THEN dt_order ELSE '2021-01-01' END AS DATE) AS dt_start,
            CAST(CASE WHEN dt_orderend <= '2021-12-31' THEN dt_orderend ELSE '2021-12-31' END AS DATE) AS dt_end
        FROM 
        	(SELECT cd_corp, ds_tydept_bi, ds_order1, CAST(dt_order AS DATE) AS dt_order, CAST(dt_orderend AS DATE) AS dt_orderend
        	FROM DATAMART_DAAV_ORDER DDO
        	WHERE cd_corp = 'A101' AND ds_tydept_bi = '현장' and ds_order1 != '퇴사' and ds_order1 != '유급휴직' and ds_order1 != '무급휴직') A) B) C
WHERE interv > 0

SELECT *
FROM DATAMART_DAUV_ANNUALINCOME DDA;
--- 업데이트: 매일 0시 20분.
--- 사용 영역: `평균 연봉`

SELECT *
FROM DATAMART_DAUV_ANNUALINCOME DDA
WHERE id_sabun = '6363';

--## 근무지, 채용구분별 평균 연봉
SELECT DDBD.ds_tydept_bi, DDBD.ds_emptype_bi, SUM(DDA.am_salary) AS '연봉 합', COUNT(DISTINCT DDBD.id_sabun) AS '인원 수', ROUND(SUM(DDA.am_salary)/COUNT(*)/10000, 0) AS '평균'
FROM DATAMART_DAUV_ANNUALINCOME DDA LEFT OUTER JOIN DATAMART_DAAV_BASEINFO_DETAIL DDBD ON DDA.cd_corp = DDBD.cd_corp AND DDA.id_sabun = DDBD.id_sabun
WHERE DDA.cd_corp = 'A101' AND DDBD.ds_retire != '퇴직'
GROUP BY ds_tydept_bi, ds_emptype_bi
ORDER BY ds_tydept_bi, ds_emptype_bi;

--## 직급, 채용구분별 평균 연봉
SELECT DDBD.ds_position_bi, DDBD.ds_emptype_bi, SUM(DDA.am_salary) AS '연봉 합', COUNT(DISTINCT DDBD.id_sabun) AS '인원 수', ROUND(SUM(DDA.am_salary)/COUNT(*)/10000, 0) AS '평균'
FROM DATAMART_DAUV_ANNUALINCOME DDA LEFT OUTER JOIN DATAMART_DAAV_BASEINFO_DETAIL DDBD ON DDA.cd_corp = DDBD.cd_corp AND DDA.id_sabun = DDBD.id_sabun
WHERE DDA.cd_corp = 'A101' AND DDBD.ds_retire != '퇴직'
GROUP BY ds_position_bi, ds_emptype_bi
ORDER BY ds_position_bi, ds_emptype_bi;

--## 직원별 연봉
SELECT *
FROM (
	SELECT DDBD.id_sabun, DDBD.ds_hname, SUM(DDA.am_salary) AS sal
	FROM DATAMART_DAUV_ANNUALINCOME DDA LEFT OUTER JOIN DATAMART_DAAV_BASEINFO_DETAIL DDBD ON DDA.cd_corp = DDBD.cd_corp AND DDA.id_sabun = DDBD.id_sabun
	WHERE DDA.cd_corp = 'A101' AND DDBD.ds_retire != '퇴직'
	GROUP BY DDBD.id_sabun, DDBD.ds_hname) A
--WHERE ds_hname = ''
ORDER BY sal DESC;

SELECT *
FROM DATAMART_DHDT_TOTAL DDT
ORDER BY ym_magam DESC;
--- 업데이트: 협의 필요
--- 사용 영역: `인당 생산성·인건비당 생산성`

--## 매출액
SELECT ym_magam, CAST(ROUND(SUM(am_account)/100000000, 0) AS BIGINT)
FROM datamart_dhdt_total
WHERE cd_corp = 'A101' AND cd_dept_acnt = 'HD00' AND LEFT(cd_trial, 2) = '41' AND RIGHT(ym_magam, 2) IN('03', '06', '09', '12')
GROUP BY ym_magam
ORDER BY ym_magam;

SELECT *
FROM 프로젝트목록_현장수추이_월별_언피벗 프현월언;
--- 업데이트: 매일 4시.
--- 사용 영역: `현장 수`
--- 지난달은 말일 기준, 이달은 오늘 기준으로 현장 수 카운트

--## 자격증 취득 이력
SELECT DISTINCT cd_corp, id_sabun, ds_license, no_license, ds_organ, CAST(dt_acquire AS DATE) AS dt_acquire, ds_license_bi
FROM DATAMART_DAAV_LICENSE DDL
WHERE dt_acquire > '2009-12-31'
ORDER BY cd_corp, id_sabun, dt_acquire;

-## 학력 (석사, 박사)
SELECT ds_degree, ds_bonbu, ds_dept, ds_hname
FROM (
	SELECT DDBD.ds_bonbu, DDBD.ds_dept, DDBD.ds_hname, DDS.ds_degree, ROW_NUMBER() OVER(PARTITION BY DDS.id_sabun ORDER BY DDS.cd_degree DESC) AS max_degree
	FROM DATAMART_DAAV_SCHOOLCAREER DDS LEFT OUTER JOIN DATAMART_DAAV_BASEINFO_DETAIL DDBD
		ON DDS.cd_corp = DDBD.cd_corp AND DDS.id_sabun = DDBD.id_sabun
	WHERE DDS.cd_degree IS NOT NULL
		AND DDBD.ds_retire != '퇴직'
		AND DDBD.DS_EMPTYPE_BI = '정규직') A
WHERE max_degree = 1
	AND ds_degree IN('석사', '박사')
ORDER BY ds_degree, ds_bonbu, ds_dept, ds_hname

-- ## 발령 이력
SELECT ds_corp, id_sabun, ds_hname, CAST(dt_order AS DATE) AS dt_order, CAST(dt_orderend AS DATE) AS dt_orderend, ds_order1, ds_order2, ds_dept, ds_remark
FROM DATAMART_DAAV_ORDER DDO
WHERE ds_hname IS NOT NULL;

--## 2021년 부서, 직원별 근무 이력
SELECT *, DATEDIFF(DAY, dt_start, dt_end) AS days
FROM (
	SELECT id_sabun, ds_hname, ds_order1, ds_dept,
		CAST(CASE WHEN dt_order >= '2021-01-01' THEN dt_order ELSE '2021-01-01' END AS DATE) AS dt_start,
		CAST(CASE WHEN dt_orderend <= '2022-01-01' THEN dt_orderend ELSE '2022-01-01' END AS DATE) AS dt_end
	FROM
		(SELECT id_sabun, ds_hname, ds_order1, ds_dept, CAST(dt_order AS DATE) AS dt_order, CAST(dt_orderend AS DATE) AS dt_orderend
		FROM DATAMART_DAAV_ORDER DDO
		WHERE cd_corp = 'A101' and ds_order1 != '퇴사' and ds_order1 != '유급휴직' and ds_order1 != '무급휴직') A) B
WHERE dt_start <= dt_end
ORDER BY id_sabun

--## 2021년 부서, 직원별 근무 일수 합계
SELECT id_sabun, ds_hname, ds_dept, SUM(days) AS days
FROM (
	SELECT *, DATEDIFF(DAY, dt_start, dt_end) AS days
	FROM (
		SELECT id_sabun, ds_hname, ds_order1, ds_dept,
			CAST(CASE WHEN dt_order >= '2021-01-01' THEN dt_order ELSE '2021-01-01' END AS DATE) AS dt_start,
			CAST(CASE WHEN dt_orderend <= '2022-01-01' THEN dt_orderend ELSE '2022-01-01' END AS DATE) AS dt_end
		FROM
			(SELECT id_sabun, ds_hname, ds_order1, ds_dept, CAST(dt_order AS DATE) AS dt_order, CAST(dt_orderend AS DATE) AS dt_orderend
			FROM DATAMART_DAAV_ORDER DDO
			WHERE cd_corp = 'A101' AND ds_order1 != '퇴사' AND ds_order1 != '유급휴직' AND ds_order1 != '무급휴직' AND ds_hname IS NOT NULL AND ds_dept IS NOT NULL) A) B
	WHERE dt_start <= dt_end) C
GROUP BY id_sabun, ds_hname, ds_dept
ORDER BY id_sabun

--## 입퇴사 이력
SELECT *
FROM (
	SELECT CAST(dt_order AS DATE) AS dt_order, id_sabun, ds_hname, ds_emptype_bi, ds_order1, ds_order2, ds_join_bi, ds_retire_bi
	FROM DATAMART_DAAV_ORDER DDO) A
WHERE (ds_order1 = '입사'
	OR ds_order1 = '퇴사')
	AND dt_order > '1999-12-31'
	AND (ds_retire_bi != ''
		OR ds_join_bi != '')
ORDER BY dt_order, ds_emptype_bi, ds_order1, ds_order2, id_sabun

--## 연도별 임금피크 대상자
SELECT *
FROM (
	SELECT CAST(LEFT(use_month, 4) AS INT) AS yr, ds_bonbu, ds_dept, ds_hname
	FROM DATAMART_DAAV_BASEINFO_MONTH DDBM
	WHERE (RIGHT(use_month, 2) = 12
		OR use_month = CONCAT(YEAR(GETDATE() - 1), MONTH(GETDATE() - 1) - 1))
		AND ds_emptype_bi = '정규직'
		AND ds_position_bi != '임원'
		AND DATEDIFF(YEAR, CAST('19' + LEFT(dt_birth, 6) AS DATE), CAST(LEFT(use_month, 4) + '-' + RIGHT(use_month, 2) + '-' + '01' AS DATE)) BETWEEN 57 AND 60
	UNION ALL
	SELECT CAST(LEFT(use_month, 4) AS INT) AS yr, ds_bonbu, ds_dept, ds_hname
	FROM DATAMART_DAAV_BASEINFO_MONTH_FUTURE DDBMF
	WHERE RIGHT(use_month, 2) = '01'
		AND ds_emptype_bi = '정규직'
		AND ds_position_bi != '임원'
		AND DATEDIFF(YEAR, CAST('19' + LEFT(dt_birth, 6) AS DATE), CAST(CONCAT_WS('-', LEFT(use_month, 4), RIGHT(use_month, 2), '01') AS DATE)) BETWEEN 57 AND 60) A
WHERE yr >= 2018
ORDER BY yr, ds_bonbu, ds_dept, ds_hname
