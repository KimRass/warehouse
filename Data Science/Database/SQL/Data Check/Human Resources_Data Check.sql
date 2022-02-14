SELECT * 
FROM INFORMATION_SCHEMA.TABLES
ORDER BY table_type, table_name;

SELECT *
FROM DATAMART_DAAV_BASEINFO_DETAIL DDBD;
--- 업데이트: 매일 0시 20분.
--- 사용 영역: 카드 지표 전체, `인원 현황`, `평균 연봉`, `보직자 현황`, `전문역량`
--- `dt_birth`: 끝 1자리: 1900년대생 남자 1 여자 2, 2000년대생 남자 3 여자 4.

SELECT *
FROM DATAMART_DAAV_BASEINFO_DETAIL DDBD
WHERE ds_dept IN('상품기획팀', '실시설계팀', '설계그룹', '설계그룹(개발영업)', '상품설계팀', '예산견적팀', '일반견적팀', '견적그룹', '견적그룹(개발영업)') AND ds_retire = '재직' AND ds_emptype = '일반직'
ORDER BY ds_dept;

SELECT ds_dept, COUNT(DISTINCT id_sabun) 
FROM DATAMART_DAAV_BASEINFO_DETAIL DDBD
WHERE ds_dept IN('상품기획팀', '실시설계팀', '설계그룹', '설계그룹(개발영업)', '상품설계팀', '예산견적팀', '일반견적팀', '견적그룹', '견적그룹(개발영업)') AND ds_retire = '재직' AND ds_emptype = '일반직'
GROUP BY ds_dept
ORDER BY ds_dept;

SELECT *
FROM DATAMART_DAAV_BASEINFO_DETAIL DDBD
WHERE ds_dept LIKE '%설계%';

SELECT *
FROM DATAMART_DAAV_BASEINFO_MONTH DDBM;
--- 업데이트: 매일 1시
--- 레코드 추가: 매달 1일에 전달의 말일 기준으로 추가.
--- 사용 영역: `월별 인건비·인원 추이`, `인당 생산성·인건비당 생산성`, `채용구분별 인원`, `근무지별 인원`, `임금피크·정년퇴직 (예측)`
--- `use_month`의 가장 큰 값 기준으로 12개월 또는 10년치 표현.
--- `use_month`: 2011년 3월부터 2019년 9월까지 3개월 단위로만 값 존재. 2019년 10월부터는 1개월 단위로 존재. `use_month`별로 전직원 레코드 포함.
--- `am_incomesum_month`: 임직원별 인건비

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

--## 연도별 인건비
SELECT LEFT(use_month, 4), SUM(am_incomesum_month)
FROM DATAMART_DAAV_BASEINFO_MONTH DDBM
GROUP BY LEFT(use_month, 4)
ORDER BY LEFT(use_month, 4) DESC

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

--## 2021년 현장 man-months 합
-- 2021.01.01: CAST(DATEADD(YEAR, DATEDIFF(YEAR, 0, GETDATE()), 0) AS DATE),
-- 2021.12.31: CAST(DATEADD(YEAR, DATEDIFF(YEAR, 0, GETDATE()) + 1, -1) AS DATE)
SELECT SUM(interv)
FROM (
    SELECT DATEDIFF(DAY, dt_start, dt_end) AS interv
    FROM ( 
        SELECT
        	CAST(CASE WHEN dt_order >= DATEADD(YEAR, DATEDIFF(YEAR, 0, GETDATE()), 0) THEN dt_order ELSE DATEADD(YEAR, DATEDIFF(YEAR, 0, GETDATE()), 0) END AS DATE) AS dt_start,
            CAST(CASE WHEN dt_orderend <= DATEADD(YEAR, DATEDIFF(YEAR, 0, GETDATE()) + 1, -1) THEN dt_orderend ELSE DATEADD(YEAR, DATEDIFF(YEAR, 0, GETDATE()) + 1, -1) END AS DATE) AS dt_end
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

--## 평균 연봉
SELECT SUM(DDA.am_salary), COUNT(DISTINCT DDBD.id_sabun), ROUND(SUM(DDA.am_salary)/COUNT(*)/10000, 0), DDBD.ds_emptype_bi, DDBD.ds_tydept_bi
FROM DATAMART_DAUV_ANNUALINCOME DDA LEFT OUTER JOIN DATAMART_DAAV_BASEINFO_DETAIL DDBD ON DDA.cd_corp = DDBD.cd_corp AND DDA.id_sabun = DDBD.id_sabun
WHERE DDA.cd_corp = 'A101' AND DDBD.ds_retire != '퇴직'
GROUP BY ds_emptype_bi, ds_tydept_bi;

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