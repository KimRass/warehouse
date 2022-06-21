SELECT *
FROM 공수데이터리스트 공;
--- 업데이트: 매달 8일

SELECT *
FROM 공수데이터리스트 공
WHERE 업무코드내용 LIKE '%디지털%';

--## 인원 수
SELECT 부서명, COUNT(DISTINCT 사번)
FROM 공수데이터리스트 공
WHERE 연월 = '202111'
GROUP BY 부서명

--## 연월, 구분별 인원, 공수 투입량 추이
SELECT 연월, 구분, COUNT(DISTINCT 사번), SUM(투입률_MH) AS mh
FROM (
	SELECT *, CASE WHEN 부서명 IN('상품기획팀', '실시설계팀', '설계그룹', '설계그룹(개발영업)', '상품설계팀') THEN '설계' WHEN 부서명 IN('예산견적팀', '일반견적팀', '견적그룹', '견적그룹(개발영업)') THEN '견적' END AS 구분
	FROM 공수데이터리스트 공
	WHERE 부서명 IN('상품기획팀', '실시설계팀', '설계그룹', '설계그룹(개발영업)', '상품설계팀', '예산견적팀', '일반견적팀', '견적그룹', '견적그룹(개발영업)')) A
GROUP BY 연월, 구분
ORDER BY 연월, 구분;

--## 부서, 요청부서별 공수 투입량
SELECT 부서명, 요청부서명, SUM(투입률_MH) AS mh
FROM 공수데이터리스트 공
WHERE 연월 = '202111' AND 부서명 IN('상품기획팀', '실시설계팀', '설계그룹', '설계그룹(개발영업)', '상품설계팀', '예산견적팀', '일반견적팀', '견적그룹', '견적그룹(개발영업)')
GROUP BY 부서명, 요청부서명;

--## 요청부서, 구분별 공수 투입량
SELECT 요청부서명, 구분, SUM(투입률_MH) AS mh
FROM (
	SELECT *, CASE WHEN 부서명 IN('상품기획팀', '실시설계팀', '설계그룹', '설계그룹(개발영업)', '상품설계팀') THEN '설계' WHEN 부서명 IN('예산견적팀', '일반견적팀', '견적그룹', '견적그룹(개발영업)') THEN '견적' END AS 구분
	FROM 공수데이터리스트 공
	WHERE 연월 = '202111' AND 부서명 IN('상품기획팀', '실시설계팀', '설계그룹', '설계그룹(개발영업)', '상품설계팀', '예산견적팀', '일반견적팀', '견적그룹', '견적그룹(개발영업)')) A
GROUP BY 요청부서명, 구분
ORDER BY mh DESC;

--## 구분별 공수 투입량
SELECT 구분, 중분류, SUM(투입률_MH) AS mh
FROM (
	SELECT *, CASE WHEN 부서명 IN('상품기획팀', '실시설계팀', '설계그룹', '설계그룹(개발영업)', '상품설계팀') THEN '설계' WHEN 부서명 IN('예산견적팀', '일반견적팀', '견적그룹', '견적그룹(개발영업)') THEN '견적' END AS 구분
	FROM 공수데이터리스트 공
	WHERE 연월 = '202111' AND 부서명 IN('상품기획팀', '실시설계팀', '설계그룹', '설계그룹(개발영업)', '상품설계팀', '예산견적팀', '일반견적팀', '견적그룹', '견적그룹(개발영업)')) A
GROUP BY 구분, 중분류
ORDER BY 구분, mh DESC, 중분류;

--## Man-Hour 연계
SELECT *
FROM DATAMART_DAAV_ORDER DDO
WHERE id_sabun = '5717';

SELECT *
FROM (
	SELECT id_sabun, ds_hname, ds_order1, ds_order2, ds_dept, ds_tydept_bi, dt_order, dt_orderend, RANK() OVER(PARTITION BY id_sabun ORDER BY dt_order DESC) rnk
	FROM DATAMART_DAAV_ORDER
	WHERE cd_corp = 'A101' AND dt_order > '20150101' AND ds_order2 = '겸직') A
WHERE rnk = 1
ORDER BY id_sabun

--## 역량 총원
SELECT DISTINCT ds_dept
FROM DATAMART_DAAV_BASEINFO_DETAIL DDBD
WHERE cd_corp = 'A101'
	AND ds_dept LIKE '%설계%'
	AND ds_retire != '퇴직';

SELECT *
FROM DATAMART_DAAV_BASEINFO_DETAIL DDBD
WHERE ds_dept IN('실시설계팀', '설계그룹(개발영업)', '상품기획팀')
ORDER BY ds_dept

--SELECT CASE ds_upper_target_dept WHEN '현장' THEN '건설본부' ELSE ds_upper_target_dept END AS upper, ds_upper_dept, ds_site, SUM(mh_gongsu) AS mh
--FROM 공수데이터리스트_부서
--WHERE dt_yymm = '202110' AND ds_upper_target_dept IS NOT NULL
--GROUP BY ds_upper_dept, ds_site, ds_upper_target_dept
--
--SELECT *
--FROM (
--	SELECT CASE ds_upper_target_dept WHEN '현장' THEN '건설본부' ELSE ds_upper_target_dept END AS upper, ds_upper_dept, ds_site, SUM(mh_gongsu) AS mh
--	FROM 공수데이터리스트_부서 DM
--	WHERE dt_yymm = '202110' AND ds_upper_target_dept IS NOT NULL
--	GROUP BY ds_upper_dept, ds_site, CASE ds_upper_target_dept WHEN '현장' THEN '건설본부' ELSE ds_upper_target_dept END) A
--ORDER BY upper, ds_upper_dept, ds_site

--## 본부별 공수 투입량
--SELECT B.ds_bonbu, B.ds_dept, C.ds_bonbu, SUM(mh)
--FROM (
--	SELECT 부서명, 요청부서명, SUM(투입률_MH) AS mh
--	FROM 공수데이터리스트 공
--	WHERE 연월 = '202110' AND 부서명 IN('상품기획팀', '실시설계팀', '설계그룹(개발영업)', '예산견적팀', '일반견적팀', '견적그룹(개발영업)')
--	GROUP BY 부서명, 요청부서명) A LEFT OUTER JOIN (
--		SELECT DISTINCT ds_dept, ds_bonbu
--		FROM DATAMART_DAAV_BASEINFO_DETAIL DDBD) B ON A.부서명 = B.ds_dept
--	LEFT OUTER JOIN (
--		SELECT DISTINCT ds_dept, ds_bonbu
--		FROM DATAMART_DAAV_BASEINFO_DETAIL DDBD) C ON A.요청부서명 = C.ds_dept
--WHERE C.ds_bonbu IN('개발영업본부', '건설본부')
--GROUP BY B.ds_bonbu, B.ds_dept, C.ds_bonbu;