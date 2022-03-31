--## 올해 월별 소화 실적 합계, 소화 계획 합계
SELECT 현장코드, 현장명, CAST(날짜 AS DATE) AS 날짜, ROUND(SUM(CAST(실적 AS FLOAT)), 0) AS 실적, ROUND(SUM(CAST(계획 AS FLOAT)), 0) AS 계획
FROM DATAMART_COMBBZPLN_ALL DCA
WHERE 날짜 > '2019-12-31'
	AND 실적 IS NOT NULL
GROUP BY 날짜, 현장코드, 현장명
ORDER BY 현장코드, 날짜;

--## 계약일보
SELECT 코드, 현장명, 분양구분, 분양종류, 동, 호수, 평형, 분양가, 계약일자, 해약일자
FROM 계약일보_작업;

--## 자격증 취득 이력
SELECT DISTINCT cd_corp, id_sabun, ds_license, no_license, ds_organ, CAST(dt_acquire AS DATE) AS dt_acquire, ds_license_bi
FROM DATAMART_DAAV_LICENSE DDL
WHERE dt_acquire > '2009-12-31'
ORDER BY cd_corp, id_sabun, dt_acquire;

--## 학력 (석사, 박사)
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

--## 발령 이력
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

--## Change Management
SELECT *
FROM CM현황_누계 C누;

--## 수주
SELECT 본부, 팀명, 담당자명, 접수일, 사업유형, 수주사업지명, 사업지구분1, 사업지구분2, 사업장주소1, 사업장상세주소, 사업비유형코드, 진행현황코드, '진행현황(상세)', 처리결과, 등록자, 등록일자, 수정자, 수정일자
FROM 수주정보_작업
ORDER BY 접수일;

--## 공수
SELECT 연월, 부서명, 이름, 요청부서명, 대상프로젝트코드, 대상프로젝트명, lvl1, lvl2, lvl3, lvl4, CAST(시작일 AS DATE) AS 시작일, CAST(종료일 AS DATE) AS 종료일, 변경요청투입률_MH, 지원만족도, 요청만족도, 주요업무내용
FROM 공수데이터리스트;

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


SELECT use_month, ds_bonbu, ds_dept, id_sabun
FROM datamart_daav_baseinfo_month
WHERE (RIGHT(use_month, 2) = 12 OR use_month = CONCAT(YEAR(GETDATE() - 1), MONTH(GETDATE() - 1) - 1)) AND ds_emptype_bi = '정규직' AND ds_position_bi != '임원' AND DATEDIFF(YEAR, CAST('19' + LEFT(dt_birth, 6) AS DATE), CAST(LEFT(use_month, 4) + '-' + RIGHT(use_month, 2) + '-' + '01' AS DATE)) BETWEEN 57 AND 60
--GROUP BY use_month
ORDER BY use_month

SELECT *
FROM datamart_daav_baseinfo_month;

SELECT *
FROM datamart_daav_baseinfo_moNTH_FUTURE;

UNION ALL
SELECT use_month, COUNT(*) AS cnt
FROM datamart_daav_baseinfo_month_future
WHERE RIGHT(use_month, 2) = '01' AND ds_emptype_bi = '정규직' AND ds_position_bi != '임원' AND DATEDIFF(YEAR, CAST('19' + LEFT(dt_birth, 6) AS DATE), CAST(CONCAT_WS('-', LEFT(use_month, 4), RIGHT(use_month, 2), '01') AS DATE)) BETWEEN 57 AND 60
GROUP BY use_month
ORDER BY use_month;