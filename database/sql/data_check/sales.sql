SELECT *
FROM DATAMART_DIFV_PL_FORCAST DDPF;
--- 업데이트: 매달 21일 새벽
--- 레코드 추가: 매달 21일에 전달 데이터 추가
--- 사용 영역: 카드 지표 전체, `누계 수주 실적`, `부문별 수주 현황`, `누계 공급 실적`, `부문별 공급 현황`, `공급 추이`, `누계 판매 실적`, `부문별 판매 현황`, `판매 추이`, `팀별 영업 실적` 대시보드 전체

--## 본부별 수주, 공급, 판매 연간 계획, 누계 계획, 연간 실적
SELECT A.ds_item, A.ds_bonbu, A.연간_계획, B.누계_계획, A.연간_실적 
FROM (
	SELECT ds_item, ds_bonbu, SUM(am_plan) AS 연간_계획, SUM(am_siljuk) AS 연간_실적
	FROM DATAMART_DIFV_PL_FORCAST DDPF
	WHERE ds_item != '수주잔고' AND YEAR = 2021
	GROUP BY ds_item, ds_bonbu) AS A,
	(SELECT ds_item, ds_bonbu, SUM(am_plan) AS 누계_계획
	FROM DATAMART_DIFV_PL_FORCAST DDPF
	WHERE ds_item != '수주잔고' AND YEAR = 2021
--		AND MONTH <= MONTH(GETDATE()) - 2
	GROUP BY ds_item, ds_bonbu) AS B
WHERE A.ds_item = B.ds_item AND A.ds_bonbu = B.ds_bonbu
ORDER BY ds_item, ds_bonbu;

--## 본부, 팀별 수주, 공급, 판매 연간 계획, 누계 계획, 연간 실적
SELECT A.ds_item, A.ds_bonbu, A.ds_spnc_dept, A.연간_계획, B.누계_계획, A.연간_실적 
FROM (
	SELECT ds_item, ds_bonbu, ds_spnc_dept, SUM(am_plan) AS 연간_계획, SUM(am_siljuk) AS 연간_실적
	FROM DATAMART_DIFV_PL_FORCAST DDPF
	WHERE ds_item != '수주잔고' AND YEAR = 2021
	GROUP BY ds_item, ds_bonbu, ds_spnc_dept) AS A,
	(SELECT ds_item, ds_bonbu, ds_spnc_dept, SUM(am_plan) AS 누계_계획
	FROM DATAMART_DIFV_PL_FORCAST DDPF
	WHERE ds_item != '수주잔고' AND YEAR = 2021 AND MONTH <= MONTH(GETDATE()) - 2
	GROUP BY ds_item, ds_bonbu, ds_spnc_dept) AS B
WHERE A.ds_item = B.ds_item AND A.ds_bonbu = B.ds_bonbu AND A.ds_spnc_dept = B.ds_spnc_dept;

--## 현장별 판매 연간 실적
SELECT *
FROM (
	SELECT ds_bonbu, ds_spnc_dept, cd_site, ds_site, SUM(am_siljuk) AS 연간_실적
	FROM DATAMART_DIFV_PL_FORCAST DDPF
	WHERE ds_item = '판매' AND YEAR = 2021
	GROUP BY ds_bonbu, ds_spnc_dept, cd_site, ds_site) A
WHERE 연간_실적 != 0;

SELECT DISTINCT year
FROM DATAMART_DIFV_PL_FORCAST DDPF
WHERE ds_item = '판매'

SELECT *
FROM DATAMART_DIFV_FORCAST_REMAIN DDFR;
--- 업데이트: 	매년 1, 4, 7, 10월 21일 ?시
--- 레코드 추가: 매년 1, 4, 7, 10월 21일에 전달까지 3개월치 데이터 추가
--- 사용 영역: `수주잔고 추이`

--## 수주잔고
SELECT ym_siljuk, ROUND(SUM(am_siljuk)/100000000, 0)
FROM DATAMART_DIFV_FORCAST_REMAIN DDFR
GROUP BY ym_siljuk
ORDER BY ym_siljuk;