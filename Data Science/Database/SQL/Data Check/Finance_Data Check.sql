SELECT *
FROM DATAMART_DFCT_CHAIP_YEGUM DDCY
ORDER BY CAST(dt_kijun AS DATE);
--- 업데이트: 매달 1일 2시
--- 사용 영역: `보유현금`, `차입금`, `순현금`, `차입금 및 운용예금`

--## 연월별 보유현금, 순현금, 차입금
SELECT *
FROM (
	SELECT CAST(dt_kijun AS DATE) dt, gubun, SUM(am_account) summ
	FROM DATAMART_DFCT_CHAIP_YEGUM DDCY
	WHERE cd_corp = 'A101'
	GROUP BY CAST(dt_kijun AS DATE), gubun) A
ORDER BY dt;

SELECT *
FROM DATAMART_DFMT_CREDIT DDC;
--- 업데이트: 매달 15일 2시
--- 사용 영역: `신용등급`(BAN), `신용등급`

SELECT *
FROM DATAMART_DFMT_STOCK DDS;
--- 업데이트: 매달 1일 2시
--- 사용 영역: `주가`

SELECT *
FROM DATAMART_DFMT_WOOBAL DDW;
--- 업데이트: 매달 15일 2시
--- 사용 영역: `우발채무현황`

--## 연월별 지급보증, 자금보충, 중도금대출, 책임준공
SELECT dt, ROUND(summ, 0), ROUND(SUM(summ) OVER(PARTITION BY dt), 0) AS tot
FROM (
	SELECT CAST(dt_kijun AS DATE) dt, ds_gubun, SUM(am_account)/100000000 AS summ
	FROM DATAMART_DFMT_WOOBAL DDW
	WHERE cd_corp = 'A101'
	GROUP BY CAST(dt_kijun AS DATE), ds_gubun) A
ORDER BY dt;