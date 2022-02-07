SELECT *
FROM DATAMART_DHDT_TOTAL DDT;
--- ������Ʈ: �Ŵ� 20�� 4��?
--- ��� ����: ī�� ��ǥ ��ü, ī�� ��ǥ-`��������� ����` ��ü, `���� ����`, `���� �������ͷ�`, `���� ��������`, `���� �������ͷ�`, `���� ����`, `��� ���� ����`-(`�����`, `��������`), `�ֿ� �ι��� ����`, `�ֿ� �ι��� �������ͷ�`, `���Ͱ�꼭`, `�繫����ǥ`, `��ä����`, `���Ա�������`, `��������`, `���ں������`

--## ������ �����, �������, �ǰ���, ��������
SELECT yr, sales, sales_cost, fee, sales - sales_cost - fee AS profit
FROM (
	SELECT yr, ROUND(SUM(c1)/100000000, 0) AS sales, ROUND(SUM(c2)/100000000, 0) AS sales_cost, ROUND(SUM(c3)/100000000, 0) AS fee
	FROM (
		SELECT LEFT(ym_magam, 4) AS yr, CASE WHEN LEFT(cd_trial, 2) = '41' THEN am_account_wol END AS c1, CASE WHEN LEFT(cd_trial, 2) = '46' THEN am_account_wol END AS c2, CASE WHEN LEFT(cd_trial, 2) = '61' THEN am_account_wol END AS c3
		FROM DATAMART_DHDT_TOTAL DDT
		WHERE cd_dept_acnt = 'HD00') A
	GROUP BY yr) B;

--## ���� ���� ��������
SELECT ym_magam, SUM(sales) OVER(ORDER BY ym_magam ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW), SUM(sales_cost) OVER(ORDER BY ym_magam ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW), SUM(fee) OVER(ORDER BY ym_magam ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW), SUM(sales - sales_cost - fee) OVER(ORDER BY ym_magam ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS profit
FROM (
	SELECT ym_magam, ROUND(SUM(c1)/100000000, 0) AS sales, ROUND(SUM(c2)/100000000, 0) AS sales_cost, ROUND(SUM(c3)/100000000, 0) AS fee
	FROM (
		SELECT ym_magam, CASE WHEN LEFT(cd_trial, 2) = '41' THEN am_account_wol END AS c1, CASE WHEN LEFT(cd_trial, 2) = '46' THEN am_account_wol END AS c2, CASE WHEN LEFT(cd_trial, 2) = '61' THEN am_account_wol END AS c3
		FROM DATAMART_DHDT_TOTAL DDT
		WHERE cd_dept_acnt = 'HD00' AND LEFT(ym_magam, 4) = 2021) A
	GROUP BY ym_magam) B
ORDER BY ym_magam;

--## ������ �ڻ�, ��ä, �ں�, ��ä����
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
FROM DATAMART_DHDT_ACNT_GROUP DDAG;
--- ������Ʈ: �Ŵ� 20�� 4��?
--- ��� ����:  ī�� ��ǥ-`��������� ����` ��ü, `�ֿ� �ι��� ����`, `�ֿ� �ι��� �������ͷ�`

SELECT *
FROM DATAMART_DHDT_TOTAL_GROUP DDTG;
--- ������Ʈ: �Ŵ� 20�� 4��?
--- ��� ����: `�ֿ� �ι��� ����`, `�ֿ� �ι��� �������ͷ�`

--## �׷�, ������ �����
SELECT ds_group, LEFT(max_yr, 4) AS yr, sales
FROM (
	SELECT DDT.ym_magam, MAX(DDT.ym_magam) OVER(PARTITION BY LEFT(DDT.ym_magam, 4)) AS max_yr, DDTG.ds_group, ROUND(SUM(am_account_wol)/100000000, 0) AS sales
	FROM DATAMART_DHDT_TOTAL DDT INNER JOIN DATAMART_DHDT_ACNT_GROUP DDAG ON DDT.cd_corp = DDAG.cd_corp AND DDT.cd_dept_acnt = DDAG.cd_dept_acnt INNER JOIN DATAMART_DHDT_TOTAL_GROUP DDTG ON DDAG.cd_group = DDTG.cd_group
	WHERE LEFT(DDT.cd_trial, 2) = '41' AND ds_group != '��Ÿ'
	GROUP BY DDT.ym_magam, DDTG.ds_group) A
WHERE ym_magam = max_yr
ORDER BY ds_group, yr;

SELECT *
FROM DATAMART_DIFV_PL DDP;
--- ������Ʈ: �Ŵ� 20�� 4��?
--- ��� ����: `��� ���� ����`-(`����� ��ȹ`, `�������� ��ȹ`)

--## ���� ���� ����� ��ȹ
SELECT MONTH, ROUND(SUM(pl) OVER(ORDER BY CAST(MONTH AS INT) ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)/100000000, 0)
FROM (
	SELECT MONTH, SUM(am_plan) AS pl
	FROM DATAMART_DIFV_PL DDP
	WHERE ds_item = '�����' AND YEAR = 2021
	GROUP BY MONTH) AS A
ORDER BY CAST(MONTH AS INT);

--## ���� ���� �������� ��ȹ
SELECT MONTH, ROUND(SUM(pl) OVER(ORDER BY CAST(MONTH AS INT) ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)/100000000, 0)
FROM (
	SELECT MONTH, SUM(am_plan) AS pl
	FROM DATAMART_DIFV_PL DDP
	WHERE ds_item = '��������' AND YEAR = 2021
	GROUP BY MONTH) AS A
ORDER BY CAST(MONTH AS INT);

--# ���� ����-����
--- 21.07~21.09 �����͸� ����. ���� �����ʹ�`DATAMART_DHDT_TOTAL`�� ����. ������ ��� ���
--- ��� ����: ī�� ��ǥ(`�濵 ��Ȳ`),`���� ����`,`���� �������ͷ�`,`���� ��������`,`���� �������ͷ�`,`���� ����`,`��� ���� ����`,`���Ͱ�꼭`,`�繫����ǥ`,`��ä����`,`���Ա�������`,`��������`,`���ں������`

--# ���� ����-�ι�
--- DM�� ������ x ������ �ܵ����� ���
--- ��� ����: ī�� ��ǥ(`��������� ����`),`�ֿ� �ι��� ����`,`�ֿ� �ι��� �������ͷ�`