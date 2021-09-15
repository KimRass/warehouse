# DATETIME에서 DATE로 형 변환
- Source: https://programmers.co.kr/learn/courses/30/lessons/59414
```sql
SELECT animal_id, name, DATE_FORMAT(datetime, "%Y-%m-%d")
FROM animal_ins
ORDER BY animal_id;
```

# NULL 처리하기
- Source: https://programmers.co.kr/learn/courses/30/lessons/59410
```sql
SELECT animal_type,
IFNULL(name, "No name"),
sex_upon_intake
FROM animal_ins
ORDER BY animal_id;
```

# 고양이와 개는 몇 마리 있을까
- Source: https://programmers.co.kr/learn/courses/30/lessons/59040
```sql
SELECT animal_type, COUNT(animal_id)
FROM animal_ins
GROUP BY animal_type
HAVING animal_type in ("Cat", "Dog")
ORDER BY animal_type;
```

# 동명 동물 수 찾기
- Source: https://programmers.co.kr/learn/courses/30/lessons/59041
```sql
SELECT name, COUNT(name)
FROM animal_ins
GROUP BY name
HAVING COUNT(name) >= 2
ORDER BY name;
- Source: 

# 동물 수 구하기
- Source: https://programmers.co.kr/learn/courses/30/lessons/59406
```sql
SELECT COUNT(animal_id)
FROM animal_ins;
```

# 동물의 아이디와 이름
- Source: https://programmers.co.kr/learn/courses/30/lessons/59403
```sql
SELECT animal_id, name
FROM animal_ins
ORDER BY animal_id ASC;
```

# 루시와 엘라 찾기
- Source: https://programmers.co.kr/learn/courses/30/lessons/59046
```sql
SELECT animal_id, name, sex_upon_intake
FROM animal_ins
WHERE name IN ("Lucy", "Ella", "Pickle", "Rogan", "Sabrina", "Mitty");
```

# 모든 레코드 조회하기
- Source: https://programmers.co.kr/learn/courses/30/lessons/59034
```sql
SELECT animal_id, animal_type, datetime, intake_condition, name, sex_upon_intake
FROM animal_ins
ORDER BY animal_id ASC;
```

# 모든 레코드 조회하기
- Source: https://programmers.co.kr/learn/courses/30/lessons/59034
```sql
SELECT animal_id, animal_type, datetime, intake_condition, name, sex_upon_intake
FROM animal_ins
ORDER BY animal_id ASC;
```

# 보호소에서 중성화한 동물
- Source: https://programmers.co.kr/learn/courses/30/lessons/59045
```sql
SELECT animal_ins.animal_id, animal_ins.animal_type, animal_ins.name
FROM animal_ins
INNER JOIN animal_outs
ON animal_ins.animal_id = animal_outs.animal_id
WHERE animal_ins.sex_upon_intake LIKE "Intact%"
AND (animal_outs.sex_upon_outcome LIKE "Spayed%"
OR animal_outs.sex_upon_outcome LIKE "Neutered%")
ORDER BY animal_ins.animal_id;
```

# 상위 n개 레코드
- Source: https://programmers.co.kr/learn/courses/30/lessons/59405
```sql
SELECT name
FROM animal_ins
ORDER BY datetime
LIMIT 1;
```

# 아픈 동물 찾기
- Source: https://programmers.co.kr/learn/courses/30/lessons/59037
```sql
SELECT animal_id, name
FROM animal_ins
WHERE intake_condition = "Sick";
```

# 없어진 기록 찾기
- Source: https://programmers.co.kr/learn/courses/30/lessons/59042
```sql
SELECT animal_outs.animal_id, animal_outs.name
FROM animal_outs
LEFT OUTER JOIN animal_ins
ON animal_ins.animal_id = animal_outs.animal_id
WHERE animal_ins.animal_id IS NULL
ORDER BY animal_id;
```

# 여러 기준으로 정렬하기
- Source: https://programmers.co.kr/learn/courses/30/lessons/59404
```sql
SELECT animal_id, name, datetime
FROM animal_ins
ORDER BY name ASC, datetime DESC;
```

# 역순 정렬하기
- Source: https://programmers.co.kr/learn/courses/30/lessons/59035
```sql
SELECT name, datetime
FROM animal_ins
ORDER BY animal_id DESC;
```

# 오랜 기간 보호한 동물(1)
- Source: https://programmers.co.kr/learn/courses/30/lessons/59044
```sql
SELECT animal_ins.name, animal_ins.datetime
FROM animal_ins
LEFT OUTER JOIN animal_outs
ON animal_ins.animal_id = animal_outs.animal_id
WHERE animal_outs.datetime IS NULL
ORDER BY animal_ins.datetime ASC
LIMIT 3;
```

# 오랜 기간 보호한 동물(2)
- Source: https://programmers.co.kr/learn/courses/30/lessons/59411
```sql
SELECT animal_outs.animal_id, animal_outs.name
FROM animal_outs
LEFT OUTER JOIN animal_ins
ON animal_outs.animal_id = animal_ins.animal_id
ORDER BY animal_outs.datetime - animal_ins.datetime DESC
LIMIT 2;
```

# 우유와 요거트가 담긴 장바구니
- Source: https://programmers.co.kr/learn/courses/30/lessons/62284
```sql
SELECT DISTINCT cart_id
FROM cart_products
WHERE (name = "Milk"
AND cart_id
IN (SELECT DISTINCT cart_id
FROM cart_products
WHERE name = "Yogurt"));
```
```sql
SELECT yogurt_cart.cart_id
FROM (SELECT DISTINCT cart_id
FROM cart_products
WHERE name = "Yogurt")
AS yogurt_cart
INNER JOIN (SELECT DISTINCT cart_id
FROM cart_products
WHERE name = "Milk")
AS milk_cart
ON yogurt_cart.cart_id = milk_cart.cart_id;
```
# 이름에 el이 들어가는 동물 찾기
- Source: https://programmers.co.kr/learn/courses/30/lessons/59047
```sql
SELECT animal_id, name
FROM animal_ins
WHERE animal_type = "Dog"
AND UPPER(name) LIKE "%EL%"
ORDER BY name;
```

# 이름이 없는 동물의 아이디
- Source: https://programmers.co.kr/learn/courses/30/lessons/59039
```sql
SELECT animal_id
FROM animal_ins
WHERE name IS NULL;
```

# 이름이 있는 동물의 아이디
- Source: https://programmers.co.kr/learn/courses/30/lessons/59407
```sql
SELECT animal_id
FROM animal_ins
WHERE name IS NOT NULL
ORDER BY animal_id;
```

# 입양 시각 구하기(1)
- Source: https://programmers.co.kr/learn/courses/30/lessons/59412
```sql
SELECT HOUR(datetime) HOUR, COUNT(*)
FROM animal_outs
GROUP BY HOUR(datetime)
HAVING HOUR BETWEEN 9 AND 19
ORDER BY HOUR;
```

# 입양 시각 구하기(2)*
- Source: https://programmers.co.kr/learn/courses/30/lessons/59413
```sql
SET @hour := -1;

SELECT (@hour := @hour + 1) as "hour",
(SELECT COUNT(*) FROM animal_outs WHERE @hour = HOUR(datetime)) as "count"
FROM animal_outs
WHERE @hour < 23;
```

# 있었는데요 없었습니다
- Source: https://programmers.co.kr/learn/courses/30/lessons/59043
```sql
SELECT animal_ins.animal_id, animal_ins.name
FROM animal_ins
INNER JOIN animal_outs
ON animal_ins.animal_id = animal_outs.animal_id
WHERE animal_ins.datetime > animal_outs.datetime
ORDER BY animal_ins.datetime;
```

# 중복 제거하기
- Source: https://programmers.co.kr/learn/courses/30/lessons/59408
```sql
SELECT COUNT(DISTINCT name)
FROM animal_ins;
```

# 중성화 여부 파악하기
- Source: https://programmers.co.kr/learn/courses/30/lessons/59409
```sql
SELECT animal_id, name,
CASE WHEN (sex_upon_intake LIKE "%Neutered%"
OR sex_upon_intake LIKE "%Spayed%")
THEN "O"
ELSE "X" END AS "중성화 여부"
FROM animal_ins
ORDER BY animal_id;
```

# 최댓값 구하기
- Source: https://programmers.co.kr/learn/courses/30/lessons/59415
```sql
SELECT MAX(datetime)
FROM animal_ins
```

# 최솟값 구하기
- Source: https://programmers.co.kr/learn/courses/30/lessons/59038
```sql
SELECT datetime
FROM animal_ins
ORDER BY datetime ASC
LIMIT 1;
```

# 헤비 유저가 소유한 장소
- Source: https://programmers.co.kr/learn/courses/30/lessons/77487
```sql
SELECT places.id, places.name, places.host_id
FROM places
RIGHT OUTER JOIN (SELECT host_id
FROM places
GROUP BY host_id
HAVING COUNT(id) >= 2)
AS heavy
ON places.host_id = heavy.host_id
ORDER BY id;
```