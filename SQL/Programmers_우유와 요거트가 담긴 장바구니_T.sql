https://programmers.co.kr/learn/courses/30/lessons/62284

SELECT DISTINCT cart_id
FROM cart_products
WHERE (name = "Milk"
AND cart_id
IN (SELECT DISTINCT cart_id
FROM cart_products
WHERE name = "Yogurt"));

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