# Set
- Mutable.
- Unhashable.
- No order.
- Not subscriptable.
- 데이터에 접근하는 데 필요한 time complexity: O(1)

# If Statement


## 1) `if A and B and C:`
- This is a short-circuit evaluation that checks each condition individually, from left to right.
- If `A` is `False`, it will stop there (i.e., short-circuit) and won’t check `B` or `C` because the entire condition is already `False`.
- If `A` is `True`, it moves on to check `B`. If `B` is `False`, it stops, and so on.
- This ensures that as soon as one condition is `False`, it doesn’t evaluate the rest.

## 2) `if all([A, B, C]):`
- The function `all()` takes an iterable (like a list) and checks whether all the elements are `True`.
- It also short-circuits, meaning that if it encounters a `False` value, it will stop checking further.
- The main difference is that `all()` can take any number of conditions and works well when you want to check multiple conditions in a structured way (like a list or generator).