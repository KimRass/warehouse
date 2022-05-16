Written by KimRass

# Automatically Reload Modules
```json
# "settings.json"
{
    "jupyter.runStartupCommands": [
        "%load_ext autoreload", "%autoreload 2"
    ]
}
```

# `bin()`, `oct()`, `hex()`
- Source: https://wiki.python.org/moin/BitwiseOperators
- `~x`(`NOT`): Returns the complement.
- `x & y`(`AND`): Returns `1` if the corresponding bit of `x` and of `y` is `1`, otherwise returns `0`.
- `x | y`(`OR`): Returns `0` if the corresponding bit of `x` and of `y` is `0`, otherwise returns `1`.
- `x ^ y`(`XOR`): Returns the same bit as the corresponding bit in `x` if that bit in `y` is `0`, otherwise the complement.
- `x << y`, `x >> y`: Returns `x` with the bits shifted to the left(right) by `y` places (and new bits on the right-hand-side are zeros).
# `int()`
- `base`: (Default `10`) Number format.
# `round()`
```python
print(round(summ/leng, 1))
```
# `hash()`
# `display()`
# `print()`
- `end`: (default `"\n"`)
- `sep`: (default `" "`) Determine the value to join elements with.
```python
print(f"{'a':0>10}")
```
- You can also append characters other than white spaces, by adding the specified characters before the `>`(right align), `^`(center align) or `<`(left align) character:
# `isinstance()`
```python
if not isinstance(movie, frozenset):
    movie = frozenset(movie)
```
# `type()`
```python
type(test_X[0][0])
```
# `sum()`
```python
sum(sentences, [])
```
- 두번째 층의 대괄호 제거
# `assert`
```python
assert model_name in self.model_list, "There is no such a model."
```
# `eval()`
```python
A = [eval(f"A{i}") for i in range(N, 0, -1)]
```
- `eval()` is for expression and returns the value of expression.
# `exec()`
```python
for data in ["tasks", "comments", "projects", "prj_members", "members"]:
    exec(f"{data} = pd.read_csv('D:/디지털혁신팀/협업플랫폼 분석/{data}.csv')")
```
```python
exec(f"{table} = pd.DataFrame(result)")
```
- `exce()` is for statement and return `None`.
# `open()`
```python
# `mode`: (`"r"`, `"rb"`, `"w"`, `"wb"`, ...) 
with open(file, mode, [encoding]) as f:
    ...
```
## `f.readline()`, `f.readlines()`
# `input()`
```python
A = list(map(int, input("Message").split()))
```
# `ord()`
- Returns the unicode code of a specified character.
# `chr()`
- Returns the character that represents the specified unicode code.
# `Variable.data`
## `Variable.data.nbytes`
```python
print(f"{sparse_mat.data.nbytes:,}Bytes"
```
# List
- Mutable.
- Unhashable.
- Subscriptable.
## `List.index()`
## `List.append()`
- Adds the argument as a single element to the end of a List. 
## `List.extend()`
- Iterates over the argument and adding each element to the List and extending the List.
## `List.insert()`
- idx, value 순으로 arg를 입력합니다.
## `List.remove()`
```python
features.remove("area")
```
## `List.count()`
## `sorted()`
```python
sorted(confs, key=lambda x:(x[0], x[1]))
```
- `reverse`: (Bool, default `False`)
- `key`: Define a function to sort by.
## `reversed()`
```python
list(reversed([int(i) for i in str(n)]))
```
## `map(function, iterable)`
## `filter(function, iterable)`
## `sum()`
```python
sum(sents, [])
```
## List Comprehension
```python
chars = set([char for word in words for char in word])
```
```python
idxs = [idx for idx, num in zip(range(len(nums)), nums) if num!=0]
```
# Set
- Mutable.
- Unhashable.
- No order.
- Not subscriptable.
## `<<Set1>> & <<Set2>>`
- Returns the union of `<<Set1>>` and `<<Set2>>`.
## `<<Set1>> | <<Set2>>`
- Returns the intersection of `<<Set1>>` and `<<Set2>>`.
## `Set.add()`
- Adds the argument as a single element to the end of a Set if it is not in the Set.
## `Set.update()`
- It expects a single or multiple iterable sequences as arguments and appends all the elements in these iterable sequences to the Set.
## `Set.discard()`
# Frozenset
- 구성 요소들이 순서대로 들어 있지 않아 인덱스를 사용한 연산을 할 수 없고
- 유일한 항목 1개만 들어가 있습니다.
- set 객체는 구성 항목들을 변경할 수 있지만 frozenset 객체는 변경할 수 없는 특성을 지니고 있어, set에서는 사용할 수 있는 메소드들 add, clear, discard, pop, remove, update 등은 사용할 수 없습니다. 
- 항목이 객체에 포함되어 있는지(membership)를 확인할 수 있으며,  항목이 몇개인지는 알아낼 수 있습니다.
- frozenset은 dictionary의 key가 될 수 있지만 set은 될 수 없음.
# `Dictionary`
- Mutable.
- Unhashable.
- Subscriptable.
## `Dictionary[]`, `Dictionary.get()`
- key를 입력받아 value를 반환합니다.
## `Dictionary.items()`
```python
for key, value in dic.items():
    print(key, value)
```
## `Dictionary.setdefault()`
## `Dictionary.update()`
```python
dic.update({key1:value1, key2:value2})
```
## `Dictionary.pop()`
```python
dic.pop(<<key>>)
```
## `Dictionary.keys()`, `Dictionary.values()`
- Data type: `dict_keys`, `dict_values` respectively.
## `Dictionary.fromkeys()`
## `sorted()`
```python
word2cnt = dict(sorted(tkn.word_counts.items(), key=lambda x:x[1], reverse=True))
```
## Dictionary Comprehension
```python
min_dists = {i:0 if i == start else math.inf for i in range(1, V + 1)}
```
# String
- Immutable.
- Subscriptable.
## `String.format()`
## `String.ljust()`, `String.rjust()`
```python
string.ljust(<<Target Length>>, <<Character to Pad>>)
```
## `String.zfill()`
## `String.join()`
```python
" ".join(["good", "bad", "worse", "so good"])
```
 - Join all items in a Tuple or List into a string, using `String`.
## `String.split()`
- Split a string into a list where each word is a list item.
- `maxsplit`: How many splits to do.
## `String.upper()`, `String.lower()`
```python
data.columns = data.columns.str.lower()
```
## `String.isupper()`, `String.islower()`
## `String.isalpha()`
## `String.isdigit()`
## `String.count()`
```python
"저는 과일이 좋아요".count("과일이")
```
## `String.find()`
- Return the first index of the argument.
## `String.startswith()`, `String.endswith()`
- Return `True` if a string starts with the specified prefix. If not, return `False`
## `String.strip()`, `String.lstrip()`, `String.rstrip()`
## `String.replace()`
- `count`: (int, optional) A number specifying how many occurrences of the old value you want to replace. Default is all occurrences.
