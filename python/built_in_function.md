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

# Bit Operation
- `~x`(`NOT`): Returns the complement.
- `x & y`(`AND`): Returns `1` if the corresponding bit of `x` and of `y` is `1`, otherwise returns `0`.
- `x | y`(`OR`): Returns `0` if the corresponding bit of `x` and of `y` is `0`, otherwise returns `1`.
- `x ^ y`(`XOR`): Returns the same bit as the corresponding bit in `x` if that bit in `y` is `0`, otherwise the complement.
- `x << y`, `x >> y`: Returns `x` with the bits shifted to the left(right) by `y` places (and new bits on the right-hand-side are zeros).

# `hash()`

# `display()`

# `print()`
```python
# `end`: (default `"\n"`)
# `sep`: (default `" "`) Determine the value to join elements with.
# `">"`: Right align, `"^"`: Center align, `"<"`: Left align
# Examples
print(f"{'a':0>10}")
print(f"    - {'The program ended at:':<24s}{datetime.strftime(end, format='%Y-%m-%d %H:%M:%S'):>20s}.")
```

# `isinstance()`
```python
# `classinfo`: (`str`, `frozenset`, `np.ndarray`, `PurePath`, `Image.Image`, ...)
# Example
isinstance(movie, frozenset)
```

# `type()`
```python
type(test_X[0][0])
```
# Open Text File
```python
[line.rstrip() for line in open("....txt", mode="r").readlines()]
```
# `sum()`
```python
# 두번째 층의 대괄호 제거
sum(<List of Lists>, list)
```
# `assert`
```python
assert model_name in self.model_list, "There is no such a model."
```

# `eval()` and `exec()`
## `eval()`
```python
# `eval()` is for expression and returns the value of expression.
A = [eval(f"A{i}") for i in range(N, 0, -1)]
```
## `exec()`
```python
# `exec()` is for the statement and returns `None`.
# Example
for data in ["tasks", "comments", "projects", "prj_members", "members"]:
    exec(f"{data} = pd.read_csv('D:/디지털혁신팀/협업플랫폼 분석/{data}.csv')")
exec(f"{table} = pd.DataFrame(result)")
```

# Read or Write Data
```python
# `mode`: (`"r"`, `"rb"`, `"w"`, `"wb"`, ...) 
with open(file, mode, [encoding]) as f:
    # Read
    data = f
    # Write
    f.write(data)
    ...
```
## Read Line by Line
```python
# Example
[line.strip() for line in open(classes_txt_path, mode="r").readlines()]
```
# `input()`
```python
A = list(map(int, input("Message").split()))
```

# Unicode
```python
# Returns the unicode code of a specified character.
ord()
# Returns the character that represents the specified unicode code.
chr()
```

# `__slot__`
- Reference: https://ddanggle.gitbooks.io/interpy-kr/content/ch10-slots-magic.html
- `__slots__`을 사용해서 파이썬이 dict형을 사용하지 않도록 하고 고정된 속성의 집합에만 메모리를 할당하도록 합니다.
```python
# Example
class MyClass(object):
    __slots__ = ["name", "age"]

    def __init__(name, age):
        self.name = name
        self.age = age
        ...
    ...
```

# List
- Mutable.
- Unhashable.
- Subscriptable.
## Get Index of Element in List
```python
List.index(...)
```
## Add Single Element to List
```python
# Adds the argument as a single element to the end of a List.
List.append()
# Index, Value 순으로 Argument를 입력합니다.
List.insert()
```
## Add Multiple Elements to List
```python
# Iterates over the argument and adding each element to the List and extending the List.
List.extend()
```
## `List.remove()`
```python
features.remove("area")
```
## `List.count()`
## Sort List
```python
# `reverse`: (Bool, default `False`)
# `key`: Define a function to sort by.
sorted(confs, key=lambda x:(x[0], x[1]))
```
## Reverse List
```python
list(reversed([int(i) for i in str(n)]))
```
## `map(function, iterable)`
## Filter List
```python
filter(function, iterable)
# Example
langs = list(filter(lambda x: x != unk_lang, langs))
```
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
## Intersection
```python
<Set1> & <Set2>
<Set1>.intersection(<Set2>, <Set3>, ...)
```
## Union
```python
<Set1> | <Set2>
<Set1>.union(<Set2>, <Set3>, ...)
```
## Add Single Element
```python
<Set>.add()
```
## Add Multiple Elements
```python
# It expects a single or multiple iterable sequences as arguments and appends all the elements in these iterable sequences to the Set.
<Set>.update()
```
## `Set.discard()`
## Frozenset
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
## Add Element
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
## Sort Dictionary
```python
word2cnt = dict(sorted(tkn.word_counts.items(), key=lambda x:x[1], reverse=True))
```
## Merge Dictionary
```python
{**<dic1>, **<dic2>, ...}
```
## Dictionary Comprehension
```python
min_dists = {i:0 if i == start else math.inf for i in range(1, V + 1)}
```
# String
- Immutable.
- Subscriptable.
## `String.format()`
## `String.startswith()`, `String.endswith()`
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
## Replace String
```python
# `count`: (int, optional) A number specifying how many occurrences of the old value you want to replace. Default is all occurrences.
String.replace(old, new, [count])
```

# `dataclasses`
- Reference: https://www.daleseo.com/python-dataclasses/
```python
from dataclasses import class
from datetime import date

# Example
# The two classes `ClassName1` and `ClassName2` are the same.
class ClassName1:
    def __init__(
        self, param1: int, param2: str, param3: date, param4: bool = False
    ) -> None:
        self.param1 = param1
        self.param2 = param2
        self.param3 = param3
        self.param4 = param4

    def __repr__(self):
        return (
            self.__class__.__qualname__ + f"(id={self.param1!r}, name={self.param2!r}, param3={self.param3!r}, param4={self.param4!r})"
        )

    def __eq__(self, other):
        if other.__class__ is self.__class__:
            return (
                self.param1, self.name, self.param3, self.param4
            ) == (
                other.id, other.name, other.param3, other.param4,
            )
        return NotImplemented

@dataclass
class ClassName2:
    param1: int
    param2: str
    param3: date
    param4: bool = False
```

# `hydra`
- Install
```sh
pip intall hydra-core
```
- Reference: https://hydra.cc/docs/tutorials/
```python
import hydra
from omegaconf import DictConfig, OmegaConf

# Hydra configuration files are yaml files and should have the ".yaml" file extension.
# Specify the config name by passing a config_name parameter to the `@hydra.main()` decorator. Note that you should omit the ".yaml" extension.
# `config_path` is a directory relative to "my_app.py".
@hydra.main(version_base=None, config_path="<directory_to_yaml_file>", config_name="<yaml_file_name>")
def function_name(config):
    print(OmegaConf.to_yaml(config))

if __name__ == "__main__":
    # "config.yaml" is loaded automatically when you run your application.
    function_name()
```
```python
# Directory layout:
# ├─ config
# │  └─ db
# │      └─ mysql.yaml
# └── function_name.py

# "config/db/mysql.yaml":
# driver: mysql
# user: omry
# password: secret

# What if we want to add an postgresql option now? Yes, we can easily add a "db/postgresql.yaml" config group option. But that is not the only way! We can also use `ConfigStore` to make another config group option for "db" available to Hydra.

# "my_app.py"
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf, DictConfig

@dataclass
class PostgresSQLConfig:
    ...

cs = ConfigStore.instance()
# Registering the Config class with the name "<yaml_file_name>" with the config group "db"; "db.<yaml_file_name>.yaml"
cs.store(name="<yaml_file_name>", group="<config_group_name, e.g., db>", node=PostgresSQLConfig)

@hydra.main(version_base=None, config_path="config")
def function_name(config: DictConfig) -> None:
    print(OmegaConf.to_yaml(config))


if __name__ == "__main__":
    function_name()
```
```sh
python3 my_app.py \
    <config_group_name>=<yaml_file_name>
    # You can remove a default entry from the defaults list by prefixing it with ~:
    ~<config_group_name>

# Output:
# db:
#    driver: mysql
#    user: omry
#    password: secret
```