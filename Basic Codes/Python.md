## arguments, parameters
- arguments : 인수
- parameters : 인자
# print()
```python
print('variable', variable, end="") 
```
- 다음 print문이 이어서 출력.
```python
print("This is string :", str, "This is number :" , num1, num2)
```
- 콤마로 연결 시 띄어쓰기로 연결
```python
print("This is string :" + str, "This is number :" , num1 + num2)
```
- +로 연결 시 문자끼리는 공백 없이 이어지고 숫자끼리는 합산.
## print() + format()
```python
print("{0} and {1}".format("spam", "eggs")
```
```python
print("[{0:>4d}], [{1:>20d}]".format(100, 200))
```
```python
print("[{0:<20s}]".format("string"))
```
```python
print("[{0:<20.20f}], [{1:>10.2f}]".format(3.14, 10.925))
```
```python
print("{:>4d} | {:>7.4f} | {:>7.4f} | {:>9.6f}".format(i + 1, W.numpy(), b.numpy(), loss.numpy()))
```
# f-string
```python
name = "Song"
sex = "male"

f"Hi, I am {name}. I am {sex}."
```
- \>\>\> "Hi, I am song. I am male."
# isinstance()
```python
if not isinstance(movie, frozenset):
    movie = frozenset(movie)
```

# list
## list[]
## list.index()
```python
names.index((17228, '아트빌'))
```
## list.append()
```python
feature_to_shuffle.append("area")
```
## list.remove()
```python
features.remove("area")
```
## list.sort()
```python
A.sort(reverse=True)
```
```python
m.sort(key=len)
```
- in-place 함수
## sorted(), reversed()
```python
A = reversed(A)
```
## str.join()
```python
" ".join(["good", "bad", "worse", "so good"])
```
- str을 사이에 두고 리스트의 모든 원소들을 하나로 합침
## map()
```python
x_data = list(map(lambda word : [char2idx.get(char) for char in word], words))
```
## split()
```python
msg_tkn = [msg.split() for msg in data["msg"]]
```
## filter()
## sum()
```python
sum(sents, [])
```
# set
## set1 & set2
## set1 | set2
## set.add()
## set.update()
- list.append()와 동일.
# frozenset()
- 구성 요소들이 순서대로 들어 있지 않아 인덱스를 사용한 연산을 할 수 없고
- 유일한 항목 1개만 들어가 있습니다.
- set 객체는 구성 항목들을 변경할 수 있지만 frozenset 객체는 변경할 수 없는 특성을 지니고 있어, set에서는 사용할 수 있는 메소드들 add, clear, discard, pop, remove, update 등은 사용할 수 없습니다. 
- 항목이 객체에 포함되어 있는지(membership)를 확인할 수 있으며,  항목이 몇개인지는 알아낼 수 있습니다.
- frozenset은 dictionary의 key가 될 수 있지만 set은 될 수 없음.
# dictionary
## dic[key]
## dic[key] = value
```python
aut2doc = {}
for user, idx in auts.groups.items():
    aut2doc[user] = list(idx)
```
## dic.items()
```python
for key, value in dic.items():
    print(key, value)
```
## dic.setdefault() : 추가
```python
dic.setdefault(key)
dic.setfefault(key, value)
```
## dic.update() : 추가 또는 수정
```python
dic.update({key1:value1, key2:value2})
```
## dic.pop() : 삭제
```python
dic.pop(key)
```
- \>\>\> value
## dic.get()
```python
dic.get(key)
```
- \>\>\> value
## dic.keys(), dic.values()
## dic.fromkeys(list or tuple, value)
## dictionary comprehension
```python
{idx:char for idx, char in enumerate(char_set)}
```
# exec()
```python
for i in range(N):
    exec(f"a{i} = int(input())")
```
# eval()
```python
A = [eval(f"A{i}") for i in range(N, 0, -1)]
```

# class
## instance attribute(instance variables)
```python
class CLASS:
    def __init__(self):
        self.ATTRIBUTE=VALUE
```
* INSTANCE.ATTRIBUTE로 사용
## class attriubute(class variables)
```python
class CLASS:
    ATTRIBUTE=VALUE
```
* CLASS.ATTRIBUTE로 사용
* 모든 INSTANCE가 ATTRIBUTE 값을 공유.
* 동일한 instance attribute와 class attribute가 있으면 instance attribute -> class attribute 순으로 method를 탐색.
* INSTANCE.ATTRIBUTE로 사용 시 INSTANCE의 namespace에서 ATTRIBUTE를 찾고 없으면 CLASS의 namespace로 이동한 후 다시 ATTRIBUTE를 찾아 그 값을 반환.
## \_\_init\_\_
```python
INSTANCE=CLASS() #instance를 initiate 할 때 실행
```
## \_\_init\_\_, \_\_call\_\_
```python
class CLASS:
    def __init__(self, parameter1, parameter2, ...)
        ...
        
    def __call__(self, parameter3, parameter4, ...)
        ...
        return ...
        
    def FUNCTION(self, parameter5, parameter6, ...)
        ...
        return ...
...
INSTANCE=CLASS(parameter1, parameter2, ...) #__init__문은 instance를 initiate 할 때 실행
INSTANCE(parameter3, parameter4, ...) #__call__문은 instance를 call 할 때 실행
INSTANCE.FUNCTION(parameter5, parameter6, ...)
```
## method
- method : class 정의문 안에서 정의된 함수
- method의 첫번째 parameter는 항상 self여야 함
- method의 첫 번째 parameter는 self지만 호출할 때는 아무것도 전달하지 않는 이유는 첫 번째 parameter인 self에 대한 값은 파이썬이 자동으로 넘겨주기 때문입니다.

### class variables
- class 정의문 안에서 정의된 variables
### instance variables
- self가 붙어 있는 variables
## override
- 출처 : https://rednooby.tistory.com/55
## super()
- 출처 : https://rednooby.tistory.com/56?category=633023
```python
class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
```
# with open() as f:
```python
with open("C:/Users/5CG7092POZ/nsmc-master/ratings_train.txt", "r", encoding="utf-8") as f:
    train_docs = [line.split("\t") for line in f.read().splitlines()][1:]
```
# input()
```python
A = list(map(int, input("A를 차례대로 입력 : ").split()))
```  
