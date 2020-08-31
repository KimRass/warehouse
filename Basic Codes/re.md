# re
```python
 import re
```
## re.search()
## re.match()
- re.search()와 유사하나 주어진 문자열의 맨 처음과 대응할 때만 object를 반환.
## re.findall()
- re.search()와 유사하나 대응하는 모든 문자열을 list로 반환.
### re.search().group(), re.match().group()
```python
re.search(r"(\w+)@(.+)", "test@gmail.com").group(0) #test@gmail.com
re.search(r"(\w+)@(.+)", "test@gmail.com").group(1) #test
re.search(r"(\w+)@(.+)", "test@gmail.com").group(2) #gmail.com
```
## re.sub()
```python
re.sub(r"\w+@\w+.\w+", "email address", "test@gmail.com and test2@gmail.com", count=1)
```
- count=0 : 전체 치환
## re.compile()
```python
p = re.compile(".+\t[A-Z]+")
```
- 이후 p.search(), p.match() 등의 형태로 사용.

## regular expressions
### . : newline을 제외한 어떤 character
### \w,   [a-zA-Z0-9_]: 어떤 알파벳 소문자 또는 알파벳 대문자 또는 숫자 또는 _
### \W, [^a-zA-Z0-9_] : 어떤 알파벳 소문자 또는 알파벳 대문자 또는 숫자 또는 _가 아닌
### \d, [0-9] : 어떤 숫자
### \D, [^0-9] : 어떤 숫자가 아닌
### \s : 공백
### \S : 공백이 아닌 어떤 character
### \t : tab
### \n : newline
### \r : return
### \가 붙으면 문자 그 자체를 의미.
### [] : [] 안의 문자를 1개 이상 포함하는
- [] 내부의 문자는 해당 문자 자체를 나타냄.
#### [abc]
- "a"는 정규식과 일치하는 문자인 "a"가 있으므로 매치
- "before"는 정규식과 일치하는 문자인 "b"가 있으므로 매치
- "dude"는 정규식과 일치하는 문자인 a, b, c 중 어느 하나도 포함하고 있지 않으므로 매치되지 않음
### * : 0개~무한대의 바로 앞의 character
### + : 1개~무한대의 바로 앞의 character
### ? : 0개~1개의 바로 앞의 character
### {m,n} : m개~n개의 바로 앞의 character
- 생략된 m은 0과 동일, 생략된 n은 무한대와 동일
- *?, +?, {m,n}? : non-greedy way
### {n} : n개의 바로 앞의 character
### ? : 0개~1개의 바로 앞의 character
### ^ : 바로 뒤의 문자열로 시작하는
### [^] : 바로 뒤의 문자열을 제외한
### $ : 바로 앞의 문자열로 끝나는
