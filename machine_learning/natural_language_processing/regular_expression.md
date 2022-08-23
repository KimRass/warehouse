# `re`
```python
 import re
```
- The meta-characters which do not match themselves because they have special meanings are: `.`, `^`, `$`, `*`, `+`, `?`, `{`, `}`, `[`, `]`, `(`, `)`, `\`, `|`.
- `.`: Match any single character except newline.
- `\n`: New line
- `\r`: Return
- `\t`: Tab
- `\f`: Form
- `\w`, `[a-zA-Z0-9_]`: Match any single "word" character: a letter or digit or underbar. 
- `\W`, `[^a-zA-Z0-9_]`: Match any single non-word character.
- `\s`, `[ \n\r\t\f]`: Match any single whitespace character(space, newline, return, tab, form).
- `\S`: Match any single non-whitespace character.
- `\d`, `[0-9]`: Match any single decimal digit.
- `\D`, `[^0-9]`: Match any single non-decimal digit.
- `\*`: 0개 이상의 바로 앞의 character(non-greedy way)
- `\+`: 1개 이상의 바로 앞의 character(non-greedy way)
- `\?`: 1개 이하의 바로 앞의 character
- `{m, n}`: m개~n개의 바로 앞의 character(생략된 m은 0과 동일, 생략된 n은 무한대와 동일)(non-greedy way)
- `{n}`: n개의 바로 앞의 character
- `^`: Match the start of the string. 바로 뒤의 expression으로 시작하는 패턴.
- `$`: Match the end of the string. 바로 앞의 expression으로 시작하는 패턴.
## Search Pattern
```python
# Scan through string looking for the first location where the regular expression pattern produces a match, and return a corresponding match object. Return None if no position in the string matches the pattern; note that this is different from finding a zero-length match at some point in the string.
re.search(pattern, string)
```
## `re.match()`
- If zero or more characters at the beginning of string match the regular expression pattern, return a corresponding match object. Return None if the string does not match the pattern; note that this is different from a zero-length match.
## Find All Matches
```python
# Return all non-overlapping matches of pattern in string, as a list of strings. The string is scanned left-to-right, and matches are returned in the order found. If one or more groups are present in the pattern, return a list of groups; this will be a list of tuples if the pattern has more than one group. Empty matches are included in the result.
re.findall(pattern, string)
```
## Split String by Delimiter
```python
re.split(maxsplit)
```
## Split String by Delimiters
```python
re.split(pattern, string, maxsplit)
```
## Replace Pattern
```python
# Example
# Add a whitespace after comma.
sentence = re.sub(pattern=r"(?<=[,])(?=[^\s])", repl=r" ", string=sentence)
```
## `re.compile()`
## Match Objects
```python
match.group(g)
match.span()
match.start()
match.end()
```

# Regular Expressions for Languages
```python
lang2regex = {
	"ko": r"[ㄱ-ㅎㅏ-ㅣ가-힣]+",
	# Reference: https://gist.github.com/terrancesnyder/1345094
	# `r"[ぁ-ん]+"`: Hiragana
	# `r"[ァ-ン]+"`: Full-width Katakana (Zenkaku 全角)
	# `r"[ｧ-ﾝﾞﾟ]+"`: Half-width Katakana (Hankaku 半角)
	# `r"[一-龯]+"`: ALL Japanese common & uncommon Kanji (`"4e00"` ~ `"9fcf"`)
	"ja": r"[ぁ-んァ-ンｧ-ﾝﾞﾟ一-龯々〆〤ヶ]+"
}
```