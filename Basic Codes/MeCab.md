# MeCab
```python
import MeCab
```
```python
def pos(text):
    p = re.compile(".+\t[A-Z]+")
    return [tuple(p.match(line).group().split("\t")) for line in MeCab.Tagger().parse(text).splitlines()[:-1]]

def morphs(text):
    p = re.compile(".+\t[A-Z]+")
    return [p.match(line).group().split("\t")[0] for line in MeCab.Tagger().parse(text).splitlines()[:-1]]

def nouns(text):
    p = re.compile(".+\t[A-Z]+")
    temp = [tuple(p.match(line).group().split("\t")) for line in MeCab.Tagger().parse(text).splitlines()[:-1]]
    nouns=[]
    for word in temp:
        if word[1] in ["NNG", "NNP", "NNB", "NNBC", "NP", "NR"]:
            nouns.append(word[0])
    return nouns

def cln(text):
    return re.sub("[^ㄱ-ㅣ가-힣 ]", "", text)

def def_sw(path):
    sw = set()
    for i in string.punctuation:
        sw.add(i)
    with open(path, encoding="utf-8") as f:
        for word in f:
            sw.add(word.split("\n")[0])
    return sw
```
```python
train_data = []
for line in tqdm(train_docs):
    review = line[1]
    label = line[2]
    review_tkn = nouns(cln(review))
    review_tkn = [word for word in review_tkn if (word not in sw)]
    train_data.append((review_tkn, label))
```
