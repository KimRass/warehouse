# bs4
## BeautifulSoup
```python
from bs4 import BeautifulSoup as bs
```
### bs()
```python
soup = bs(xml,"lxml")
```
#### soup.find_all()
##### soup.find_all().find()
###### soup.find_all().find().get_text()
```python
features = ["bjdcode", "codeaptnm", "codehallnm", "codemgrnm", "codesalenm", "dorojuso", "hocnt", "kaptacompany", "kaptaddr", "kaptbcompany",  "kaptcode", "kaptdongcnt", "kaptfax", "kaptmarea", "kaptmarea",  "kaptmparea_136", "kaptmparea_135", "kaptmparea_85", "kaptmparea_60",  "kapttarea", "kapttel", "kapturl", "kaptusedate", "kaptdacnt", "privarea"]
for item in soup.find_all("item"):
    for feature in features:
        try:
            kapt_data.loc[index, feature] = item.find(feature).get_text()
        except:
            continue
```
------------------------------------------------------------------------------------
```python
!pip install --upgrade category_encoders
```
# category_encoders
```python
import category_encoders as ce
```
## ce.target_encoder
### ce.target_encoder.TargetEncoder()
```python
encoder = ce.target_encoder.TargetEncoder(cols=["company1"])
encoder.fit(data["company1"], data["money"]);
data["company1_label"] = encoder.transform(data["company1"]).round(0)
```

# collections
## deque
```python
from collections import deque
```

# cv2
```python
import cv2
```
## cv2.waitKey()
```python
k=cv2.waitKey(5) & 0xFF
    if k==27:
        break
```
## cv2.VideoCapture()
```python
cap=cv2.VideoCapture(0)
```
## cv2.destroyAllWindows()
```python
cv2.destroyAllWindows()
```
## cv2.rectangle()
```python

```
## cv2.circle()
```python

```
## cv2.puttext()
```python

```
## cv2.resize()
```python
img_resized=cv2.resize(img, dsize=(640, 480), interpolation=cv2.INTER_AREA)
```
## cv2.cvtColor()
```python
img_gray=cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
```
## cv2.imread()
```python
img=cv2.imread("300.jpg")
```
## cv2.imshow()
```python
cv2.imshow("img_resized", img_resized)
```
## cv2.findContours()
```python
mask=cv2.inRange(hsv,lower_blue,upper_blue)
image,contours,hierachy=cv2.findContours(mask.copy(),
                                         cv2.RETR_EXTERNAL,
                                         cv2.CHAIN_APPROX_SIMPLE)
```
```python
cap=cv2.VideoCapture(0)

while (1):
    success,frame=cap.read()
    cv2.imshow('Original',frame)
    edges=cv2.Canny(frame,100,200)
    cv2.imshow('Edges',edges)

k=cv2.waitKey(5) & 0xFF
    if k==27:
        break

cv2.destroyAllWindows()
cap.release()
```
```python
mport cv2
import numpy as np
import matplotlib.pyplot as plt

img=cv2.imread("300.jpg")
img_resized=cv2.resize(img,dsize=(640,480),interpolation=cv2.INTER_AREA)
img_gray=cv2.cvtColor(img_resized,cv2.COLOR_BGR2GRAY)
img_blur=cv2.GaussianBlur(img_gray,(5,5),0)

ret,img_binary=cv2.threshold(img_blur,100,230,cv2.THRESH_BINARY_INV)

image,contours,hierachy=cv2.findContours(img_binary.copy(),
                                          cv2.RETR_EXTERNAL,
                                          cv2.CHAIN_APPROX_SIMPLE)

rects=[cv2.boundingRect(contour) for contour in contours]

for rect in rects:
    cv2.rectangle(img_resized,(rect[0],rect[1]),
                  (rect[0]+rect[2],rect[1]+rect[3]),(0,255,0),1)

cv2.imshow("img_resized", img_resized)
cv2.waitKey(0)
cv2.destroyAllWindows()
```
```
import cv2
import numpy as np

img=cv2.imread("CM.png")

hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

lower_blue=np.array([100,50,150]) #파란색 계열의 범위 설정
upper_blue=np.array([130,255,255])

mask=cv2.inRange(hsv,lower_blue,upper_blue)
image,contours,hierachy=cv2.findContours(mask.copy(),
                                         cv2.RETR_EXTERNAL,
                                         cv2.CHAIN_APPROX_SIMPLE)

rects=[cv2.boundingRect(contour) for contour in contours] #contour를 감싸는 사각형들의 x,y,w,h를 'rects'에 저장

rects_selected=[]
for rect in rects:
    if rect[0]>1200 and 100<rect[1]<200:
        rects_selected.append(rect)

rects_selected.sort(key=lambda x:x[0])

for i,rect in enumerate(rects_selected):
    cv2.rectangle(img,(rect[0],rect[1]),
                  (rect[0]+rect[2],rect[1]+rect[3]),(0,0,255),2)
    cv2.putText(img,str(i+1),(rect[0]-5,rect[1]-5),fontFace=0,
                fontScale=0.6,color=(0,0,255),thickness=2)
    cv2.circle(img,(rect[0]+1,rect[1]-12),12,(0,0,255),2
```
## cv2.TERM_CRITERIA_EPS, cv2.TERM_CRITERIA_MAX_ITER
```python
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
```
-  Define criteria = (type, max_iter = 10, epsilon = 1.0)
## CV2.KMEANS_RANDOM_CENTERS
```python
flags = cv2.KMEANS_RANDOM_CENTERS
```
- 초기 중심점을 랜덤으로 설정.
```python
compactness, labels, centers = cv2.kmeans(z, 3, None, criteria, 10, flags)
```
- KMeans를 적용. k=2,  10번 반복한다.


# datetime
```python
import datetime
```
## datetime.datetime
```python
datetime.datetime(2018, 5, 19)
```
- \>\>\> datetime.datetime(2018, 5, 19, 0, 0)
### datetime.datetime.now()
```python
datetime.datetime.now()
```
- \>\>\> datetime.datetime(2020, 8, 18, 21, 44, 20, 835233)
## timestamp()
```python
datetime.datetime.now().timestamp()
```
- 1970년 1월 1일 0시 0분 0초로부터 몇 초가 지났는지 출력.


# gensim
```python
import gensim
```
## gensim.corpora
### gensim.corpora.Dictionary()
```python
id2word = gensim.corpora.Dictionary(docs_tkn)
```
#### id2word.id2token
- dict(id2word)와 dict(id2word.id2token)은 서로 동일.
#### id2word.token2id
- dict(id2word.token2id)는 key와 value가 서로 반대.
#### id2word.doc2bow()
```python
dtm = [id2word.doc2bow(doc) for doc in docs_tkn]
```
#### gensim.corpora.Dictionary.load()
```python
id2word = gensim.corpora.Dictionary.load("kakaotalk id2word")
```
### gensim.corpora.BleiCorpus
#### gensim.corpora.BleiCorpus.serizalize()
```python
gensim.corpora.BleiCorpus.serialize("kakotalk dtm", dtm)
```
### gensim.corpora.bleicorpus
#### gensim.corpora.bleicorpus.BleiCorpus()
```python
dtm = gensim.corpora.bleicorpus.BleiCorpus("kakaotalk dtm")
```
## gensim.models
### gensim.models.TfidfModel()
```python
tfidf = gensim.models.TfidfModel(dtm)[dtm]
```
### gensim.models.AuthorTopicModel()
```python
model = gensim.models.AuthorTopicModel(corpus=dtm, id2word=id2word, num_topics=n_topics, author2doc=aut2doc, passes=1000)
```
### gensim.models.Word2Vec()
```python
model = gensim.models.Word2Vec(sentences, min_count=5, size=300, sg=1, iter=10, workers=4, ns_exponent=0.75, window=7)
```
### gensim.models.FastText()
```python
model = gensim.models.FastText(sentences, min_count=5, sg=1, size=300, workers=4, min_n=2, max_n=7, alpha=0.05, iter=10, window=7)
```
#### model.save()
```python
model.save("kakaotalk model")
```
#### model.show_topic()
```python
model.show_topic(1, topn=20)
```
#### model.wv.most_similar()
```python
model.wv.most_similar("good)
```
#### gensim.models.AuthorTopicModel.load()
```python
model = gensim.models.AuthorTopicModel.load("kakaotalk model")
```
## gensim.models.ldamodel.Ldamodel()
```python
model = gensim.models.ldamodel.LdaModel(dtm, num_topics=n_topics, id2word=id2word, alpha="auto", eta="auto")
```
### model.show_topic()
```python
model.show_topic(2, 10)
```
- arguments : topic의 index, 출력할 word 개수

