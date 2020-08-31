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
