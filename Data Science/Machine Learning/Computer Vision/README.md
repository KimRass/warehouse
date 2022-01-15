Written by KimRass

- ![image.png](/files/3008821901398452965)
- ![image.png](/files/3011488520092691369)
- Feature Extraction -> Classification
- AI 기술 발달 이전: Feature Extraction을 어떻게 할 것인가에 대한 수많은 고민. 인간이 손수 언뜻 보기에 이해가 가지 않는 이상한 Rule 생성.
- AI: 모델 구조만 잘 짜면 컴퓨터가 알아서 Feature Extraction까지 해버림
- Feature Extraction을 위한 도구: Convolution과 Pooling
- ![image.png](/files/3010814950474546329)

## Obeject Detection은 왜 어려운가?
### Classification + Regression 동시에 수행
- 어떤 Object의 Class가 무엇인지 판별(Classification)
- 해당 Object의 둘러싸는 Bounding Box를 정의하는 4개의 변수(x, y, w, h) 계산(Regression)
### Object가 너무 다양함
- 크기도 천차만별, 색깔도 천차만별, Image 내 위치도 천차만별
### 실시간 Detection을 위해서는 연산 속도도 중요
- 상용 가능한 수준이 되기 위해서는 1초 미만의 시간 내에 결과를 도출해야 함
### 인간이 보기에도 애매한 상황이 다수
- 전체 Image에서 Object가 차지하는 비중이 매우 적고 대부분이 Background임.
- 다수의 Object들이 서로 겹쳐져 있음.
- 어떤 Object의 Bounding Box는 유일하지 않음.
### AI Model의 학습에 필요한 데이터 셋의 부족
- Image를 수집하는 것뿐만 아니라 각각에 대해 Annotation을 만들어야 함.
```
./racoon_images/raccoon-1.jpg 81,88,522,408
./racoon_images/raccoon-2.jpg 60,51,462,499
./racoon_images/raccoon-3.jpg 1,1,720,476
./racoon_images/raccoon-4.jpg 21,11,200,183
./racoon_images/raccoon-5.jpg 3,3,260,179
./racoon_images/raccoon-6.jpg 1,44,307,316
./racoon_images/raccoon-7.jpg 92,79,271,264
./racoon_images/raccoon-8.jpg 16,11,236,175
./racoon_images/raccoon-9.jpg 10,7,347,471
./racoon_images/raccoon-10.jpg 130,2,446,488
./racoon_images/raccoon-11.jpg 3,1,461,431
./racoon_images/raccoon-12.jpg 28,21,126,181,85,33,235,193
...
```
- ![raccoon-12.jpg](/files/3010819745248355667)

## Obeject Detection을 위한 알고리즘
### Two-Stage Detector
- 정답(Ground Truth) Bounding Box가 될 후보들을 먼저 생성하는 단계 존재(Region Proposal) -> 후보들 중 정답을 판별
- ![image.png](/files/3010821334595381762)
- 다수의 Bounding Box 후보들을 생성하기 위한 연산 때문에 실시간 Object Detection을 구현하기에는 속도가 너무 느림.

### One-Stage Detector
- Region Proposal 과정 없이 Deep Learning 알고리즘만으로 한 번에 Object Detection 수행.
- 대표적인 알고리즘으로 2016년 발표된 YOLO(You Only Look Once)가 존재.

## YOLO의 Architecture
- ![image.png](/files/3010825335194637938)
- Image를 예를 들어 13x13의 격자로 나누고 각 격자마다 3개의 Detector(= Anchor Box)를 설계.
- Image 내 존재하는 정답(Ground Truth) Object에 대해서 해당 Obeject의 중심에 위치한 격자에 속해있는 3개의 Detector 중 1개가 이 정답을 산출할 수 있도록 훈련시킴.
- 13\*13*3 = 507의 Detector중 1개를 제외한 506개의 Detector는 버림.
- 왜 격자별로 3개의 Detector를 사용하는가? 다양한 크기의 Object들을 모두 Detect할 수 있도록 하기 위함
- 큰 Object에 대한 Detector: 13\*13\*3 = 507개, 중간 크기: 26\*26\*3 = 2,028개, 작은 크기: 52\*52\*3 = 8,112개(총 10,647개)
- ![image.png](/files/3010825060871638425)

## YOLO v3 구동
- 학습한 Image 종류: person/bicycle/car/motorbike/aeroplane/bus/train/truck/boat/traffic light/fire hydrant/stop/sign/parking meter/bench/bird/cat/dog/horse/sheep/cow/elephant/bear/zebra/giraffe/backpack/umbrella/handbag/tie/suitcase/frisbee/skis/snowboard/sports ball/kite/baseball bat/baseball glove/skateboard/surfboard/tennis racket/bottle/wine glass/cup/fork/knife/spoon/bowl/banana/apple/sandwich/orange/broccoli/carrot/hot dog/pizza/donut/cake/chair/sofa/pottedplant/bed/diningtable/toilet/tvmonitor/laptop/mouse/remote/keyboard/cell phone/microwave/oven/toaster/sink/refrigerator/book/clock/vase/scissors/teddy bear/hair drier/toothbrush/
- 가중치 개수: 62,000,000개

- ![KakaoTalk_20210413_232331017.png](/files/2986874056206113170)
- ![KakaoTalk_20210413_232351298.png](/files/2986874086023699373)

# One-Stage Object Detection
- source: https://machinethink.net/blog/object-detection/
- The model can predict only one bounding box and so it has to choose one of the objects, but instead the box ends up somewhere in the middle. Actually what happens here makes perfect sense: the model knows there are two objects but it has only one bounding box to give away, so it compromises and puts the predicted box in between the two horses. The size of the box is also halfway between the sizes of the two horses.
- You may think, “This sounds easy enough to solve, let’s just add more bounding box detectors by giving the model additional regression outputs.” After all, if the model can predict N bounding boxes then it should be able find up to N objects, right? Sounds like a plan… but it doesn’t work. Even with a model that has multiple of these detectors, we still get bounding boxes that all end up in the middle of the image.
- Why does this happen? The problem is that the model doesn’t know which bounding box it should assign to which object, and to be safe it puts them both somewhere in the middle.
The model has no way to decide, “I can put bounding box 1 around the horse on the left, and bounding box 2 around the horse on the right.” Instead, each detector still tries to predict all of the objects rather than just one of them. Even though the model has N detectors, they don’t work together as a team. A model with multiple bounding box detectors still behaves exactly like the model that predicted only one bounding box.
- One-stage detectors such as YOLO, SSD, and DetectNet all solve this problem by assigning each bounding box detector to a specific position in the image. That way the detectors learn to specialize on objects in certain locations. For even better results, we can also let detectors specialize on the shapes and sizes of objects.
- The key thing here is that the position of a detector is fixed: it can only detect objects located near that cell (in fact, the object’s center must be inside the grid cell). This is what lets us avoid the problem from the previous section, where detectors had too much freedom. With this grid, a detector on the left-hand side of the image will never predict an object that is located on the right-hand side. We use the 13×13 grid as a spatial constraint, to make it easier for the model to learn how to predict objects. The model is trained so that the detectors in a given grid cell are responsible only for detecting objects whose center falls inside that grid cell. The naive version of the model did not have such constraints, and so its regression layers never got the hint to look only in specific places.
- Note that confidence score only says something about whether or not this is an object, but says nothing about what kind of object this is — that’s what the class probabilities are for. It tells us which predicted boxes we can ignore.
- Typically we’ll end up with a dozen or so predictions that the model thinks are good. Some of these will overlap — this happens because nearby cells may all make a prediction for the same object, and sometimes a single cell will make multiple predictions (although this is discouraged in training).
- NMS keeps the predictions with the highest confidence scores and removes any other boxes that overlap these by more than a certain threshold (say 60%).
- It’s much harder for a machine learning model to learn about images if we only use plain FC layers. The constraints imposed upon the convolutional layer — it looks only at a few pixels at a time, and the connections share the same weights — help the model to extract knowledge from images. We use these constraints to remove degrees of freedom and to guide the model into learning what we want it to learn.
- Why are there 5 detectors per grid cell instead of just one? Well, just like it’s hard for a detector to learn how to predict objects that can be located anywhere, it’s also hard for a detector to learn to predict objects that can be any shape or size. We use the grid to specialize our detectors to look only at certain spatial locations, and by having several different detectors per grid cell, we can make each of these object detectors specialize in a certain object shape as well.
- The anchors are nothing more than a list of widths and heights. Just like the grid puts a location constraint on the detectors, anchors force the detectors inside the cells to each specialize in a particular object shape. The first detector in a cell is responsible for detecting objects that are similar in size to the first anchor, the second detector is responsible for objects that are similar in size to the second anchor, and so on. Because we have 5 detectors per cell, we also have 5 anchors.
- The widths and heights of the anchors in the code snippet above are expressed in the 13×13 coordinate system of the grid, so the first anchor is a little over 1 grid cell wide and nearly 2 grid cells tall. The last anchor covers almost the entire grid at over 10×10 cells. This is how YOLO stores its anchors.
- YOLO chooses the anchors by running k-means clustering on all the bounding boxes from all the training images (with k = 5 so it finds the five most common object shapes). Therefore, YOLO’s anchors are specific to the dataset that you’re training (and testing) on. The k-means algorithm finds a way to divide up all data points into clusters. Here the data points are the widths and heights of all the ground-truth bounding boxes in the dataset. If we run k-means on the boxes from the Pascal VOC dataset, we find the following 5 clusters. These clusters represent five “averages” of the different object shapes that are present in this dataset. You can see that k-means found it necessary to group very small objects together in the blue cluster, slightly larger objects in the red cluster, and very large objects in green. It decided to split medium objects into two groups: one where the bounding boxes are wider than tall (yellow), and one that’s taller than wide (purple).
- We can run k-means several times on a different number of clusters, and compute the average IOU between the ground-truth boxes and the anchor boxes they are closest to. Not surprisingly, using more centroids (a larger value of k) gives a higher average IOU, but it also means we need more detectors in each grid cell and that makes the model run slower. For YOLO v2 they chose 5 anchors as a good trade-off between recall and model complexity.
- What the model predicts for each bounding box is not their absolute coordinates in the image but four “delta” values, or offsets:
  - delta_x, delta_y: the center of the box inside the grid cell
  - delta_w, delta_h: scaling factors for the width and height of the anchor box
- Each detector makes a prediction relative to its anchor box. The anchor box should already be a pretty good approximation of the actual object size (which is why we’re using them) but it won’t be exact. This is why we predict a scaling factor that says how much larger or smaller the box is than the anchor, as well as a position offset that says how far off the predicted box is from this grid center. It’s OK for the predicted box to be wider and/or taller than the original image, but it does not make sense for the box to have a negative width or height. That’s why we take the exponent of the predicted number.
- A key feature of YOLO is that it encourages a detector to predict a bounding box only if it finds an object whose center lies inside the detector’s grid cell. This helps to avoid spurious detections, so that multiple neighboring grid cells don’t all find the same object. To enforce this, delta_x and delta_y must be restricted to a number between 0 and 1 that is a relative position inside the grid cell. That’s what the sigmoid function is for.
- Recall that our example model always predicts 845 bounding boxes, no more, no less. But typically there will be only a few real objects in the image. During training we encourage only a single detector to make a prediction for each ground-truth, so there will be only a few predictions with a high confidence score. The predictions from the detectors that did not find an object — by far the most of them — should have a very low confidence score.
- Since most boxes will not contain any objects, we can now ignore all boxes whose confidence score is below a certain threshold (such as 0.3), and then perform non-maximum suppression on the remaining boxes to get rid of duplicates. We typically end up with anywhere between 1 and about 10 predictions.
- Now, even though I keep saying there are 5 detectors in each grid cell, for 845 detectors overall, the model really only learns five detectors in total — not five unique detectors per grid cell. This is because the weights of the convolution layer are the same at each position and are therefore shared between the grid cells.
- The model really learns one detector for every anchor. It slides these detectors across the image to get 845 predictions, 5 for each position on the grid. So even though we only have 5 unique detectors in total, thanks to the convolution these detectors are independent of where they are in the image and therefore can detect objects regardless of where they are located. This also explains why model always predicts where the bounding box is relative to the center of the grid cell. Due to the convolutional nature of this model, it cannot predict absolute coordinates. Since the convolution kernels slide across the image, their predictions are always relative to their current position in the feature map.
- Now the loss function needs to know which ground-truth object belongs to which detector in which grid cell, and likewise, which detectors do not have ground-truths associated with them. This is what we call “matching”.
- First we find the grid cell that the center of the bounding box falls in. That grid cell will be responsible for this object. If any other grid cells also predict this object they will be penalized for it by the loss function.
- The VOC annotations gives the bounding box coordinates as xmin, ymin, xmax, ymax. Since the model uses a grid and we decide which grid cell to use based on the center of the ground-truth box, it makes sense to convert the box coordinates to center x, center y, width, and height.
- Just picking the cell is not enough. Each grid cell has multiple detectors and we only want one of these detectors to find the object, so we pick the detector whose anchor box best matches the object’s ground-truth box. This is done with the usual IOU metric.
- Remember that the object doesn’t have to be the exact same size as the anchor, as the model predicts a position offset and size offset relative to the anchor box. The anchor box is just a hint.
- If a training image has 3 unique objects in it, and thus 3 ground-truth boxes, then only 3 of the 845 detectors are supposed to make a prediction and the other 842 detectors are supposed to predict “no object” (which in terms of our model output is a bounding box with a very low confidence score, ideally 0%).
- From now on, I’ll say positive example to mean a detector that has a ground-truth and negative example for a detector that does not have an object associated with it. A negative example is sometimes also called a “no object” or background.
- With classification we use the word “example” to refer to the training image as a whole but here it refers to an object inside an image, not to the image itself.
- YOLO solves this by first randomly shuffling the ground-truths, and then it just picks the first one that matches the cell. So if a new ground-truth box is matched to a cell that already is responsible for another object, then we simply ignore this new box. Better luck next epoch!
- This means that in YOLO at most one detector per cell is given an object — the other detectors in that cell are not supposed to detect anything (and are punished if they do).
- For detectors that are not supposed to detect an object, we will punish them when they predict a bounding box with a confidence score that is greater than 0.
- This part of the loss function only involves the confidence score — since there is no ground-truth box here, we don’t have any coordinates or class label to compare the prediction to.
This loss term is only computed for the detectors that are not responsible for detecting an object. If such a detector does find an object, this is where it gets punished.
- The no_object_scale is a hyperparameter. It’s typically 0.5, so that this part of the loss term doesn’t count as much as the other parts. Since the image will only have a handful of ground-truth boxes, most of the 845 detectors will only be punished by this “no object” loss and not any of the other loss terms I’ll show below.
- ecause we don’t want the model to learn only about “no objects”, this part of the loss shouldn’t become more important than the loss for the detectors that do have objects.
- For detectors that are responsible for finding an object, the no_object_loss is always 0. In SqueezeDet, the total no-object loss is also divided by the number of “no object” detectors to get the mean value but in YOLO we don’t do that.
- Why sum-squared-error and not mean-squared-error? I’m not 100% sure but it might be because every image has a different number of objects in it (positive examples). If we were to take the mean, then an image with just 1 object in it may end up with the same loss as an image with 10 objects. With SSE, that latter image will have a loss that is roughly 10 times larger, which might be considered more fair.
- The previous section described what happens to detectors that are not responsible for finding objects. The only thing they can do wrong is find an object where there is none.
- YOLO v3 and SSD take a different approach. They don’t see this as a multi-class classification problem but as a multi-label problem. Hence they don’t use softmax (which always chooses a single label to be the winner) but a logistic sigmoid, which allows multiple labels to be chosen. They use a standard binary cross-entropy to compute this loss term.
- The scale factor coord_scale is used to make the loss from the bounding box coordinate predictions count more heavily than the other loss terms. A typical value for this hyperparameter is 5.

# Selective Search
```python
_, regions = selectivesearch.selective_search(img_rgb, scale=100, min_size=2000)
```
```python
img_recs = cv2.rectangle(img=img_rgb_copy, pt1=(rect[0], rect[1]),
                                 pt2=(rect[0]+rect[2], rect[1]+rect[3]),
                                 color=green_rgb, thickness=2)
```

# `cv2`
```python
!pip install opencv-python
```
```python
import cv2
```
## `cv2.waitKey()`
```python
k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
```
## `cv2.VideoCapture()`
```python
cap = cv2.VideoCapture(0)
```
## `cv2.destroyAllWindows()`
## `cv2.rectangle()`
```python
for i, rect in enumerate(rects_selected):
    cv2.rectangle(img=img, pt1=(rect[0], rect[1]), pt2=(rect[0]+rect[2], rect[1]+rect[3]), color=(0, 0, 255), thickness=2)
```
## `cv2.circle()`
```python
for i, rect in enumerate(rects_selected):
    cv2.circle(img, (rect[0]+1, rect[1]-12), 12, (0, 0, 255), 2))
```
## `cv2.getTextSize()`
```python
(text_width, text_height), baseline = cv2.getTextSize(text=label, fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                                                  fontScale=font_scale, thickness=bbox_thick)
```
## `cv2.puttext()`
```python
cv2.putText(img=img, text=label, org=(x1, y1-4), fonFace=cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=font_scale, color=text_colors, thickness=bbox_thick, lineType=cv2.LINE_AA)
```
## `cv2.resize()`
```python
img_resized = cv2.resize(img, dsize=(640, 480), interpolation=cv2.INTER_AREA)
```
- `dsize` : (new_width, new_height)
## `cv2.cvtColor()`
```python
img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
```
## `cv2.imread()`
```python
img = cv2.imread("300.jpg")
```
## `cv2.imwrite()`
```python
cv2.imwrite("/content/drive/My Drive/Computer Vision/fire hydrants.png", ori_img)
```
## `cv2.imshow()`
```python
cv2.imshow("img_resized", img_resized)
```
## `cv2.findContours()`
```python
mask = cv2.inRange(hsv,lower_blue,upper_blue)
contours, hierachy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
```
## `cv2.TERM_CRITERIA_EPS`, `cv2.TERM_CRITERIA_MAX_ITER`
```python
criteria = (type, max_iter, epsilon)
```
## `CV2.KMEANS_RANDOM_CENTERS`
```python
flags = cv2.KMEANS_RANDOM_CENTERS
```
- 초기 중심점을 랜덤으로 설정.
```python
compactness, labels, centers = cv2.kmeans(z, 3, None, criteria, 10, flags)
```

# `Image`
```python
from PIL import Image
```
## `Image.open()`
```python
img = Image.open("20180312000053_0640 (2).jpg")
```
### `img.size`
### `img.save()`
### `img.thumbnail()`
```python
img.thumbnail((64, 64))
```
### `img.crop()`
```python
img_crop = img.crop((100, 100, 150, 150))	
```
### `img.resize()`
```python
img = img.resize((600, 600))
```
### `img.convert()`
```python
img.convert("L")
```
- (`"RGB"`, `"RGBA"`, `"CMYK"`, `"L"`, `"1"`)
### `img.paste()`
```python
img1.paste(img2, (20,20,220,220))
```
- img2.size와 동일하게 두 번째 parameter 설정.	
## `Image.new()`
```python
mask = Image.new("RGB", icon.size, (255, 255, 255))
```

# `colorsys`
## `colorsys.hsv_to_rgb()`
```python
hsv_tuples = [(idx/n_clss, 1, 1) for idx in idx2cls.keys()]
colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
colors = list(map(lambda x: (int(x[0]*255), int(x[1]*255), int(x[2]*255)), colors))
```