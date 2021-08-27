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