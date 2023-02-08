# Region Proposal
## Sliding Window Algorithm
- In the sliding window approach, we slide a box or window over an image to select a patch and classify each image patch covered by the window using the object recognition model. It is an exhaustive search for objects over the entire image. *Not only do we need to search all possible locations in the image, we have to search at different scales.* This is because object recognition models are generally trained at a specific scale (or range of scales). This results into classifying tens of thousands of image patches.
- The problem doesn’t end here. Sliding window approach is good for fixed aspect ratio objects such as faces or pedestrians. Images are 2D projections of 3D objects. Object features such as aspect ratio and shape vary significantly based on the angle at which image is taken. *The sliding window approach is computationally very expensive when we search for multiple aspect ratios.*
## Selective Search Algorithm
- The problems we have discussed so far can be solved using region proposal algorithms. These methods take an image as the input and output bounding boxes corresponding to all patches in an image that are most likely to be objects. *These region proposals can be noisy, overlapping and may not contain the object perfectly but amongst these region proposals, there will be a proposal which will be very close to the actual object in the image. We can then classify these proposals using the object recognition model. The region proposals with the high probability scores are locations of the object.*
- Region proposal algorithms identify prospective objects in an image using segmentation. *In segmentation, we group adjacent regions which are similar to each other based on some criteria such as color, texture etc. Unlike the sliding window approach where we are looking for the object at all pixel locations and at all scales, region proposal algorithm work by grouping pixels into a smaller number of segments. So the final number of proposals generated are many times less than sliding window approach. This reduces the number of image patches we have to classify. These generated region proposals are of different scales and aspect ratios.*
- An important property of a region proposal method is to have a very high recall. This is just a fancy way of saying that the regions that contain the objects we are looking have to be in our list of region proposals. To accomplish this our list of region proposals may end up having a lot of regions that do not contain any object. In other words, *It is ok for the region proposal algorithm to produce a lot of false positives so long as it catches all the true positives.* Most of these false positives will be rejected by object recognition algorithm. The time it takes to do the detection goes up when we have more false positives and the accuracy is affected slightly. *But having a high recall is still a good idea because the alternative of missing the regions containing the actual objects severely impacts the detection rate.*
- Selective Search starts by over-segmenting the image based on intensity of the pixels using a graph-based segmentation method by Felzenszwalb and Huttenlocher. The output of the algorithm is shown below. The image on the right contains segmented regions represented using solid colors.
- Can we use segmented parts in this image as region proposals? The answer is no and there are two reasons why we cannot do that. Most of the actual objects in the original image contain 2 or more segmented parts. Region proposals for occluded objects such as the plate covered by the cup or the cup filled with coffee cannot be generated using this method.
- If we try to address the first problem by further merging the adjacent regions similar to each other we will end up with one segmented region covering two objects. Perfect segmentation is not our goal here. We just want to predict many region proposals such that some of them should have very high overlap with actual objects. Selective search uses oversegments from Felzenszwalb and Huttenlocher’s method as an initial seed. An oversegmented image looks like this.
- *At each iteration, larger segments are formed and added to the list of region proposals. Hence we create region proposals from smaller segments to larger segments in a bottom-up approach.*
- Using `selectivesearch.selective_search()`
	- Reference: https://velog.io/@tataki26/Selective-Search-%EC%8B%A4%EC%8A%B5-%EB%B0%8F-%EC%8B%9C%EA%B0%81%ED%99%94
	```python
	from selectivesearch import selective_search
	# `scale`: Free parameter. Higher means larger clusters in felzenszwalb segmentation.
	# `sigma` Width of Gaussian kernel for felzenszwalb segmentation.
    # `min_size` Minimum component size for felzenszwalb segmentation.
	# `regions`: List of Dictionary such that `{'labels`: 해당 bbox 내에 존재하는 Objects의 고유 ID, 'rect': (left_top_x, left_top_y, width, height), 'size': component_size}`
	_, regions = selective_search(im_orig, scale, min_size)

	```

# Anchor Box
- 핵심은 사전에 크기와 비율이 모두 결정되어 있는 박스를 전제하고, 학습을 통해서 이 박스의 위치나 크기를 세부 조정하는 것을 말합니다.
- 아예 처음부터 중심점의 좌표와 너비, 높이를 결정하는 방식보다 훨씬 안정적으로 학습이 가능합니다.
- 앵커 박스는 적당히 직관적인 크기의 박스로 결정하고, 비율을 1:2, 1:1, 2:1로 설정하는 것이 일반적이었습니다만, yolo v2는 여기에 learning algorithm을 적용합니다.
- 기존의 yolo가 그리드의 중심점을 예측했다면, yolov2에서는 left top 꼭지점으로부터 얼만큼 이동하는 지를 예측합니다. 이것이 bx=σ(tx) + cx가 의미하는 바입니다.

# Object Detection
	- Obeject Detection은 왜 어려운가?
	- Classification + Regression 동시에 수행
		- 어떤 Object의 Class가 무엇인지 판별(Classification)
		- 해당 Object의 둘러싸는 Bounding Box를 정의하는 4개의 변수(x, y, w, h) 계산(Regression)
	- Object가 너무 다양함
		- 크기도 천차만별, 색깔도 천차만별, Image 내 위치도 천차만별
	- 실시간 Detection을 위해서는 연산 속도도 중요
		- 상용 가능한 수준이 되기 위해서는 1초 미만의 시간 내에 결과를 도출해야 함
	- 인간이 보기에도 애매한 상황이 다수
		- 전체 Image에서 Object가 차지하는 비중이 매우 적고 대부분이 Background임.
		- 다수의 Object들이 서로 겹쳐져 있음.
		- 어떤 Object의 Bounding Box는 유일하지 않음.
	- AI Model의 학습에 필요한 데이터 셋의 부족
		- Image를 수집하는 것뿐만 아니라 각각에 대해 Annotation을 만들어야 함.
## Two-Stage Detector
### R-CNN (Region-based CNN)
- Selective Search
- Region proposal by selective search
- Warping, Croping
### Fast R-CNN
- ROI Pooling
### Faster R-CNN
- Region proposal by region proposal network (RPN)
## One-Stage Detector
- Source: https://machinethink.net/blog/object-detection/
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

# YOLO v3
- Image를 예를 들어 13x13의 격자로 나누고 각 격자마다 3개의 Detector(= Anchor Box)를 설계.
- Image 내 존재하는 정답(Ground Truth) Object에 대해서 해당 Obeject의 중심에 위치한 격자에 속해있는 3개의 Detector 중 1개가 이 정답을 산출할 수 있도록 훈련시킴.
- 13\*13*3 = 507의 Detector중 1개를 제외한 506개의 Detector는 버림.
- 왜 격자별로 3개의 Detector를 사용하는가? 다양한 크기의 Object들을 모두 Detect할 수 있도록 하기 위함
- 큰 Object에 대한 Detector: 13\*13\*3 = 507개, 중간 크기: 26\*26\*3 = 2,028개, 작은 크기: 52\*52\*3 = 8,112개(총 10,647개)
- Here `B` is the number of bounding boxes a cell on the feature map can predict.
- `5` is for the 4 bounding box attributes and one object confidence, and `C` is the number of classes.
- If trained on COCO, `B` = 3 and `C` = 80, so the kernel size is `1x1x255`.
## v1
- Feature Extractor: Inception의 변형
- Number of Anchor Boxes per Grid: 2
- Size of Output Feature Map: 7x7
- Number of bounding boxes: 7x7x2
- Size of output tensor: 7\*7\*(5\*B + C)
## v2
- Feature Extractor: Darknet 19
- Number of Anchor Boxes per Grid: 2
- Size of Output Feature Map: 13x13
- 13\*13\*(B\*(5 + C))
- Batch normalizations are used
- No fully connected layer and instead anchor boxes to predict bounding boxes. It results higher recall from 81% to 88%
- left top 꼭지점으로부터 얼만큼 이동하는 지를 예측합니다. 이것이 bx=σ(tx) + cx가 의미하는 바입니다. 다음으로 너비와 높이는 사전에 정의된 박스의 크기를 얼만큼 비율로 조절할 지를 지수승을 통해 예측
즉, 신경망은 물체의 최종 크기를 예측하는 것이 아니고, 가장 크기가 비슷한 anchor 에서 물체크기로의 조정값을 예측한다.
## v3
- Feature Extractor: Darknet 53
- Number of Anchor Boxes per Grid: 3\*3
- Size of Output Feature Map: 13x13, 26x26, 52x52
- 13\*13\*(B\*(5 + C)) + 26\*26\*(B\*(5 + C)) + 52\*52\*(B\*(5 + C))
- Uses FPN(Feature Pyramid Network)
- The most salient feature of v3 is that it makes detections at three different scales the
detection is done by applying 1 x 1 detection kernels on feature maps of three different sizes at three different places in the network
- Softmax classes rests on the assumption that classes are mutually exclusive, or in simple words, if an object belongs to one class, then it cannot belong to the other. This works fine in COCO dataset
- The first detection is made by the 82nd layer.
    - For the first 81 layers(12 convolutional layers and 23 residual blocks which has 2 convoluitional layers and 1 skip connection, 81 = 12 + 23*(1 + 2)), the image is down sampled by the network, such that the 81st layer has a stride of 32.
    - If we have an image of 416 x 416, the resultant feature map would be of size 13 x 13. One detection is made here using the 1 x 1 detection kernel, giving us a detection feature map of 13 x 13 x 255.
    - Then, the feature map from layer 79 is subjected to a few convolutional layers before being up sampled by 2x to dimensions of 26 x 26.
    - This feature map is then depth concatenated with the feature map from layer 61.
    - Then the combined feature maps is again subjected a few 1 x 1 convolutional layers to fuse the features from the earlier layer (61).
- Then, the second detection is made by the 94th layer
    - yielding a detection feature map of 26 x 26 x 255
    - A similar procedure is followed again, where the feature map from layer 91 is subjected to few
    - convolutional layers before being depth concatenated with a feature map from layer 36.
    - Like before, a few 1 x 1 convolutional layers follow to fuse the information from the previous layer (36).
- Then, make the final of the 3 at 106th layer
    - yielding feature map of size 52 x 52 x 255
- large: [None, 13, 13, 255] 
- medium: [None, 26, 26, 255]
- small: [None, 52, 52, 255]
## Non-Maximum Suppression (NMS)
- A post-processing step we filter out the boxes whose score falls below a certain threshold.
- Object Detection 알고리즘은 object 가 있을만한 위치에 많은 Detection 을 수행하는 경향이 강함.
- NMS 는 Detect 된 object 의 bounding box 중에 비슷한 위치에 있는 box 를 제거하고 가장 적합한 box를 선택하는 기법.
- 1. Detect 된 bounding box 별로 특정 confidence thrsd 이하 bounding box 는 먼저 제거(confidence < 0.5)
2. 가장 높은 confidence score 를 가진 box 순으로 내림차순 정렬 후 아래 로직을 모든 box 에 순차적으로 적용 → 높은 confidence score 를 가진 box 와 겹치는 다른 box 를 모두 조사하여 IOU 가 특정 thrsd 이상인 Box를 모두 제거 (IOU thrsd >0.4)
3. 남아 있는 box 만 선택
- Confidence가 높을수록, IOU thrsd가 낮을수록 많은 box 가 제거됨.
- Implementation
	```python
	bboxes = np.array(bboxes)
	clss_in_img = list(set(bboxes[:, 5]))
	best_bboxes = []
	for cls in clss_in_img:
		bboxes_cls = bboxes[bboxes[:, 5] == cls]
		# Process 1: Determine whether the number of bounding boxes is greater than 0 
		while len(bboxes_cls) > 0:
			# Process 2: Select the bounding box with the highest score according to socre order A
			argmax = np.argmax(bboxes_cls[:, 4])
			best_bbox = bboxes_cls[argmax]
			best_bboxes.append(best_bbox)

			bboxes_cls = np.delete(bboxes_cls, argmax, axis=0)

			# Process 3: Calculate this bounding box A and remain all iou of the bounding box and remove
			# those bounding boxes whose iou value is higher than the thrsd.
			ious = np.array([compute_giou(best_bbox[:4], bbox_cls[:4]) for bbox_cls in bboxes_cls])

			bboxes_cls = bboxes_cls*(ious <= 0.45)[:, None]
			bboxes_cls = bboxes_cls[bboxes_cls[:, 4] > 0]

	bboxes = best_bboxes
	```
## Confidence Score
-  Each box also has a confidence score that says how likely the model thinks this box really contains an object.
- Confidence score: Bounding box 안에 object 가 있을 확률이 얼마나 되는지, 그리고 object 가 class 를 정확하게 예측했는지 나타내는 지표
- Confidence Score가 낮을수록 Bounding Box를 많이 만듦. Precision 감소, Recall 증가.
