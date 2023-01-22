- Source: https://jonathan-hui.medium.com/ssd-object-detection-single-shot-multibox-detector-for-real-time-processing-9bd8deac0e06
- SSD speeds up the process by eliminating the need for the region proposal network.
- SSD uses VGG16 to extract feature maps. Then it detects objects using the Conv4_3 layer. For illustration, we draw the Conv4_3 to be 8 × 8 spatially (it should be 38 × 38). For each cell (also called location), it makes 4 object predictions.

# Multi-scale feature maps for detection
- At first, we describe how SSD detects objects from a single layer. Actually, it uses multiple layers (multi-scale feature maps) to detect objects independently. As CNN reduces the spatial dimension gradually, the resolution of the feature maps also decrease. SSD uses lower resolution layers to detect larger scale objects. For example, the 4× 4 feature maps are used for larger scale object. 
- SSD adds 6 more auxiliary convolution layers after the VGG16. Five of them will be added for object detection. In three of those layers, we make 6 predictions instead of 4. In total, SSD makes 8732 predictions using 6 layers.
- ![ssd_architecture](https://miro.medium.com/max/720/1*up-gIJ9rPkHXUGRoqWuULQ.webp)

# Default Boundary Boxes
- The default boundary boxes are equivalent to anchors in Faster R-CNN. 
- Conceptually, the ground truth boundary boxes can be partitioned into clusters with each cluster represented by a default boundary box (the centroid of the cluster). So, instead of making random guesses, we can start the guesses based on those default boxes. 
- To keep the complexity low, the default boxes are pre-selected manually and carefully to cover a wide spectrum of real-life objects. 
- Now, instead of using global coordination for the box location, the boundary box predictions are relative to the default boundary boxes at each cell (∆cx, ∆cy, ∆w, ∆h), i.e. the offsets (difference) to the default box at each cell for its center (cx, cy), the width and the height. 
- For each feature map layers, it shares the same set of default boxes centered at the corresponding cell. But different layers use different sets of default boxes to customize object detections at different resolutions. The 4 green boxes below illustrate 4 default boundary boxes.
## Choosing Default Boundary Boxes
- Default boundary boxes are chosen manually. SSD defines a scale value for each feature map layer. Starting from the left, Conv4_3 detects objects at the smallest scale 0.2 (or 0.1 sometimes), and then increases linearly to the rightmost layer at a scale of 0.9. Combining the scale value with the target aspect ratios, we compute the width and the height of the default boxes. For layers making 6 predictions, SSD starts with 5 target aspect ratios: 1, 2, 3, 1/2, and 1/3.

# Matching strategy 
- SSD predictions are classified as positive matches or negative matches. SSD only uses positive matches in calculating the localization cost (the mismatch of the boundary box). If the corresponding default boundary box (not the predicted boundary box) has an IoU greater than 0.5 with the ground truth, the match is positive. Otherwise, it is negative.
- Once we identify the positive matches, we use the corresponding predicted boundary boxes to calculate the cost. This matching strategy nicely partitions what shape of the ground truth that a prediction is responsible for.
- This matching strategy encourages each prediction to predict shapes closer to the corresponding default box. Therefore our predictions are more diverse and more stable in the training.

# Multi-scale feature maps & default boundary boxes
- ![multi_scale_feature_maps](https://miro.medium.com/max/4800/1*-KVIXjvBO5m2MQZrzWx-wg.webp)
- Here is an example of how SSD combines multi-scale feature maps and default boundary boxes to detect objects at different scales and aspect ratios. The dog below matches one default box (in red) in the 4 × 4 feature map layer, but not any default boxes in the higher resolution 8 × 8 feature map. The cat which is smaller is detected only by the 8 × 8 feature map layer in 2 default boxes (in blue). 
- Higher-resolution feature maps are responsible for detecting small objects. The first layer for object detection conv4_3 has a spatial dimension of 38 × 38, a pretty large reduction from the input image. Hence, SSD usually performs badly for small objects comparing with other detection methods.

# Loss function 
- The localization loss is the mismatch between the ground truth box and the predicted boundary box. SSD only penalizes predictions from positive matches. We want the predictions from the positive matches to get closer to the ground truth. Negative matches can be ignored.
- The confidence loss is the loss of making a class prediction. For every positive match prediction, we penalize the loss according to the confidence score of the corresponding class. For negative match predictions, we penalize the loss according to the confidence score of the class “0”: class “0” classifies no object is detected. 

# Hard negative mining 
- However, we make far more predictions than the number of objects present. So there are many more negative matches than positive matches. This creates a class imbalance that hurts training. We are training the model to learn background space rather than detecting objects. However, SSD still requires negative sampling so it can learn what constitutes a bad prediction. So, instead of using all the negatives, we sort those negatives by their calculated confidence loss. SSD picks the negatives with the top loss and makes sure the ratio between the picked negatives and positives is at most 3:1. This leads to faster and more stable training.

# Non-maximum suppression (NMS)
- SSD uses non-maximum suppression to remove duplicate predictions pointing to the same object. SSD sorts the predictions by the confidence scores. Start from the top confidence prediction, SSD evaluates whether any previously predicted boundary boxes have an IoU higher than 0.45 with the current prediction for the same class. If found, the current prediction will be ignored. At most, we keep the top 200 predictions per image.

# Key observations
- SSD performs worse than Faster R-CNN for small-scale objects. In SSD, small objects can only be detected in higher resolution layers (leftmost layers). But those layers contain low-level features, like edges or color patches, that are less informative for classification.
- Accuracy increases with the number of default boundary boxes at the cost of speed.
- Multi-scale feature maps improve the detection of objects at a different scale.
- Design better default boundary boxes will help accuracy.
- COCO dataset has smaller objects. To improve accuracy, use smaller default boxes (start with a smaller scale at 0.15).