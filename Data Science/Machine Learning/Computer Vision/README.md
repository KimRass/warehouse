Written by KimRass
# Datasets
## CIFAR-10
```python
import ssl

ssl._create_default_https_context = ssl._create_unverified_context
```
```python
(X_tr, y_tr), (X_te, y_te) = tf.keras.datasets.cifar10.load_data()
y_tr_val = to_categorical(y_tr_val)
y_te = to_categorical(y_te)

classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
```
## Fashion MNIST
```python
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
```
## COCO (Common Objects in COntext)
- 83 classes: `"aeroplane"`, `"apple"`, `"backpack"`, `"banana"`, `"baseball bat"`, `"baseball glove"`, `"bear"`, `"bed"`, `"bench"`, `"bicycle"`, `"bird"`, `"boat"`, `"book"`, `"bottle"`, `"bowl"`, `"broccoli"`, `"bus"`, `"cake"`, `"car"`, `"carrot"`, `"cat"`, `"cell phone"`, `"chair"`, `"clock"`, `"cow"`, `"cup"`, `"diningtable"`, `"dog"`, `"dog"`, `"donut"`, `"elephant"`, `"fire hydrant"`, `"fork"`, `"frisbee"`, `"giraffe"`, `"glass"`, `"hair drier"`, `"handbag"`, `"horse"`, `"hot"`, `"keyboard"`, `"kite"`, `"knife"`, `"laptop"`, `"microwave"`, `"motorbike"`, `"mouse"`, `"orange"`, `"oven"`, `"parking meter"`, `"person"`, `"pizza"`, `"pottedplant"`, `"refrigerator"`, `"remote"`, `"sandwich"`, `"scissors"`, `"sheep"`, `"sign"`, `"sink"`, `"skateboard"`, `"skis"`, `"snowboard"`, `"sofa"`, `"spoon"`, `"sports ball"`, `"stop"`, `"suitcase"`, `"surfboard"`, `"teddy bear"`, `"tennis racket"`, `"tie"`, `"toaster"`, `"toilet"`, `"toothbrush"`, `"traffic light"`, `"train"`, `"truck"`, `"tvmonitor"`, `"umbrella"`, `"vase"`, `"wine"`, `"zebra"`

# ILSVRC (ImageNet Large Scale Visual Recognition Challenge)

# Tasks
## Image Classification (= Object Recognition)
### The Evolution of Image Classification
- Source: https://stanford.edu/~shervine/blog/evolution-image-classification-explained
- LeNet -> AlexNet -> VGGNet -> GoogLeNet -> ResNet -> DenseNet
## Object Localization
- Source: https://www.einfochips.com/blog/understanding-object-localization-with-deep-learning/#:~:text=Image%20localization%20is%20a%20spin,around%20an%20object%20of%20interest.
- Object localization is a regression problem *where the output is x and y coordinates around the object of interest to draw bounding boxes.*
### Object Detection
- *Object detection is a complex problem that combines the concepts of image localization and classification. Given an image, an object detection algorithm would return bounding boxes around all objects of interest and assign a class to them.*

# Feature Map (= Activation Map)
- Source: https://ujjwalkarn.me/2016/08/11/intuitive-explanation-convnets/
- In CNN terminology, the 3×3 matrix is called a "filter" or "kernel" or "feature detector" and the matrix formed by sliding the filter over the image and computing the dot product is called the "Activation Map" or the 
Feature Map" .It is important to note that filters acts as feature detectors from the original input image.
- ***It is evident from the animation above that different values of the filter matrix will produce different feature maps for the same input image.
- In the table below, we can see the effects of convolution of the above image with different filters. As shown, we can perform operations such as edge detection, sharpen and blur just by changing the numeric values of our filter matrix before the convolution operation – this means that different filters can detect different features from an image, for example edges, curves etc.
- It is important to note that the Convolution operation captures the local dependencies in the original image.

# AlexNet

# VGGNet (VGG16)

# GoogLeNet
- Paper: https://www.cs.unc.edu/~wliu/papers/GoogLeNet.pdf
- Source: https://www.geeksforgeeks.org/understanding-googlenet-model-cnn-architecture/
- The novel architecture was an Inception Network, and a variant of this Network called, GoogLeNet went on to achieve the state of the art performance in the classification computer vision task of the ImageNet LargeScale Visual Recognition Challenge 2014(ILVRC14).
- Global Average Pooling (GAP)
	- *This also decreases the number of trainable parameters to 0 and improves the top-1 accuracy by 0.6%*.
- Auxiliary Classifier for Training:
	- ***Inception architecture used some intermediate classifier branches in the middle of the architecture, these branches are used during training only.*** These branches consist of a 5×5 average pooling layer with a stride of 3, a 1×1 convolutions with 128 filters, two fully connected layers of 1024 outputs and 1000 outputs and a softmax classification layer. The generated loss of these layers added to total loss with a weight of 0.3. ***These layers help in combating gradient vanishing problem and also provide regularization.***
	- Two auxiliary classifier layer connected to the output of Inception (4a) and Inception (4d) layers.
	- Implementation
		```python
		def auxiliary_classifier(x, name):
			z = AveragePooling2D(pool_size=5, strides=3, padding="valid")(x)
			z = Conv2D(filters=128, kernel_size=1, strides=1, padding="same", activation="relu")(z)
			z = Flatten()(z)
			z = Dense(units=1024, activation="relu")(z)
			
			return Dense(units=1000, activation="softmax", name=name)(z)
		```
## Inception Network
- Sources: https://towardsdatascience.com/deep-learning-understand-the-inception-module-56146866e652, https://hacktildawn.com/2016/09/25/inception-modules-explained-and-implemented/
- An inception network is a deep neural network with an architectural design that consists of repeating components referred to as Inception modules.
- ***Convolutional neural networks benefit from extracting features at varying scales. The biological human visual cortex functions by identifying patterns at different scales, which accumulates to form lager perceptions of objects. Therefore multi-scale convnet have the potential to learn more.*** Large networks are prone to overfitting, and chaining multiple convolutional operations together increases the computational cost of the network.
- ***Although a 1x1 filter does not learn any spatial patterns that occur within the image, it does learn patterns across the depth(cross channel) of the image. 1x1 convolutions reduce the dimensions of inputs within the network. 1x1 convolutions are configured to have a reduced amount of filters, so the outputs typically have a reduced amount of channels in comparison to the initial input.***
- *Within a convnet, different conv filter sizes learn spatial patterns and detect features at varying scales.*
- 1x1 learns patterns across the depth of the input. 3x3 and 5x5 learns spatial patterns across all dimensional components (height, width and depth) of the input.
- ***There is an increase in representational power when combining all the patterns learned from the varying filter sizes. The Inception module consists of a concatenation layer, where all the outputs and feature maps from the conv filters are combined into one object to create a single output of the Inception module.***
- Naive Inception Module
	- ![Naive Inception Module](https://hackathonprojects.files.wordpress.com/2016/09/naive.png)
	- Pooling downsamples the input data to create a smaller output with a reduced height and width.
	- Within an Inception module, we add padding(same) to the max-pooling layer to ensure it maintains the height and width as the other outputs(feature maps) of the convolutional layers within the same Inception module. By doing this, we ensure we can concatenate the outputs of the max-pooling layer with the outputs of the conv layers within the concatenation layer.
	- Implementation
		```python
		def naive_inception_module(x, filters): 
			z1 = Conv2D(filters=filters[0], kernel_size=1, strides=1, padding="same", activation="relu")(x)
			z2 = Conv2D(filters=filters[1], kernel_size=3, strides=1, padding="same", activation="relu")(x)
			z3 = Conv2D(filters=filters[2], kernel_size=5, strides=1, padding="same", activation="relu")(x)
			z4 = MaxPool2D(pool_size=3, strides=1, padding="same")(x)
			return Concatenate(axis=-1)([z1, z2, z3, z4])
		```
- Inception Module
	- ![Inception Module](https://hackathonprojects.files.wordpress.com/2016/09/inception_implement.png)
	- Implementation
		```python
		def inception_module(x, filters): 
			z1 = Conv2D(filters=filters[0], kernel_size=1, strides=1, padding="same", activation="relu")(x)

			z2 = Conv2D(filters=filters[3], kernel_size=1, strides=1, padding="same", activation="relu")(x)
			z2 = Conv2D(filters=filters[1], kernel_size=3, strides=1, padding="same", activation="relu")(z2)

			z3 = Conv2D(filters=filters[3], kernel_size=1, strides=1, padding="same", activation="relu")(x)
			z3 = Conv2D(filters=filters[2], kernel_size=5, strides=1, padding="same", activation="relu")(z3)

			z4 = MaxPool2D(pool_size=3, strides=1, padding="same")(x)
			z4 = Conv2D(filters=filters[3], kernel_size=1, strides=1, padding="same", activation="relu")(z4)
			return Concatenate(axis=-1)([z1, z2, z3, z4])
		```
- ![GoogLeNet Architecture](https://i.stack.imgur.com/Xqv0n.png)
- Implementation
	```python
	inputs = Input(shape=(224, 224, 3))

	z = Conv2D(filters=64, kernel_size=7, strides=2, padding="same", activation="relu")(inputs)
	z = MaxPool2D(pool_size=3, strides=2, padding="same")(z)
	z = BatchNormalization()(z)
	z = Conv2D(filters=64, kernel_size=1, strides=1, padding="same", activation="relu")(z)
	z = Conv2D(filters=192, kernel_size=3, strides=1, padding="same", activation="relu")(z)
	z = BatchNormalization()(z)
	z = MaxPool2D(pool_size=3, strides=2, padding="same")(z)
	z = inception_module(z, [64, 128, 32, 32]) # inception 3a
	z = inception_module(z, [128, 192, 96, 64]) # inception 3b
	z = MaxPool2D(pool_size=3, strides=2, padding="same")(z)
	z = inception_module(z, [192, 208, 48, 64]) # inception 4a
	
	outputs1 = auxiliary_classifier(z, "outputs1")

	z = inception_module(z, [160, 224, 64, 64]) # inception 4b
	z = inception_module(z, [128, 256, 64, 64]) # inception 4c
	z = inception_module(z, [112, 288, 64, 64]) # inception 4d
	
	outputs2 = auxiliary_classifier(z, "outputs2")

	z = inception_module(z, [256, 320, 128, 128]) # inception 4e
	z = MaxPool2D(pool_size=3, strides=2, padding="same")(z)
	z = inception_module(z, [256, 320, 128, 128]) # inception 5a
	z = inception_module(z, [384, 384, 128, 128]) # inception 5b
	z = GlobalAveragePooling2D()(z)
	z = Dropout(rate=0.4)(z)
	z = Flatten()(z)

	outputs3 = Dense(units=1000, activation="softmax")(z)

	model = Model(inputs=inputs, outputs=[outputs1, outputs2, outputs3])
	```

# ResNet (Residual Neural Network)
- Source: https://en.wikipedia.org/wiki/Residual_neural_network, https://www.analyticsvidhya.com/blog/2021/08/all-you-need-to-know-about-skip-connections/#:~:text=Skip%20Connections%20(or%20Shortcut%20Connections,input%20to%20the%20next%20layers.&text=Neural%20networks%20can%20learn%20any,%2Ddimensional%20and%20non%2Dconvex, https://www.analyticsvidhya.com/blog/2021/06/understanding-resnet-and-analyzing-various-models-on-the-cifar-10-dataset/#h2_3
- Residual Networks were proposed in 2015 to solve the image classification problem. ****In ResNets, the information from the initial layers is passed to deeper layers by matrix addition. This operation doesn’t have any additional parameters as the output from the previous layer is added to the layer ahead.
## Skip Connection
- ***There are two main reasons to add skip connections: to avoid the problem of vanishing gradients, or to mitigate the Degradation (accuracy saturation) problem; where adding more layers to a suitably deep model leads to higher training error.***
- *Skipping effectively simplifies the network, using fewer layers in the initial training stages. This speeds learning by reducing the impact of vanishing gradients, as there are fewer layers to propagate through.*
- *While training deep neural nets, the performance of the model drops down with the increase in depth of the architecture. This is known as the degradation problem.*
- From this construction, *the deeper network should not produce any higher training error than its shallow counterpart because we are actually using the shallow model’s weight in the deeper network with added identity layers. But experiments prove that the deeper network produces high training error comparing to the shallow one. This states the inability of deeper layers to learn even identity mappings.*
- The degradation of training accuracy indicates that not all systems are similarly easy to optimize.
- *One of the primary reasons is due to random initialization of weights with a mean around zero, L1, and L2 regularization.  As a result, the weights in the model would always be around zero and thus the deeper layers can’t learn identity mappings as well.* Here comes the concept of skip connections which would enable us to train very deep neural networks.
- Skip Connections (or Shortcut Connections) as the name suggests skips some of the layers in the neural network and feeds the output of one layer as the input to the next layers.
- Skip Connections were introduced to solve different problems in different architectures. In the case of ResNets, skip connections solved the degradation problem that we addressed earlier.
- As you can see here, the loss surface of the neural network with skip connections is smoother and thus leading to faster convergence than the network without any skip connections.
- Skip Connections can be used in 2 fundamental ways in Neural Networks: Addition and Concatenation.
- One problem that may happen is regarding the dimensions. *Sometimes the dimensions of `x` and `F(x)` may vary and this needs to be solved.* Two approaches can be followed in such situations. *One involves padding the input x with weights such as it now brought equal to that of the value coming out. The second way includes using a convolutional layer from `x` to addition to `F(x)`*.
## Bottleneck Design
- Source: https://towardsdatascience.com/review-resnet-winner-of-ilsvrc-2015-image-classification-localization-detection-e39402bfa5d8
- Since the network is very deep now, the time complexity is high. A bottleneck design is used to reduce the complexity.
- The 1×1 conv layers are added to the start and end of network. This is a technique suggested in Network In Network and GoogLeNet (Inception-v1). ***It turns out that 1×1 conv can reduce the number of connections (parameters) while not degrading the performance of the network so much. (Please visit my review if interested.)***
- *With the bottleneck design, 34-layer ResNet become 50-layer ResNet. And there are deeper network with the bottleneck design: ResNet-101 and ResNet-152.*
- ![ResNet Architecture](https://neurohive.io/wp-content/uploads/2019/01/resnet-architectures-34-101.png)
- Implementation of 50-layered ResNet (for CIFAR-10)
	```python
	def residual_block(x, filters): 
		z1 = Conv2D(filters=filters[0], kernel_size=1, strides=1, padding="same")(x) 
		z1 = BatchNormalization()(z1)
		z1 = Activation("relu")(z1)
		z1 = Conv2D(filters=filters[0], kernel_size=3, strides=1, padding="same")(z1)
		z1 = BatchNormalization()(z1)
		z1 = Activation("relu")(z1)
		z1 = Conv2D(filters=filters[1], kernel_size=1, strides=1, padding="same")(z1)
		z1 = BatchNormalization()(z1)
		z1 = Activation("relu")(z1)

		z2 = Conv2D(filters=filters[1], kernel_size=1, strides=1, padding="same")(x)
		z2 = BatchNormalization()(z2)

		z = Activation("relu")(z1 + z2)
		return z

	inputs = Input(shape=(32, 32, 3))

	z = Conv2D(filters=64, kernel_size=7, strides=2, padding="valid")(inputs)
	z = MaxPool2D(pool_size=3, strides=2, padding="same")(z)
	for _ in range(3):
		z = residual_block(z, filters=[64, 256])
	for _ in range(4):
		z = residual_block(z, filters=[128, 512])
	for _ in range(6):
		z = residual_block(z, filters=[256, 1024])
	for _ in range(3):
		z = residual_block(z, filters=[512, 2048])
	z = GlobalAveragePooling2D()(z) 

	outputs = Dense(units=10, activation="softmax")(z)

	model = Model(inputs=inputs, outputs=outputs)
	```
	
# SENet (Squeeze-and-Excitation Network)
- Source: https://towardsdatascience.com/squeeze-and-excitation-networks-9ef5e71eacd7
- Squeeze-and-Excitation Networks (SENets) introduce a building block for CNNs that improves channel interdependencies at almost no computational cost. ***Besides this huge performance boost, they can be easily added to existing architectures.*** The main idea is this: ***Let’s add parameters to each channel of a convolutional block so that the network can adaptively adjust the weighting of each feature map.***
- CNNs use their convolutional filters to extract hierarchal information from images. *Lower layers find trivial pieces of context like edges or high frequencies, while upper layers can detect faces, text or other complex geometrical shapes.* They extract whatever is necessary to solve a task efficiently.
- ***All you need to understand for now is that the network weights each of its channels equally when creating the output feature maps. SENets are all about changing this by adding a content aware mechanism to weight each channel adaptively. In it’s most basic form this could mean adding a single parameter to each channel and giving it a linear scalar how relevant each one is.***
- *First, they get a global understanding of each channel by squeezing the feature maps to a single numeric value. This results in a vector of size n, where n is equal to the number of convolutional channels. Afterwards, it is fed through a two-layer neural network, which outputs a vector of the same size. These n values can now be used as weights on the original features maps, scaling each channel based on its importance.*
- ![SENet Architecture](https://miro.medium.com/max/658/1*WNk-atKDUsZPvMddvYL01g.png)
- Implementation
	```python
	def se_block(x, c, r=16):
		z = GlobalAveragePooling2D()(x)
        z = Dense(units=c//r, activation="relu")(z)
        z = Dense(units=c, activation="sigmoid")(z)
        return z*x
	```
# EfficientNet
- Source: https://keras.io/examples/vision/image_classification_efficientnet_fine_tuning/
- The input data should range [0, 255]. Normalization is included as part of the model.
- When the images are much smaller than the size of EfficientNet input, we can simply upsample the input images. It has been shown in Tan and Le, 2019 that transfer learning result is better for increased resolution even if input images remain small. 
- The first step to transfer learning is to freeze all layers and train only the top layers. For this step, a relatively large learning rate (1e-2) can be used. Note that validation accuracy and loss will usually be better than training accuracy and loss. This is because the regularization is strong, which only suppresses training-time metrics. 
- The second step is to unfreeze a number of layers and fit the model using smaller learning rate. In this example we show unfreezing all layers, but depending on specific dataset it may be desireble to only unfreeze a fraction of all layers.
- When the feature extraction with pretrained model works good enough, this step would give a very limited gain on validation accuracy. In our case we only see a small improvement, as ImageNet pretraining already exposed the model to a good amount of dogs.
- On the other hand, when we use pretrained weights on a dataset that is more different from ImageNet, this fine-tuning step can be crucial as the feature extractor also needs to be adjusted by a considerable amount. Such a situation can be demonstrated if choosing CIFAR-100 dataset instead, where fine-tuning boosts validation accuracy by about 10% to pass 80% on EfficientNetB0. In such a case the convergence may take more than 50 epochs.
- A side note on freezing/unfreezing models: setting trainable of a Model will simultaneously set all layers belonging to the Model to the same trainable attribute. Each layer is trainable only if both the layer itself and the model containing it are trainable. Hence when we need to partially freeze/unfreeze a model, we need to make sure the trainable attribute of the model is set to True.
- Tips for fine tuning EfficientNet on unfreezing layers: 
	- The BathcNormalization layers need to be kept frozen (more details). If they are also turned to trainable, the first epoch after unfreezing will significantly reduce accuracy.
	- In some cases it may be beneficial to open up only a portion of layers instead of unfreezing all. This will make fine tuning much faster when going to larger models like B7.
	- Each block needs to be all turned on or off. This is because the architecture includes a shortcut from the first layer to the last layer for each block. Not respecting blocks also significantly harms the final performance.
- Smaller batch size benefit validation accuracy, possibly due to effectively providing regularization.

# Semi-Supervised Learning in Computer Vision
- Source: https://amitness.com/2020/07/semi-supervised-learning/
## Pseudo-label
- Dong-Hyun Lee proposed a very simple and efficient formulation called "Pseudo-label" in 2013.
- The idea is to train a model simultaneously on a batch of both labeled and unlabeled images. The model is trained on labeled images in usual supervised manner with a cross-entropy loss. *The same model is used to get predictions for a batch of unlabeled images and the maximum confidence class is used as the pseudo-label. Then, cross-entropy loss is calculated by comparing model predictions and the pseudo-label for the unlabeled images.*
- ![Pseudo-label](https://amitness.com/images/ssl-pseudo-label.png)
- The total loss is a weighted sum of the labeled and unlabeled loss terms.
- To make sure the model has learned enough from the labeled data, the \alpha_{t} term is set to 0 during the initial 100 training steps. It is then gradually increased up to 600 training steps and then kept constant.
## Consistency Regularization
- *This paradigm uses the idea that model predictions on an unlabeled image should remain the same even after adding noise.* We could use input noise such as Image Augmentation and Gaussian noise. Noise can also be incorporated in the architecture itself using Dropout.
- ![Consistency Regularization](https://amitness.com/images/fixmatch-unlabeled-augment-concept.png)

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

# IoU (Intersection over Union)
- To score how well the predicted box matches the ground-truth we can compute the IOU (or intersection-over-union, also known as the Jaccard index) between the two bounding boxes.
- Ideally, the predicted box and the ground-truth have an IOU of 100% but in practice anything over 50% is usually considered to be a correct prediction
- Normalizes the bounding box width and height by the image width and height so that they fall between 0 and 1.
- parametrizes the bounding box x and y coordinates to be offset of a particular grid cell location so they are also bounded between 0 and 1.
- Implementation
	```python
	def giou(bbox1, bbox2):
		# (x1, y1, x2, y2)
		bbox1 = np.array(bbox1)
		bbox2 = np.array(bbox2)

		area_bbox1 = (bbox1[2] - bbox1[0])*(bbox1[3] - bbox1[1])
		area_bbox2 = (bbox2[2] - bbox2[0])*(bbox2[3] - bbox2[1])

		pt1_intsec = np.maximum(bbox1[:2], bbox2[:2])
		pt2_intsec = np.minimum(bbox1[2:], bbox2[2:])
		w_intsec, h_intsec = np.maximum(pt2_intsec - pt1_intsec, 0)
		area_intsec = w_intsec*h_intsec

		area_union = area_bbox1 + area_bbox2 - area_intsec

		iou = np.maximum(area_intsec/area_union, np.finfo(np.float32).eps)

		pt1_enclosed = np.minimum(bbox1[:2], bbox2[:2])
		pt2_enclosed = np.maximum(bbox1[2:], bbox2[2:])
		w_enclosed, h_enclosed = np.maximum(pt2_enclosed - pt1_enclosed, 0)
		area_enclosed = w_enclosed*h_enclosed

		return iou - (area_enclosed - area_union)/area_enclosed
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

# `tf.keras.utils.image_dataset_from_directory()`
- References: https://www.tensorflow.org/api_docs/python/tf/keras/utils/image_dataset_from_directory, https://www.tensorflow.org/tutorials/load_data/images
```python
ds_tr = image_dataset_from_directory(
    "computer_vision_task/train",
    validation_split=0.2,
    subset="training",
    seed=seed,
    shuffle=True,
    image_size=(img_size, img_size),
    batch_size=64)
ds_val = image_dataset_from_directory(
    "computer_vision_task/train",
    validation_split=0.2,
    subset="validation",
    seed=seed,
    shuffle=True,
    image_size=(img_size, img_size),
    batch_size=64)
ds_te = image_dataset_from_directory(
    "computer_vision_task/test",
    seed=seed,
    shuffle=False,
    image_size=(img_size, img_size),
    batch_size=64)
```

# `cv2`
## Install
- On Windows
	```python
	conda install -c conda-forge opencv
	```
- On MacOS
	```python
	pip install opencv-python
	```
## Import
```python
import cv2
```
## `cv2.imread()`
## `plt.imshow()`
## `cv2.cvtColor(image, code)`
- `code`: (`cv2.COLOR_BGR2GRAY`, `cv2.COLOR_BGR2RGB`, `cv2.COLOR_BGR2HSV`)
## `cv2.resize(img, dsize, interpolation)`
## `cv2.rectangle(img, pt1, pt2, color, thickness)`
## `cv2.circle(img, center, radius, color, [thickness], [lineType], [shift])`
## `cv2.getTextSize(text, fontFace, fontScale, thickness)`
```python
(text_width, text_height), baseline = cv2.getTextSize(text=label, fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=font_scale, thickness=bbox_thick)
```
## `cv2.putText(img, text, org, fontFace, fontScale, color, [thickness], [lineType])`
- Reference: https://docs.opencv.org/4.x/d6/d6e/group__imgproc__draw.html#ga0f9314ea6e35f99bb23f29567fc16e11
- `text`: Text string to be drawn.
- `org`: Bottom-left corner of the text string in the image.
- `fonFace`: (`cv2.FONT_HERSHEY_SIMPLEX`, ...) Font type.
- `fontScale`: Font scale factor that is multiplied by the font-specific base size.

# Contour Detection
```python
img = cv2.imread(...)

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

lower_blue = np.array([100, 50, 150]) #파란색 계열의 범위 설정
upper_blue = np.array([130, 255, 255])

mask = cv2.inRange(hsv, lower_blue, upper_blue)
contours, hierachy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#contour를 감싸는 사각형들의 x, y, w, h를 'rects'에 저장
rects = [cv2.boundingRect(contour) for contour in contours]

rects_selected = []
for rect in rects:
    if rect[0] > 1200 and 100 < rect[1] < 200:
        rects_selected.append(rect)
rects_selected.sort(key=lambda x:x[0])

for i, rect in enumerate(rects_selected):
    cv2.rectangle(img, (rect[0], rect[1]), (rect[0]+rect[2], rect[1]+rect[3]), (0, 0, 255), 2)
    cv2.putText(img, str(i+1), (rect[0]-5, rect[1]-5), fontFace=0, fontScale=0.6, color=(0, 0, 255), thickness=2)
    cv2.circle(img, (rect[0]+1, rect[1]-12), 12, (0, 0, 255), 2)

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
```

# `Image`
```python
from PIL import Image
```
## Open Image
```python
img = Image.open(fp)
```
### Manipulate Image
```python
img.size()
img.save()
img.thumbnail()
img.crop()
img.resize()

# (`"RGB"`, `"RGBA"`, `"CMYK"`, `"L"`, `"1"`)
img.convert("L")
```
```
### `img.paste()`
```python
img1.paste(img2, (20,20,220,220))
```
- img2.size와 동일하게 두 번째 parameter 설정.	
## `Image.new()`
```python
mask = Image.new("RGB", icon.size, (255, 255, 255))
```

# Data Augmentation
## Using `Sequential()`
- Reference: https://keras.io/examples/vision/image_classification_from_scratch/
- Make it part of the model
- ***With this option, your data augmentation will happen on device, synchronously with the rest of the model execution, meaning that it will benefit from GPU acceleration.***
- ***Data augmentation is inactive at test time, so the input samples will only be augmented during `fit()`, not when calling `evaluate()` or `predict()`.***
```python
inputs = Input(shape=(img_size, img_size, 3))

data_aug = Sequential([
	RandomRotation(factor=0.1, fill_mode="constant", fill_value=255), 
	RandomTranslation(height_factor=0.1, width_factor=0.1, fill_mode="constant", fill_value=255),  
	RandomFlip("horizontal"), 
	RandomZoom(height_factor=0.1, fill_mode="constant", fill_value=255)]
	)
z = data_aug(inputs)
```
## Using `tf.data.Dataset.map()`
- Apply it to the dataset, so as to obtain a dataset that yields batches of augmented images.
- With this option, your data augmentation will happen on CPU, asynchronously, and will be buffered before going into the model.
- If you're training on CPU, this is the better option, since it makes data augmentation asynchronous and non-blocking.
## Using `ImageDataGenerator()`
- Reference: https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator, https://m.blog.naver.com/PostView.nhn?blogId=isu112600&logNo=221582003889&proxyReferer=https:%2F%2Fwww.google.com%2F
- Each new batch of our data is randomly adjusting according to the parameters supplied to `ImageDataGenerator()`.
```python
# `shear_range`: (float). Shear Intensity (Shear angle in counter-clockwise direction as radians)
# `zoom_range`
	# (`[lower, upper]`) Range for random zoom.
	# (float) Range of `[1 - zoom_range, 1 + zoom_range]`
# `rotation_range`
# `brightness_range`: (Tuple or List of two floats) Range for picking a brightness shift value from.
# `rescale`: rescaling factor. Defaults to None. If None or 0, no rescaling is applied, otherwise we multiply the data by the value provided (before applying any other transformation).
# `horizontal_flip`, `vertical_flip`: (bool). Randomly flip inputs horizontally.
# `width_shift_range`, `height_shift_range`
	# (float) Fraction of total width (or height).
	# (Tuple or List) Random elements from the array.
	# (int) Pixels from interval (-`width_shift_range`, `width_shift_range`) (or (-`height_shift_range`, `height_shift_range`))
gen = ImageDataGenerator([shear_range], [zoom_range], [ratation_range], [brightness_range], [rescale], [horizontal_flip], [vertical_flip], [width_shift_range], [height_shift_range], [validation_split])
# Fits the data generator to some sample data.
# This computes the internal data stats related to the data-dependent transformations, based on an array of sample data.
# Only required if `featurewise_center` or `featurewise_std_normalization` or `zca_whitening` are set to True.
# When `rescale` is set to a value, rescaling is applied to sample data before computing the internal data stats.
# `x`: Sample data. Should have rank 4
gen.fit(x, [seed])
```
```python
gen.apply_transform()
```
- Using `ImageDataGenerator().flow()` (for GoogLeNet, for example)
	- References: https://www.tensorflow.org/api_docs/python/tf/keras/Model, https://gist.github.com/swghosh/f728fbba5a26af93a5f58a6db979e33e
	```python
	def gen_flow(x, y, subset):
		gen = ImageDataGenerator(..., validation_split=0.2)
		gen.fit(X_tr_val)
		# `subset`: (`"training"`, `"validation"`) If `validation_split` is set in `ImageDataGenerator()`.
		# `save_to_dir`: This allows you to optionally specify a directory to which to save the augmented pictures being generated.
		return gen.flow(x=x, y=y, batch_size=batch_size, subset=subset)
	def generator(flow):
		for xi, yi in flow:
			yield xi, [yi, yi, yi]
			
	flow_tr = gen_flow(X_tr_val, y_tr_val, subset="training")
	flow_val = gen_flow(X_tr_val, y_tr_val, subset="validation")

	# `steps_per_epoch`: We can calculate the value of `steps_per_epoch` as the total number of samples in your dataset divided by the batch size.
	# `validation_steps`: Only if the `validation_data` is a generator then only this argument can be used. It specifies the total number of steps taken from the generator before it is stopped at every epoch and its value is calculated as the total number of validation data points in your dataset divided by the validation batch size.
	hist = model.fit(x=generator(flow_tr), validation_data=generator(flow_val), epochs, steps_per_epoch=len(flow_tr), validation_steps=len(flow_val), callbacks=[es, mc])
	```
- Using `ImageDataGenerator().flow_from_directory()`
	```python
	# `target_size`: the dimensions to which all images found will be resized.
	# `class_mode`: (`"binary"`, `"categorical"`, `"sparse"`, `"input"`, `None`)
		# `"binary"`: for binary classification.
		# `"categorical"`: for multi-class classification(OHE).
		# `"sparse"`: for multi-class classification(no OHE).
		# `"input"`
		# `None`: Returns no label.
	```