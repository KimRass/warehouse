# Feature Map (= Activation Map)
- Source: https://ujjwalkarn.me/2016/08/11/intuitive-explanation-convnets/
- In CNN terminology, the 3×3 matrix is called a "filter" or "kernel" or "feature detector" and the matrix formed by sliding the filter over the image and computing the dot product is called the "Activation Map" or the 
Feature Map" .It is important to note that filters acts as feature detectors from the original input image.
- ***It is evident from the animation above that different values of the filter matrix will produce different feature maps for the same input image.
- In the table below, we can see the effects of convolution of the above image with different filters. As shown, we can perform operations such as edge detection, sharpen and blur just by changing the numeric values of our filter matrix before the convolution operation – this means that different filters can detect different features from an image, for example edges, curves etc.
- It is important to note that the Convolution operation captures the local dependencies in the original image.

# Semi-Supervised Learning in Computer Vision
- Sources: https://amitness.com/2020/07/semi-supervised-learning/, https://animilux.github.io/paper_review/2021/10/10/scan.html
## Pseudo-label
- Dong-Hyun Lee proposed a very simple and efficient formulation called "Pseudo-label" in 2013.
- The idea is to train a model simultaneously on a batch of both labeled and unlabeled images. The model is trained on labeled images in usual supervised manner with a cross-entropy loss. *The same model is used to get predictions for a batch of unlabeled images and the maximum confidence class is used as the pseudo-label. Then, cross-entropy loss is calculated by comparing model predictions and the pseudo-label for the unlabeled images.*
- ![Pseudo-label](https://amitness.com/images/ssl-pseudo-label.png)
- The total loss is a weighted sum of the labeled and unlabeled loss terms.
- To make sure the model has learned enough from the labeled data, the \alpha_{t} term is set to 0 during the initial 100 training steps. It is then gradually increased up to 600 training steps and then kept constant.
- Source: https://deep-learning-study.tistory.com/553
- 우리가 제안하는 신경망은 labeled data와 unlabeled data로 지도 학습 방식으로 학습 됩니다. Unlabeled data(Pseudo-Labels)는 매 가중치 업데이트마다 확률이 가장 높은 클래스를 true label로 사용합니다.
- Pseudo Label은 unlabeled data의 target class 입니다. 각 unlabeled sample에 대해 가장 높은 예측된 확률을 true class로 선택합니다.
- Pseudo Label은 fine-tunning 상황에서 drop out과 함께 사용합니다. pre-trainined network는 지도 학습 방식으로 labeled data와 unlabeled data 동시에 학습됩니다. Pseudo-Labels는 매 가중치 업데이트마다 새롭게 계산됩니다. 그리고 지도 학습의 손실 함수에 사용됩니다.
- labeled data와 unlabeled data의 총 개수는 다릅니다. 그리고 이들 사이의 균형 있는 학습은 상당히 중요합니다. 따라서 손실 함수에 상수를 추가하여 둘 사이의 균형을 조절합니다.
- Low-Density Separation between Classes: 준지도 학습의 목표는 unlabeled data를 사용하여 성능을 향상시키는 것입니다. 성능을 향상시키기 위해서는 decision boundary가 low-density regions에 있어야 합니다.
- Source: https://blog.francium.tech/semi-supervised-learning-with-pseudo-labeling-de65988bb3b3
- We have semi-supervised learning(SSL) methods to counter the unlabeled data. It is an approach that combines a small amount of labeled data and a large amount of unlabeled data during training. It improves the model's robustness as well.
- ***Instead of pseudo labeling all the data at once we can iterate over the unlabeled data and for every new 500 pseudo-labeled data we can retrain the model, and repeat this process until we no longer have any unlabeled data.***
- Few things we should have in our mind before starting with SSL is, The size of the unlabeled data should be substantially higher than the size of the labeled data. otherwise, we could simply solve the problem with supervised learning methods itself.
## Consistency Regularization
- *This paradigm uses the idea that model predictions on an unlabeled image should remain the same even after adding noise.* We could use input noise such as Image Augmentation and Gaussian noise. Noise can also be incorporated in the architecture itself using Dropout.
- ![Consistency Regularization](https://amitness.com/images/fixmatch-unlabeled-augment-concept.png)
