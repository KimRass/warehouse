# Tasks
## Image Classification (= Object Recognition)
- The evolution of image classification
  - Source: https://stanford.edu/~shervine/blog/evolution-image-classification-explained
  - LeNet -> AlexNet -> VGGNet -> GoogLeNet -> ResNet -> DenseNet
## Object Localization
- Source: https://www.einfochips.com/blog/understanding-object-localization-with-deep-learning/#:~:text=Image%20localization%20is%20a%20spin,around%20an%20object%20of%20interest.
- Object localization is a regression problem *where the output is x and y coordinates around the object of interest to draw bounding boxes.*
## Object Detection
- *Object detection is a complex problem that combines the concepts of image localization and classification. Given an image, an object detection algorithm would return bounding boxes around all objects of interest and assign a class to them.*
## Image Inpainting

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

# Latent Space

# CAM & Grad-CAM
- References: https://tyami.github.io/deep%20learning/CNN-visualization-Grad-CAM/, https://jsideas.net/grad_cam/
## Class Activation Map (CAM)
- ![CAM](https://tyami.github.io/assets/images/post/DL/2020-10-27-CNN-visualization-Grad-CAM/2020-10-27-cnn-visualization-grad-cam-3-cam-gap.png)
- Image classification을 위한 Model에서는 Convolutional layers 등이 연속한 후 마지막에는 FC layers가 오게 됩니다.
- 이때 기존의 Architecture에서 FC layers를 없애고 대신 Global average pooling (GAP) layer를 둡니다.
- 그리고 이 위에 각 Class를 의미하는 FC layer를 붙이고 Fine-tunning합니다.
- 마지막 두 Layers간의 Weight를 각각의 마지막 Feature map과 곱하고 (Feature map에 대한 가중치 역할) 더해서 (Pixel-wise summation) Feature maps와 동일한 차원을 갖는 CAM을 만듭니다.
- ![CAM2](https://tyami.github.io/assets/images/post/DL/2020-10-27-CNN-visualization-Grad-CAM/2020-10-27-cnn-visualization-grad-cam-9-cam-1.png)
- 마지막 Convolutional layer를 사용하는 이유?: 이미지에 대한 가장 고차원적인 Feature를 추출한 layer이기 때문입니다.
- 한계
  - 원래 Model의 Architecture에서 GAP가 사용되지 않는다면 FC layers를 GAP로 대체한 후 Fine-tunning을 해야 합니다.
  - 마지막 Convolutional layer에 대해서만 CAM을 생성하는 것이 의미가 있다고 합니다.
## Grad-CAM (Gradient CAM)
- 수식적으로 CAM과 동일하여 Generalized CAM이라고 할 수 있다고 합니다.
- CAM애서 GAP와 각 Class를 나타내는 FC layer간의 Weights 대신에 Gradient를 사용하는 것이 차이점입니다.
- Architecture
  - ![Grad-CAM](https://tyami.github.io/assets/images/post/DL/2020-10-27-CNN-visualization-Grad-CAM/2020-10-27-cnn-visualization-grad-cam-19-grad-cam-1.png)
  - 마지막 Convolutional layer를 Flatten한 뒤 각 Class를 나타내는 FC layer와 연결합니다.
  - Feature map의 각 픽셀별로 Gradient를 구하고 Feature map별로 평균을 구합니다 (`a_k`). 이를 CAM의 Weights와 동일한 개념으로 사용합니다.
  - 각 Weight와 Feature map을 곱한 값을 더하고 최종적으로 ReLU 함수를 통과시키면 Grad-CAM이 생성됩니다.
  - `S_c`: Softmax layer의 Input value.
  - `Z`: Feature map의 Width * Height
- CAM과의 차이점
  - GAP를 사용하지 않아도 되므로 Fine-tunning이 필요 없습니다.
  - 마지막 Convolutional layer에만 국한되지 않습니다.
```python
def generate_grad_cam(img_tensor, model, class_index, activation_layer):
    """
    params:
    -------
    img_tensor: resnet50 모델의 이미지 전처리를 통한 image tensor
    model: pretrained resnet50 모델 (include_top=True)
    class_index: 이미지넷 정답 레이블
    activation_layer: 시각화하려는 레이어 이름

    return:
    grad_cam: grad_cam 히트맵
    """
    inp = model.input
    y_c = model.output.op.inputs[0][0, class_index]
    A_k = model.get_layer(activation_layer).output
    
    ## 이미지 텐서를 입력해서
    ## 해당 액티베이션 레이어의 아웃풋(a_k)과
    ## 소프트맥스 함수 인풋의 a_k에 대한 gradient를 구한다.
    get_output = K.function([inp], [A_k, K.gradients(y_c, A_k)[0], model.output])
    [conv_output, grad_val, model_output] = get_output([img_tensor])

    ## 배치 사이즈가 1이므로 배치 차원을 없앤다.
    conv_output = conv_output[0]
    grad_val = grad_val[0]
    
    ## 구한 gradient를 픽셀 가로세로로 평균내서 a^c_k를 구한다.
    weights = np.mean(grad_val, axis=(0, 1))
    
    ## 추출한 conv_output에 weight를 곱하고 합하여 grad_cam을 얻는다.
    grad_cam = np.zeros(dtype=np.float32, shape=conv_output.shape[0:2])
    for k, w in enumerate(weights):
        grad_cam += w * conv_output[:, :, k]
    
    grad_cam = cv2.resize(grad_cam, (224, 224))

    ## ReLU를 씌워 음수를 0으로 만든다.
    grad_cam = np.maximum(grad_cam, 0)

    grad_cam = grad_cam / grad_cam.max()
    return grad_cam
```
## Guided Grad-CAM