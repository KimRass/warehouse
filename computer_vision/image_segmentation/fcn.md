- Paper: [Fully Convolutional Networks for Semantic Segmentation](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Long_Fully_Convolutional_Networks_2015_CVPR_paper.pdf)

# 기존 방식의 문제점
- Source: https://m.blog.naver.com/laonple/220958109081
- FCN이 주목한 부분은 classification에서 성능을 검증 받은 좋은 네트워크 (AlexNet, VGGNet, GoogLeNet) 등을 이용하는 것이다. 이들 대부분의 classification을 위한 네크워크는 뒷단에 분류를 위한 fully connected layer가 오는데, 이 fully connected layer가 고정된 크기의 입력만을 받아들이는 문제가 있다.
- 또 한가지 결정적인 문제는 fully connected layer를 거치고 나면, 위치 정보가 사라지는 것이다. Segmentation에 사용하려면 위치정보를 알 수 있어야 하는데 불가능하기 때문에 심각한 문제가 된다.

# Convolutionalization
- ![convolutionalization_2](https://miro.medium.com/max/1400/1*2IuHjzPjjGXtDU-eAUzxeA.webp)
  - 7x7x512의 Feature map을 4,096차원의 FC layer로 변환하기 위한 가중치의 수: (7 * 7 * 512) * 4096
  - 7x7x512의 Feature map을 1x1x4096의 Feature map으로 변환하기 위한 Convolution layer의 가중치의 수: (7x7x512) * 4096
  - 즉 동일한 가중치의 수를 갖습니다.

# Deconvolution
- ![deconvolution](https://miro.medium.com/max/1400/1*gtBk1yTapyFzvh00DUkGBA.webp)

# Skip Connection
- ![fcn16s](https://miro.medium.com/max/1400/1*-1hOIxlnFn3qd7n5JEgzFg.webp)
- ![fcn8s](https://miro.medium.com/max/1400/1*1r-KVNqt9V7JiDT-zyOEAQ.webp)