* Dense(8, input_dim=4, init='uniform', activation='relu'))
주요 인자는 다음과 같습니다.

    - 첫번째 인자 : 출력 뉴런의 수를 설정합니다.
    - input_dim : 입력 뉴런의 수를 설정합니다.
    - init : 가중치 초기화 방법 설정합니다.
    - ‘uniform’ : 균일 분포
    - ‘normal’ : 가우시안 분포
    - activation : 활성화 함수 설정합니다.
        - ‘linear’ : 디폴트 값, 입력뉴런과 가중치로 계산된 결과값이 그대로 출력으로 나옵니다.
        * ‘relu(Rectified Linear Unit)’ : rectifier 함수, 은닉층에 주로 쓰입니다. <span style="color:#666666">최근 신경망 모델들은 대부분 activation function으로 ReLU를 사용한다. 그 이유는 vanishing gradient 현상을 해결하기 때문인데</span>    
        * ‘sigmoid’ : 시그모이드 함수, 이진 분류 문제에서 출력층에 주로 쓰입니다. <span style="color:#666666">step function을 사용하는 perceptron은 0과 1만 출력했다면, sigmoid를 사용하는 신경망 모델은 역속적인 값을 전달한다. </span>시그모이드 함수가 계단 함수에 비해 더 많은 정보를 전달 할 수 있다.
        * ‘softmax’ : 소프트맥스 함수, 다중 클래스 분류 문제에서 출력층에 주로 쓰입니다.
* <span style="color:#666666">활성화함수는 꼭 비선형 함수이어야 한다. 선형 함수를 사용하면 신경망의 층을 깊게 쌓는 것에 의미가 없어지기 때문이다.  그 이유는 예를 들어, 활성화 함수를 h(x) = cx 라는 선형함수라 해보자. 3층으로 구성된 네트워크라 할 때, y(x) = h(h((x))) = c\*c\*c\*x = c^3\*x이다. 이는 곧 y = ax에서 a=c^3과 같다. 즉, 기껏 3층이나 쌓았지만 1층만 쌓은 네트워크와 같아진다. 이것이 바로 활성함수의 역할이다.(</span>[https://leedakyeong.tistory.com/entry/%EB%B0%91%EB%B0%94%EB%8B%A5%EB%B6%80%ED%84%B0-%EC%8B%9C%EC%9E%91%ED%95%98%EB%8A%94-%EB%94%A5%EB%9F%AC%EB%8B%9D-%ED%99%9C%EC%84%B1%ED%99%94%ED%95%A8%EC%88%98%EB%9E%80-What-is-activation-function?category=845638](https://leedakyeong.tistory.com/entry/%EB%B0%91%EB%B0%94%EB%8B%A5%EB%B6%80%ED%84%B0-%EC%8B%9C%EC%9E%91%ED%95%98%EB%8A%94-%EB%94%A5%EB%9F%AC%EB%8B%9D-%ED%99%9C%EC%84%B1%ED%99%94%ED%95%A8%EC%88%98%EB%9E%80-What-is-activation-function?category=845638))
* [https://buttercoconut.xyz/132/]
* 2개를 분류하는 문제일 때는 Vanishing Gradient Problem때문에 sigmoid는 잘 사용하지 않고 ReLU와 그 변형된 활성화함수를 주로 이용한다. 3개 이상을 분류할 때 주로 Softmax와 그 변형된 활성화함수를 주로 이용한다.
[http://blog.naver.com/PostView.nhn?blogId=wideeyed&logNo=221017173808](http://blog.naver.com/PostView.nhn?blogId=wideeyed&logNo=221017173808)
* Softmax : [http://blog.naver.com/PostView.nhn?blogId=wideeyed&logNo=221021710286&parentCategoryNo=&categoryNo=&viewDate=&isShowPopularPosts=false&from=postView](http://blog.naver.com/PostView.nhn?blogId=wideeyed&logNo=221021710286&parentCategoryNo=&categoryNo=&viewDate=&isShowPopularPosts=false&from=postView)
* [https://leedakyeong.tistory.com/entry/밑바닥부터-시작하는-딥러닝-활성화함수란-What-is-activation-function?category=845638](https://leedakyeong.tistory.com/entry/%EB%B0%91%EB%B0%94%EB%8B%A5%EB%B6%80%ED%84%B0-%EC%8B%9C%EC%9E%91%ED%95%98%EB%8A%94-%EB%94%A5%EB%9F%AC%EB%8B%9D-%ED%99%9C%EC%84%B1%ED%99%94%ED%95%A8%EC%88%98%EB%9E%80-What-is-activation-function?category=845638)

### Softmax

* <span style="color:#666666">일반적으로 마지막에는 활성화함수를 적용하지 않으며, 일반 회귀(regression)에서는 identity\_function()을, 분류(classification)문제에서는 softmax를 사용</span>
    * <span style="color:#666666">softmax란? 신경망의 출력층에서 사용하는 활성화 함수로, 분류문제에 쓰이는 함수이다. 회귀에서는 항등함수(identity function)을 사용한다.</span>
    * [https://leedakyeong.tistory.com/entry/%EB%B0%91%EB%B0%94%EB%8B%A5%EB%B6%80%ED%84%B0-%EC%8B%9C%EC%9E%91%ED%95%98%EB%8A%94-%EB%94%A5%EB%9F%AC%EB%8B%9D-%EC%86%8C%ED%94%84%ED%8A%B8%EB%A7%A5%EC%8A%A4-%ED%95%A8%EC%88%98-%EA%B5%AC%ED%98%84%ED%95%98%EA%B8%B0-in-%ED%8C%8C%EC%9D%B4%EC%8D%AC-softmax-in-python?category=845638](https://leedakyeong.tistory.com/entry/%EB%B0%91%EB%B0%94%EB%8B%A5%EB%B6%80%ED%84%B0-%EC%8B%9C%EC%9E%91%ED%95%98%EB%8A%94-%EB%94%A5%EB%9F%AC%EB%8B%9D-%EC%86%8C%ED%94%84%ED%8A%B8%EB%A7%A5%EC%8A%A4-%ED%95%A8%EC%88%98-%EA%B5%AC%ED%98%84%ED%95%98%EA%B8%B0-in-%ED%8C%8C%EC%9D%B4%EC%8D%AC-softmax-in-python?category=845638)

# Categorical Cross-Entropy Loss
- 출처 : https://wordbe.tistory.com/entry/ML-Cross-entropyCategorical-Binary의-이해
- Softmax activation 뒤에 Cross-Entropy loss를 붙인 형태로 주로 사용하기 때문에 Softmax loss 라고도 불립니다.
→ Multi-class classification에 사용됩니다.
우리가 분류문제에서 주로 사용하는 활성화함수와 로스입니다. 분류 문제에서는 MSE(mean square error) loss 보다 CE loss가 더 빨리 수렴한 다는 사실이 알려져있습니다. 따라서 multi class에서 하나의 클래스를 구분할 때 softmax와 CE loss의 조합을 많이 사용합니다.
- <img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https://blog.kakaocdn.net/dn/Nq9TL/btqxdqsIG99/c9IiTiHBDp4cgCUtYXPSzk/img.png">
