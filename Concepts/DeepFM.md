# DeepFM
- source : https://greeksharifa.github.io/machine_learning/2020/04/07/DeepFM/
- 모델에 대해 설명할 것이다. 이 DeepFM이라는 모델은 FM과 딥러닝을 결합한 것이다. 최근(2017년 기준) 구글에서 발표한 Wide & Deep model에 비해 피쳐 엔지니어링이 필요 없고, wide하고 deep한 부분에서 공통된 Input을 가진다는 점이 특징적이다.
- DeepFM은 피쳐 엔지니어링 없이 end-to-end 학습을 진행할 수 있다. 저차원의 interaction들은 FM 구조를 통해 모델화하고, 고차원의 interaction들은 DNN을 통해 모델화한다.
- DeepFM은 같은 Input과 Embedding 벡터를 공유하기 때문에 효과적으로 학습을 진행할 수 있다.
