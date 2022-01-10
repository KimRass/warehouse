# Types of Data
## Explicit Data
## Implicit Data

# Collaborative Filtering (CF)
# Association Analysis
# BPR (Bayesian Personalizaed Ranking)
# Matrix Factorization
# Factorization Machine
# DeepFM
# A/B Test
# Multi-Armed Bandit

# Evaluation Metrics
## MAP(Mean Average Precision)
- 하지만 사용자에 따라서 실제로 소비한 아이템의 수가 천 개, 2천개까지 늘어날 수 있습니다. 이 경우 recall이 1이 되는 지점까지 고려하는 것은 무리이므로 최대 n개까지만 고려하여 mAP를 계산하며, 이 경우 mAP@n 으로 표기합니다.
## nDCG(normalized Discounted Cumulative Gain)

# 1. 신규 고객(고객에 대한 정보 없음) : 
## 1-1. 비개인화 추천(Non-personalized Recommendation)
### (1) 신규 아이템 추천
- ex. 2020년에 개봉한 영화 추천
### (2) 평이 좋은 아이템 추천
- explicit data가 있을 때만 사용 가능한 방법
    - 점수, 후기 등 사용자가 자신의 선호를 직접적으로 데이터
- ex. 평점의 평균이 높은 영화 추천(이때 평점의 갯수가 너무 적은 것은 통계적 신뢰성이 낮으므로 제외해야 함)
### (3) 인기 아이템 추천
- implicit data에도 사용 가능한 방법
    - 장바구니 기록, 시청 횟수 등 사용자가 자신의 선호를 직접적으로 드러내지 않은 데이터
    - 따라서 고객은 자신이 선호하는 아이템에 대해서 주로 로그를 발생시킨다고 가정하여 접근(ex. 사람들은 좋아하는 음악을 많이 듣지, 좋아하지 않는 음악은 거의 듣지 않음. 따라서 주로 긍정적 반응이 기록됨)
    - 사용자의 행동에 따라 자동으로 기록되므로 explicit data보다 밀하다(dense)(평점은 고객이 작성해야 데이터화되지만 로그는 자동으로 기록됨)
    - 그러나 로그 횟수를 별점으로 해석하면 안 됨. 즉 10번 들은 곡보다 1000번 들은 곡을 100배 좋아하는 것은 아님. 또한 로그는 오래된 곡에 대한 것일수록 많을 수밖에 없음.
- ex. 쿠팡에서 고객들의 클릭 횟수가 가장 많은 제품 추천
### 1-2. 연관 분석(Association Analysis)
- 대용량의 거래 데이터로부터 "X를 구매했으면, Y를 구매할 것이다" 형식의 아이템 간 연관 관계를 분석하는 방법
- 보통 장바구니 분석(Market Basket Analysis)로 불리기도 합니다. 즉, 고객의 장바구니에 어떤 아이템이 동시 담겼는지 패턴을 파악하여 상품을 추천하는 방법입니다.
- 전체 사용자들 중 아이템 y를 좋아하는 비율(support)보다 아이템 x를 좋아하는 사용자들 중 아이템 y도 좋아하는 비율(confidence)가 매우 크다면 아이템 x와 y는 매우 연관 관계가 높을 것입니다.(lift = confidence/support가 클수록 연관 관계가 높음. lift가 얼마 이상이어야 연관 관계가 높은지에 대한 판단은 정해진 것이 없음)
- 반면에 전체 사용자들 중 아이템 y를 좋아하는 비율(support)보다 아이템 x를 좋아하는 사용자들 중 아이템 y도 좋아하는 비율(confidence)이 매우 작다면 아이템 x와 y는 매우 연관 관계가 낮을 것입니다.
- "lift"의 의미는 어떤 사용자가 아이템 x를 좋아한다는 사실이 아이템 y를 좋아할 가능성을 높인다는 의미입니다.
- (평점을 4점 이상 준 것을 해당 영화를 좋아한다는 표시로 볼 때)

# 2. 기존 고객 재방문(고객에 대한 약간의 정보 있음)
## 2-1. Content-based Filtering
- 고객 또는 아이템의 세부 정보를 바탕으로 추천하는 것(ex. 영화 "One Day"에 높은 평점을 준 고객에게 동일한 장르의 영화를 추천)
- 그러나 추천한 영화가 "One Day"와 장르는 동일하지만 세부적인 내용이 완전히 달라 선호하지 않을 수 있음. 또한 고객이 좋아하는 영화 중 "One Day"와 장르가 다른 영화는 추천할 수 없음
-  가 동일하지만 세부적인 내용은 상당히 다를 수 있습니다. 또한 장르는 다르지만 해당 사용자가 좋아할 만한 영화도 있을 수 있습니다.
- 이렇게 장르와 같이 한정적인 정보만으로는 정교한 추천이 어려우며 영화별로 출연 배우나 감독 등 더 세부적인 정보를 수집하는 것 또한 많은 시간과 노력을 필요로 함
## 2-2. Collaborative Filtering(CF)
- 고객과 아이템 간 상호작용 데이터 활용
- User-Item Matrix 만들기
고객 79,044명, 영화 3,413개
### (1) Memory-based
- 별도의 특성 추출 과정 없음
#### User-based CF
-  어떤 사용자와 유사한 취향을 가진 사용자가 좋아하는 아이템을 추천
#### Item-based CF
- 어떤 아이템과 유사한 아이템을 추천
- Item Similarity Matrix 구하기
- 한계 : 연산량이 지나치게 많아 실시간으로 반영하기 어려움
### (2) Model-based CF
- 특성을 추출하여 embedding vectors 생성
#### 행렬 분해(Matrix Factorization)
- 아래 예시는 2차원이지만 실제로는 100차원 이상을 사용하기도 합니다.
- embedding vectors를 생성했다면 평점을 예측할 수 있습니다.
- 고객과 고객 또는 아이템과 아이템 간의 유사도를 계산할 수 있습니다.

# 3. 단골
- Factorization Machine(FM)

# Association Rules
- Source: 강의 자료
- 연관분석(Association Analysis)은 대용량의 거래(transaction) 데이터로부터 "X를 구매했으면, Y를 구매할 것이다" 형식의 아이템 간 연관 관계를 분석하는 방법입니다.
보통 장바구니 분석(Market Basket Analysis)로 불리기도 합니다. 즉, 고객의 장바구니에 어떤 아이템이 동시 담겼는지 패턴을 파악하여 상품을 추천하는 방법입니다.
## 2. 연관 분석의 주요 지표
- 연관분석에서는 크게 지지도, 신뢰도, 리프트라는 세 가지 지표를 통해 아이템 간의 관계를 표현합니다. 각각의 의미를 알아봅시다.
### (1) Support
- 스타워즈2를 재미있게 본 유저가 있다고 합시다. 이 유저에게 어떤 영화를 추천하는 것이 좋을까요? 가장 단순한 방법은 각각의 영화를 전체 유저 중에 얼마나 되는 사람이 좋아하는지 알아보고, 많은 인기(혹은 지지)를 받은 영화를 찾아서 추천하는 것입니다. 예를 들면 대부분 사람들이 타이타닉을 선호하는 만큼 해당 유저도 타이타닉을 선호할 거으로 보는 것이죠.
이 확률값을 '지지도(Support)'라고 부릅니다. 전체 유저 중에 스타워즈 3, 스타트렉, 러브액츄얼리, 타이타닉을 선호하는 유저의 수를 각각 구하면 알 수 있습니다.
### (2) Confidence
- "유저가 스타워즈2를 재미있게 보았다"는 정보를 이용해서 영화 스타워즈3를 좋아할 확률을 보다 정확하게 알 수는 없을까요? 스타워즈2를 좋아하는 유저 중에는 대상 영화를 좋아하는 유저가 얼마나 되는지 알아볼 수 있을 것입니다. 스타워즈2를 좋아하는 유저들 중에 대상 영화를 좋아했던 유저가 많다면, 이 유저 역시 대상 영화를 좋아할 확률이 높다고 보는 것이죠. 이 확률값을 '신뢰도(Confidence)'라고 하며, 영화X를 좋아하는 유저 중에 영화Y를 좋아하는 유저(즉, 영화X와 영화Y를 모두 좋아한 유저)의 비율로 계산합니다.
### (3) Lift
- 그렇다면 스타워즈2를 선호했다는 사실이 대상 영화 대한 선호를 파악하는데 얼마나 중요했을까요? 유저 전반적으로 대상 영화 Y를 좋아할 확률(지지도)보다 스타워즈2라는 영화를 좋아하는 사람 중에 대상 영화 Y를 좋아할 확률(신뢰도)이 더 크다면, 스타워즈2를 선호한다는 사실이 대상 영화Y를 선호할 것으로 예상하는 데에 대한 확신을 높여줄 것입니다.
confidence(StarWars2→StarWars3)>support(StarWars3)
- 반면에, 전반적으로 타이타닉을 좋아할 확률이 스타워즈2를 좋아하는 사람 중에 타이타닉을 좋아할 확률(신뢰도)이 더 높다면, 타이타닉과 스타워즈2의 연관관계는 높지 않을 것입니다.
confidence(StarWars2→Titanic)<support(Titanic)
- 이처럼 지지도와 신뢰도를 이용해 아이템의 관계를 파악하는 지표가 바로 리프트(Lift)입니다. 리프트는 어떻게 구할까요?
lift(StarWars2→Y)=confidence(StarWars2→Y)support(Y)
- 리프트가 1보다 크면 전자의 상황을, 1보다 작으면 후자의 상황을 뜻하는 것이죠. 스타워즈2를 재미있게 보았다는 정보를 얻고 나니 대상 영화Y를 재미있게 볼 확률이 기본 확률값(지지도)에 비해 높아졌는지, 낮아졌는지 확인하는 것이죠. '리프트'라는 지표의 이름은 "어떤 증거가 신뢰도를 높여주는가?"라는 의미에서 나온 것입니다.
- Source: https://yamalab.tistory.com/86?category=747907
- Association Rule은 고객들의 상품 묶음 정보를 규칙으로 표현하는 가장 기본적인 알고리즘이다. 흔히 장바구니 분석이라고도 불린다. 데이터마이닝 같은 수업을 들었다면 한번 쯤 들어봤을 법한 알고리즘이다. 이 알고리즘은 기초적인 확률론에 기반한 방법으로, 전체 상품중에 고객이 함께 주문한 내역을 살펴본 뒤 상품간의 연관성을 수치화하여 나타내는 알고리즘이다. 매우 직관적이고 구현하기도 쉽지만, 그렇다고 현재로서 성능이 매우 떨어지는 알고리즘도 아니다. 추천 시스템에서 여전히 가장 중요한 알고리즘으로 분류되며 Association Rule에서 파생된 다양한 알고리즘들이 존재한다.
# Multi Armed Bandit
- source: https://brunch.co.kr/@chris-song/62
- N개의 슬롯머신이 있다. 각각의 슬롯머신은 수익률이 다르다. 그런데, 당장 나는 각 슬롯머신의 수익률을 알지는 못한다. 여기서 내 돈을 어느 슬롯머신에 걸고 슬롯머신의 암(손잡이)을 내려야 돈을 제일 많이 벌 수 있을까? 여기서 슬롯머신이 밴딧(Bandit)이고, 슬롯머신의 손잡이가 암(Arm)이다.  다양한 슬롯머신이 있는데 내 돈을 어디에 걸고 손잡이를 내려야 하나? 마지막으로, 카지노에는 여러 개의 슬롯머신이 있기 때문에, 이 문제의 이름은 Multi-Armed Bandits가 된다.
## 탐색과 활용(Exploration and Exploitation)
### 전략 1. Greedy
- 한 번씩 플레이 한 후, 점수 좋은 슬롯 머신에 몰빵
- 여기서 문제는 무엇일까? 한 번씩만 테스트 했다는 점이다. 탐험(Exploration)이 충분히 이루어지지 않았다.
### 전략 2. e-Greedy
- 동전을 던져서 윗면이 나오면 점수 좋았던 슬롯머신, 뒷면이 나오면 랜덤으로 선택
- 여기서 동전의 앞면이 나올 50%의 확률이 입실론(epsilon)이라는 하이퍼파라미터다.
### 전략 3. UCB(Upper-Confidence-Bound)
- 좋은 수익률을 보이며 최적의 선택이 될 가능성이 있는 슬롯머신을 선택한다.
## Thompson Sampling
- source: https://brunch.co.kr/@chris-song/66
- 톰슨 샘플링은 배너를 클릭할 확률을 베타 분포로 표현을 했습니다. 
- 어떻게 배너를 클릭할 확률을 베타 분포로 표현할 수 있을까요? 우리에게 필요한 숫자는 두가지입니다. 하나는 배너를 보고 클릭한 횟수, 그리고 배너를 보고 클릭하지 않은 횟수입니다.
- Beta(배너를 클릭한 횟수 + 1, 배너를 클릭하지 않은 횟수 + 1)
- greedy 알고리즘: 경험상 가장 성능이 좋았던 선택지만을 활용하기 때문에 banner1을 선택합니다.
- e-greedy 알고리즘: 확률적으로 경험상 가장 성능이 좋았던 선택지만을 활용하기 때문에 banner1을 선택하거나 랜덤 선택지를 선택합니다.
- 톰슨 샘플링에서는 3개의 분포에서 임의의 점을 샘플링해보겠습니다.
- 3개 배너의 베타 분포에서 그래프 하단의 밑 면의 넓이는 모두 1입니다. 
- 특정 x값에 해당하는 분포의 y값이 클 수록 발생할 확률이 높다는 의미입니다. 
## A/B Test
- A/B 테스트를 하면 몇 가지 문제가 있다. 
  - 테스트를 하는 데 오래걸리고 비용이 많이 든다.
  - A/B 테스트를 할 때 A안이 훨씬 좋았다면 테스트 기간 동안엔 B안으로 인해 결국 손해를 보게 된다.
  - A/B 테스트를 할 때는 A안이 좋았는데 일주일 지나니 B안 반응률이 더 좋아졌다.
- 이런 문제들에 멀티암드밴딧을 사용하면 문제가 말끔이 해결된다. 일일이 설명하지는 않겠다.

# Redis
- sources: https://medium.com/@jyejye9201/%EB%A0%88%EB%94%94%EC%8A%A4-redis-%EB%9E%80-%EB%AC%B4%EC%97%87%EC%9D%B8%EA%B0%80-2b7af75fa818, https://hwigyeom.ntils.com/entry/Windows-%EC%97%90-Redis-%EC%84%A4%EC%B9%98%ED%95%98%EA%B8%B0-1
- REDIS(REmote Dictionary Server)는 메모리 기반의 “키-값” 구조 데이터 관리 시스템이며, 모든 데이터를 메모리에 저장하고 조회하기에 빠른 Read, Write 속도를 보장하는 비 관계형 데이터베이스이다.
- 레디스는 크게 5가지< String, Set, Sorted Set, Hash, List >의 데이터 형식을 지원한다.
- Redis는 빠른 오픈 소스 인 메모리 키-값 데이터 구조 스토어이며, 다양한 인 메모리 데이터 구조 집합을 제공하므로 사용자 정의 애플리케이션을 손쉽게 생성할 수 있다.

# Frequent Set
- 빈번하게 등장한 아이템의 쌍을 빈발집합이라고 부릅니다. 앞서 연관분석의 세 가지 주요 지표의 수식을 떠올려보면, 모두 빈도(Freq())를 이용해 만들어졌음을 알 수 있습니다. 연관분석을 실제 구매 데이터에 적용한다면, 각각의 아이템의 쌍이 얼마나 등장했는지를 세어야 할 것입니다.하지만 아이템의 가짓수가 늘어나고, 확인해야 할 바스켓의 수가 커지면, 이에 대한 계산은 기하급수적으로 늘어나게 됩니다.
이 문제를 해결하기 위해 고안된 것이 자주 등장하는 아이템의 쌍만을 빠르게 추려 계산하는 빈발집합 탐색 알고리즘입니다. 대표적인 빈발집합 탐색 알고리즘으로는 Apriori 알고리즘과 FP-Growth 알고리즘이 있습니다. 둘 다 데이터 셋 내에서 빈발집합을 찾아내고, 몇 번이나 등장했는지를 세어주는 알고리즘으로, 두 알고리즘의 결과는 동일합니다. 코드의 최적화 수준에 따라 조금씩 달라지지만, 일반적으로 FP-Growth 알고리즘이 Apriori 알고리즘보다 빠릅니다.
이번에는 Apriori 알고리즘을 사용하겠습니다. Apriori 알고리즘은 모든 가능한 조합의 개수를 줄이는 전략을 사용합니다.아래 이미지를 보면, 5가지 아이템이 있다고 할 때, 이 5가지를 이용해 나올 수 있는 가능한 조합은 총  25−1=31 개 입니다.아이템 수가 늘어날수록 아이템 조합 역시 급격하게 늘어날 것입니다.
Apriori는 각 조합의 지지도를 구하면서 조합의 아이템 수를 늘리며 내려가면서 어떤 조합의 지지도가 일정 기준 이하로 떨어지면, 그 아래로 내려가도(즉, 조합의 아이템 수를 늘리더라도) 빈발집합이라고 볼 수 없다 판단하여 더 이상 가지를 따라 내려가지 않고 쳐내는 식으로 빈발집합을 탐색합니다.
- <img src="https://i.imgur.com/pZ75IjW.png">

# Factorization Machine
- sources: https://zzaebok.github.io/machine_learning/factorization_machines/, https://greeksharifa.github.io/machine_learning/2019/12/21/FM/

# DeepFM
- source: https://greeksharifa.github.io/machine_learning/2020/04/07/DeepFM/
- 모델에 대해 설명할 것이다. 이 DeepFM이라는 모델은 FM과 딥러닝을 결합한 것이다. 최근(2017년 기준) 구글에서 발표한 Wide & Deep model에 비해 피쳐 엔지니어링이 필요 없고, wide하고 deep한 부분에서 공통된 Input을 가진다는 점이 특징적이다.
- DeepFM은 피쳐 엔지니어링 없이 end-to-end 학습을 진행할 수 있다. 저차원의 interaction들은 FM 구조를 통해 모델화하고, 고차원의 interaction들은 DNN을 통해 모델화한다.
- DeepFM은 같은 Input과 Embedding 벡터를 공유하기 때문에 효과적으로 학습을 진행할 수 있다.
- source: https://orill.tistory.com/category/RecSys
- 최근 구글에서 발표한 Wide & Deep Model과 비교해보면, DeepFM은 Wide 부분과 Deep 부분이 공통된 입력(shared input)을 받고 Feature Engineering이 필요하지 않다.
- 낮은 차원, 높은 차원의 상호 작용 둘 다를 모델링하기 위해 [Cheng et al., 2016]은 linear("wide") 모델과 deep 모델을 결합한 흥미로운 혼합 네트워크 구조(Wide & Deep)를 제시했다. 이 모델에서는 두 개의 다른 inputs이 wide 부분과 deep 부분을 위해 각각 필요하다. "wide part"의 입력은 여전히 전문가의 feature engineering이 필요하다.
- 이 모델은 FM을 통해 낮은 차원의 피쳐 상호작용을 학습하고 DNN을 통해서는 높은 차원의 피쳐 상호작용을 학습한다. Wide & Deep 모델과는 다르게 DeepFM은 end-to-end로 feature engineering이 필요 없이 학습할 수 있다.
- DeepFM은 Wide & Deep과는 다르게 같은 input과 embedding layer를 공유하기 때문에 효율적으로 학습할 수 있다. Wide & Deep 에서는 input vector가 직접 고안한 pairwise 피쳐 상호작용을 wide part에 포함하기 때문에 입력 vector가 굉장히 커질 수 있고 이는 복잡도를 굉장히 증가시킨다.
- 1) 다른 field의 input vector의 길이는 다를 수 있지만 embedding은 같은 크기(k)이다. (역자 예시: gender field는 보통 length가 남, 여 2인 반면 국적이나 나이 field의 길이는 더 길다. 하지만 embedding시에는 똑같이 k=5차원 벡터로 임베딩 된다.)

# Evaluation Metrics
## MAP(Mean Average Precision)
- 하지만 사용자에 따라서 실제로 소비한 아이템의 수가 천 개, 2천개까지 늘어날 수 있습니다. 이 경우 recall이 1이 되는 지점까지 고려하는 것은 무리이므로 최대 n개까지만 고려하여 mAP를 계산하며, 이 경우 mAP@n 으로 표기합니다.
## nDCG(normalized Discounted Cumulative Gain)
- 추천 엔진은 기본적으로 각 아이템에 대해서 사용자가 얼마나 선호할 지를 평가하며, 이 스코어 값을 relevance score라고 부릅니다. 그리고 이 relevance score 값들의 총 합을 Cumulative Gain(CG)라고 부릅니다. 먼저 위치한 relavance score가 CG에 더 많은 영향을 줄 수 있도록 할인의 개념을 도입한 것이 Discounted Cumulative Gain(DCG)입니다.
- <img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https://blog.kakaocdn.net/dn/dikjW1/btqDvgFUh3K/GdjWcm9XS4zpqECsQx9Nu1/img.png" height="80px">
- 하루에 100개의 동영상을 소비하는 사용자와 10개의 동영상을 소비하는 사용자에게 제공되는 추천 아이템의 개수는 다를 수 밖에 없습니다. 이 경우 추천 아이템의 개수를 딱 정해놓고 DCG를 구하여 비교할 경우 제대로 된 성능 평가를 진행할 수 없습니다. 때문에 DCG에 정규화를 적용한 NDCG(normalized discounted cumulative gain)이 제안됩니다. NDCG를 구하기 위해서는 먼저 DCG와 함께 추가적으로 iDCG를 구해주어야 합니다. iDCG의 i는 ideal을 의미하며 가장 이상적으로 relavace score를 구한 것을 말합니다. NDCG는 DCG를 iDCG로 나누어 준 값으로 0에서 1 사이의 값을 가지게 됩니다.
- <img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https://blog.kakaocdn.net/dn/r9q0s/btqDxJzAlCa/Pccibd37tTjavu1QBeMZeK/img.png" height="200px">
## Entropy Diversity
- 그런데 아직 전체 아이템에서 얼마나 다양하게 추천을 진행했는지는 평가하지 못했습니다. Entropy Diversity는 추천 결과가 얼마나 분산 되어 있느냐를 평가하는 지표입니다.
- Entropy는 섀넌의 정보 이론에서 등장한 개념으로 머신러닝에서도 많이 사용됩니다. 간략하게 설명하면 잘 일어나지 않는 사건의 정보량은 잘 일어나는 사건의 정보량보다 많다는 것입니다. 이를 사건이 일어날 확률에 로그를 씌워서 정보량을 표현하며 로그의 밑의 경우 자연 상수를 취해줍니다. (다른 상수도 가능하긴 합니다.)
- <img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https://blog.kakaocdn.net/dn/bOwzN3/btqDxt4UBcd/s0Kx6WWXyITuoKXaSZMAsk/img.png" height="250px">
- entropy란 발생할 수 있는 모든 사건들의 정보량의 기대 값입니다.
- <img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https://blog.kakaocdn.net/dn/sXcph/btqDwdoeBsv/IjVthd4jsDV0jt3p6MCfT0/img.png" height="70px">
- Entropy Diversity란 이러한 엔트로피의 개념을 추천 결과에 적용한 것입니다. 모든 사용자들에게 비슷한 종류의 상품을 추천할 경우 해당 상품 추천은 자주 발생하므로 정보량이 낮습니다. 반면 개인에게 맞춤화 된 추천은 발생 횟수가 적으므로 정보량이 높아집니다. 이들의 기대값을 구한 것이 바로 Entropy Diversity입니다.
- 그러나 Entropy Diversity 만으로 추천 엔진이 더 정확하다고 평가할 수 는 없습니다. 어디까지나 추천 결과의 다양성을 측정하는 지표이므로 mAP나 NDCG처럼 정확도를 측정할 수 있는 지표와 함께 사용하는 것이 바람직해 보입니다.