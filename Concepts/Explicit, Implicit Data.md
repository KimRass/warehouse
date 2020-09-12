- source : https://orill.tistory.com/entry/Explicit-vs-Implicit-Feedback-Datasets [이제 며칠 후엔]
### 1. Explicit Datasets
- 영화 추천시스템을 만드는 상황을 생각해보겠습니다. 어떤 데이터를 사용할 수 있을까요?
- 가장 먼저 떠오르는 데이터는 사용자의 평점 기록 데이터입니다. 이미지 분류에 MNIST가 있다면 추천시스템에는 Movielens Data가 있습니다. ratings.csv 파일은 (userId,movieId,rating,timestamp) 형태의 데이터를 저장하고 있습니다. 유저가 본 영화에 대해서 5.0점을 만점으로 0.5점 단위로 평가한 데이터가 시간과 함께 저장되어 있습니다. Watcha는 이런 종류의 데이터를 수집하여 추천 앱을 서비스하고 있고 Neflix는 평점대신 좋아요, 싫어요 데이터를 수집하고 있습니다. 평점 기록처럼 유저가 자신의 선호도를 직접(Explicit) 표현한 Data를 Explicit Data라고 합니다. 다른 종류의 Explicit Data로는 영화 리뷰, 구독, 차단 데이터 등이 있습니다.

Explicit Data를 통해서 얻을 수 있는 정보는 강력합니다. 유저의 호불호를 명백히 알 수 있기 때문입니다. 유용성이 좋은 반면 데이터를 얻기 힘들다는 단점이 있습니다. 데이터 분석을 해보면 영화를 보고 평점을 매기는 유저의 숫자는 전체 유저에 비해 적고 리뷰까지 남기는 데이터는 훨씬 적습니다. 유저가 적극적인 Action을 취해야 하는데 단순히 귀찮거나 심리적인 거부감이 있기 때문이죠. "좋아요와 구독 부탁드려요"라는 말은 유저들이 웬만해서는 '좋아요'를 눌러주지 않는다는 점을 방증하고 있는 듯합니다. 

### 2. Implicit Datasets
- Collaborative Filtering For Implicit Feedback Datasets 논문에 따르면 2010년 이전까지는 추천시스템 분야에서 Explicit Data를 활용하는 방법에 대한 연구가 주로 이루어졌던 모양입니다. 논문에서는 Implicit라는 다른 종류의 데이터를 활용하여 추천시스템을 만드는 방법을 제시하고 있습니다. 아래부터는 이 논문에서 소개한 Implicit Datasets의 개념과 특징을 정리한 내용입니다. Model 부분에 추후에 포스팅할 예정입니다.

Implicit Data는 유저가 간접적(Implicit)으로 선호, 취향을 나타내는 데이터를 의미합니다. 예시로는 검색 기록, 방문 페이지, 구매 내역 심지어 마우스 움직임 기록이 있습니다. 이런 데이터는 일단 유저가 개인정보제공에 동의만 한다면 자동적으로 수집되기 때문에 수집의 난이도 낮고 활용할 수 있는 데이터량이 많습니다.

Implicit Datasets을 다룰 때 염두해두어야 할 몇 가지 특징이 있습니다.

부정적인 피드백이 없다(No Negative Feedback) : 유저가 어떤 영화를 본 기록이 없을 때 유저가 그 영화를 싫어해서 보지 않았는지 혹은 그저 알지 못했기 때문에 보지 않았는지 알 수 없습니다. 이런 차이점 때문에 Explicit Feedback Dataset을 다룰 때는 수집된 데이터에만 집중하고 Unobserved Data는 Missing Value 취급하여 모델을 만들어도 괜찮습니다. 유저의 불호 정보(낮은 평점, 싫어요)가 포함되어 있기 때문입니다. 하지만 Implicit Data를 모델링할 때는 수집되지 않은 데이터도 같이 모델링해야 합니다. 수집되지 않은 데이터에 (확실하지는 않아도) 불호 정보, 부정적인 정보가 담겨 있을 가능성이 크기 때문입니다.

애초에 잡음이 많다(Inherently Noisy) :  반대로 어떤 영화를 봤다고 해서 유저가 그 영화를 좋아한다고 말하기 힘듭니다. 과제 때문에 영화를 본 것일 수도 있고 영화가 마음에 안들지만 구매한 게 아쉬워서 끝까지 본 것일 수도 있기 때문입니다. 유튜브의 경우 시청시간이 중요하다는 말이 있는데 유저가 영상을 틀어놓고 잠들었을 수도 있습니다.

수치는 신뢰도를 의미한다.(The numerical value of implicit feedback indicates confidence) : Explicit Data의 경우 높은 수치는 높은 선호도를 의미합니다. 2번에서 봤듯이 Implicit Data에서는 높은 수치가 꼭 높은 선호도를 의미하는 것은 아닙니다. 인생 영화를 봤어도 시청시간은 2시간 즈음인 반면 그저 그런 드라마를 보는 경우에 10시간을 넘게 볼 수도 있습니다. 그럼에도 높은 값은 신뢰할만한 데이터임을 의미한다고 해석할 수 있습니다. 한 번 보고만 영상보다는 자주, 오래 본 영상이 유저의 선호도, 의견을 표현했을 확률이 높기 때문입니다.

Implicit-feedback Recommender System의 평가는 적절한 방법을 고민해봐야 한다 : 평점 데이터를 이용하는 경우 예측값과 실제값이 얼마나 다른지를 평가하는 Mean Squared Error 방법을 사용하면 편리합니다. 하지만 시청시간, 클릭수, 조회 기록을 이용하는 경우 정답값을 주기가 어렵습니다. 따라서 implicit model의 경우 item의 availability나 반복되는 feeback 등을 고려해야 합니다. availability란 동시간에 방영되는 두 TV Show의 경우 한쪽만 볼 수 있어서 다른 프로그램을 좋아한다고 해도 Implicit Data가 쌓이지 않는 상황을 말합니다. 반복되는 Feedback은 유저가 한 번 이상 프로그램을 봤을 때 한 번 본 경우와 어떻게 다르게 평가할 것인가에 대한 고려입니다.

### 3. Collaborative Filtering For Implicit Feedback Datasets
- 다음 포스팅에서는 앞서 언급한 Implicit Datasets의 특징을 고려해서 어떻게 추천 모델을 만들고 평가할지에 대한 내용을 정리해보겠습니다. 짧게 정리하자면 Unobserved Data와 Observed Data를 구분하고 높은 Confidence Data에 높은 Loss를 의미하는 Loss Function을 정의하여 Matrix Factorization을 수행합니다. Evaluation은 Recall 지표를 사용하는 것이 타당하다고 생각하여 Expected Percentile Ranking을 사용합니다.
