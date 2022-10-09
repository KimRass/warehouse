# Continual / Incremental Learning (IL)
- References: https://www.youtube.com/watch?v=YM7HEAlUqWg&t=756s, https://ffighting.tistory.com/entry/Incremental-Continual-learning-%EC%84%A4%EB%AA%85-%EC%84%B1%EB%8A%A5%EC%B8%A1%EC%A0%95%EB%B0%A9%EC%8B%9D-%EC%97%B0%EA%B5%AC%ED%9D%90%EB%A6%84
- 하나의 네트워크를 가지고 다수의 Sequential tasks에 대한 성능을 유지하는 것.
- Forward transfer: 과거의 Task에서 획득한 Knowledge를 통해 미래의 Task에 대한 성능을 향상시킴.
- Backward transfer: 미래의 Task에서 획득한 Knowledge를 통해 과거의 Task에 대한 성능을 향상시킴.
- Catastrophic forgetting: 네트워크를 새로운 Task에 대해 학습시키면 과거의 Tasks에 대한 성능이 떨어지는 현상
- Task-IL
- Domain-IL
- Class-IL
- ![three_scenarios](https://blog.kakaocdn.net/dn/bplV40/btrp95PZUfM/g2ASLoUQi04N9znKDAGIAK/img.png)
## Regularization-based
- Loss function에 Regularization term을 추가해서 Forgetting을 방지합니다.
### Elastic Weight Consolidation (EWC)
## Replay-based
- Training set의 크기가 커질수록 Memory 문제가 있습니다.
### Deep Genarative Replay (DGR)
- Generator: Target task의 Input data와 이전 Tasks의 Input data가 섞인 Input data를 생성하도록 학습됩니다. 
- Solver: Target task를 풉니다.
## Architecture-based
- 각 Tasks에 대한 Sub-network를 가지거나 Network가 Dynamic expandable한 형태입니다.
### Progressive Neural Networks
## DualNet