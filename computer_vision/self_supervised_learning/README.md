# History
- Exemplar -> Context Prediction -> Jigsaw Puzzle -> Count
# Exemplar
# Context Prediction
- 한계: 모델이 위치 정보를 Boundary, Texture 등으로 Cheating하여 Trivial Solution으로 빠질 수 있습니다.
# Jigsaw Puzzle
- 3x3의 9개의 Patch를 추출합니다.
- 그 순서를 Suffle합니다. 이때 모든 경우의 수는 9! (36만 여개)나 되므로 이를 Softmax를 이용해서 학습시키는 것은 매우 어렵습니다. 따라서 이 중 100개의 Pre-defined 경우만을 사용해서 학습시킵니다.
- 성능: Number of permutations가 클수록 높아집니다. Minumum hamming distance가 클수록 (많이 섞을수록) 높아집니다.
# Count
- 이미지를 다수의 Patch로 나누고, 각 Patch 안의 Object의 특징을 나타냅니다. (e.g., 코 1개, 눈 2개, 발 3개, ...)
# References
- https://greeksharifa.github.io/self-supervised%20learning/2020/11/01/Self-Supervised-Learning/
