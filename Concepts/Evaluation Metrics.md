# regression problem
## MSE(Mean Squared Error)
## RMSE(Root Mean Squared Error)
## MAE(Mean Absolute Error)
## MPE(Mean Percentage Error)
## MAPE(Mean Absolute Percentage Error)
## R Squared
## Adjusted R Squared
## RMSLE(Root Mean Squared  Logarithmic Error)
- https://shryu8902.github.io/machine%20learning/error/
# classification problem
### confusion matrix
- 정답 클래스와 예측 클래스의 일치 여부를 센 결과. 정답 클래스는 행(row)으로 예측한 클래스는 열(column)로 나타낸다.
- 출처 : https://datascienceschool.net/view-notebook/731e0d2ef52c41c686ba53dcaf346f32/
## binary classification problem
- A, B 클래스 중 B 클래스를 맞히는 문제라고 가정했을 때
### accuracy(정확도)
- 전체 샘플 중 A 또는 B라고 맞게 예측한 샘플 수의 비율
### precision(정밀도)
- B 클래스에 속한다고 출력한 샘플 중 실제로 B 클래스에 속하는 샘플 수의 비율
### recall(재현율)
- 실제 B 클래스에 속한 표본 중에 B 클래스에 속한다고 출력한 표본의 수의 비율
### F1 score(정밀도와 재현율의 조화 평균)
### fall-out(위양성률)
- 실제 B 클래스에 속하지 않는 표본 중에 B 클래스에 속한다고 출력한 표본의 비율
- 출처 : https://datascienceschool.net/view-notebook/731e0d2ef52c41c686ba53dcaf346f32/
## ROC(Receiver Operator Characteristic) curve
- 위에서 설명한 각종 평가 점수 중 재현율(recall)과 위양성률(fall-out)은 일반적으로 양의 상관 관계가 있다.
재현율을 높이기 위해서는 양성으로 판단하는 기준(threshold)을 낮추어 약간의 증거만 있어도 양성으로 판단하도록 하면 된다. 그러나 이렇게 되면 음성임에도 양성으로 판단되는 표본 데이터가 같이 증가하게 되어 위양성율이 동시에 증가한다. 반대로 위양성율을 낮추기 위해 양성을 판단하는 기준을 엄격하게 두게 되면 증거 부족으로 음성 판단을 받는 표본 데이터의 수가 같이 증가하므로 재현율이 떨어진다.
클래스 판별 기준값의 변화에 따른 위양성률(fall-out)과 재현율(recall)의 변화를 시각화한 것이다.
- 출처 : https://datascienceschool.net/view-notebook/731e0d2ef52c41c686ba53dcaf346f32/
