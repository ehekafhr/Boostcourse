# 앙상블 모델

단일 모델을 사용하는 것보다, 여러 모델을 사용하는 것이 더 좋은 예측 성능을 보일 수 있다.

<img width="1280" height="873" alt="image" src="https://github.com/user-attachments/assets/17b49195-9d4d-4565-ab05-5b909dad981a" />
https://en.wikipedia.org/wiki/Ensemble_learning#/media/File:Combining_multiple_classifiers.svg

특정한 데이터셋 분포를 잘 파악하는 여러 모델을 만들어서 사용하면, 더 좋은 성능을 이끌어낼 수 있다.

일반적으로 머신러닝은 bias-variance tradeoff가 있다. bias가 높으면 예측력이 감소, variance가 높으면 일반화가 힘든 문제가 있다.

단일 모델이 되면 과대적합이 되어 variance가 높아지는 문제가 있지만, 여러 모델을 사용하면 일반화 성능이 올라가 variance가 줄어든다.


## 배깅

Bagging은 단순히 여러 모델을 병렬적으로 학습시킨 후, 그 predcition을 vote하는 방식이다.

Hard Voting과 Soft voting이 있다. Hard는 Classification에서 `argmax`를 사용한 것처럼 Probabiliity 0.9든지 0.51이던지 1로 투표하는 것이고, Soft voting은 Probabiliity를 그대로 더해 준다.

일반적으로는 Soft Voting을 사용한다.

## 부스팅

Boosting은 여러 개의 Weak learner들을 학습시키며, 다음 모델은 이전 모델이 잘 풀지 못하는 데이터셋을 집중적으로 학습한다.

그렇게 한 뒤 모델들을 앙상블하면, 모델들이 서로 예측하기 어려운 부분을 보완해 준다.

## K-fold

K-fold 교차 검증을 한 뒤, 그 K개의 모델들을 ensemble한다. 하지만 K배로 학습시켜야 한다..

## Stacking

단순히 hard, soft voting을 하는 것이 아니라 모델별로 weight를 주어 모델별 가중치를 준다.

Stacking도 학습 비용이 높지만, 복잡한 패턴 학습이나 유연성이 장점이다.

### TTA

Test-time에 sample을 Augmentation하여, 한 sample에 대해 여러 번 추론한다.

### MOE

입력에 따라 router가 어떤 expert에 데이터를 넣을지를 선택. sigmoid나 softmax를 통과시켜서 각각의 expert model의 결과를 얼마나 반영할지 정할 수 있다.
