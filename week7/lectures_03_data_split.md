# 데이터 분할

일반적으로 데이터는 훈련, 검증, 테스트 데이터로 분리된다.

훈련 데이터는 학습시키는 데이터이고, 이를 검증 데이터셋을 통해 과적합 여부를 확인하고, 하이퍼파라미터를 튜닝한다.

검증 데이터가 없다면 Data Snooping Bias가 발생하는데, 실제 시스템에 배포했을 때 성능이 하락하는 현상이다.

<img width="773" height="393" alt="image" src="https://github.com/user-attachments/assets/c61c1304-d484-4009-a012-e5302fd1fbf0" />
https://databasecamp.de/en/ml/overfitting-en
이렇게 Overfitting된 모델의 경우에는 Train dataset에 대해서만 잘 동작한다.

<img width="674" height="587" alt="image" src="https://github.com/user-attachments/assets/f6eb7e5b-044f-427c-910b-bea7ff5be9e1" />
https://databasecamp.de/en/ml/overfitting-en
따라서, Train loss가 가장 적은 지점이 아니라 validation loss가 가장 적은 지점을 고르는 것이 좋다.



테스트 데이터는 모델의 최종 성능 평가용이다.

이 세 가지 데이터는 서로 영향을 주면 안 된다(예를 들어, fit_transform은 훈련 데이터에만 사용되어야 하고, 나머지에는 transform을 사용해야 한다)

일반적으로 6:2:2 분할을 사용하지만.. Data가 크면 Train data의 비율을 크게 해도 된다.

`train_test_split` 함수를 기본적으로 사용한다. 이때, `stratify = True`로 해 주는 것이 일반적이다.

`stratify` 는 분류 문제에서, 학습 - 검증 데이터셋의 분포가 같게 해 주는 효과이다. 이렇게 하지 않으면 train, val dataset의 분포가 달라져 학습 평가가 제대로 되지 않을 수 있다.

## Data Leak

모델을 학습 중에 예측하려는 정보가 노출되는 문제.

모델 성능이 과장되어, 실제 환경에서는 정확한 성능이 나오지 않는다.

즉, Train Data만 가지고 학습하는 것이 아니라 Validation / Test data가 어떠한 방식으로든 영향을 주는 상황을 의미한다.

예를 들어, 데이터를 가지고 Scaling을 진행할 때 "나누기 전에" Scaling을 하면 Validation과 Test data의 분포가 이미 정보로 주어진 것이다.

이뿐만 아니라, 시계열 데이터의 경우에는 순서도 신경을 써 주어야 한다. "예측 시점 이후의 데이터"가 들어가는 것도 데이터 leak이고, 변수가 결과에 의해 영향을 받는 경우(암 환자 진단 - 환자의 현재 복용약, 진단 결과를 나타내는 걸 사용했으므로 데이터 누수이다).

## Data Imbalance

클래스 간의 비율 차이가 많이 나는 상황.

소수 클래스 예측 성능이 떨어질 수 있다. 대부분 다수 클래스를 예측하는 것이 옳기 때문에, 소수 클래스 판별 능력이 떨어진다.

-> Undersampling, Oversampling 방식이 있다.
Undersampling은 단순히 다수 클래스를 줄이는 것이다. 이때, "좋은"데이터만 남기기 위해서, 
Near miss는 소수 클래스 데이터 포인트에 가까운 데이터를 선택한다(분류 line을 잘 만들기 위해서)
Tomek Links의 경우에는 "다른 클래스인데, 가장 가까운" data 쌍의 다수 클래스를 제거한다.

## K-Fold 교차 검증

데이터가 모자란 경우, 학습 데이터를 검증 데이터로 분할하는 것이 부담스러울 수 있다.

이러한 경우에는, 데이터를 무작위의 동일한 크기의 K개의 fold로 분할하여,

하나를 검증 데이터로, 나머지를 트레인 데이터로 K번 학습시켜 K개의 평가 지표를 통해 종합적 성능을 평가한다.

여기서도 Stratify를 적용하여 라벨별 밸런스를 마ㅣㅈ추는 것이 좋다.

시계열 데이터의 경우에는 시간의 흐름에 맞추어야 하기 때문에, K개의 dataset으로 나누면 K-1번의 학습을 진행한다. I번째 학습에서는 I개의 fold만 training dataset으로 들어가기에, 초기 분할에는 데이터셋의 양이 적어진다.

Sklearn에는 `Kfold`와 `StratifiedKfold`로 교차 검증을 지원한다.

Oversampling은 단순히 Resampling을 하거나,

동의어 대체(단어 교체), 무작위 삽입, 무작위 삭제, 무작위 교환 등(EDA) 혹은 특수기호를 추가하는 AEDA로 Augment를 한다.
다른 방식으로는, 다른 언어로 번역한 후 재번역을 통해 문장을 다시 만드는 것이 있다. LLM을 기반으로 만들 수도 있다.

하지만, 이러한 경우 문맥을 해치는 데이터셋이 생기거나 과적합이 생길 수 있기 때문에, task를 고려한 방식을 사용하는 것이 좋다.
