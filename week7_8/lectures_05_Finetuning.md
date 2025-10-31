# Fine tuning

일반적으로 AI를 학습시킬때, 대규모 데이터셋으로 미리 학습된 모델의 weight를 가져와서 일부 가중치만 태스크에 맞게 조정하는 Fine-tuning, 전이학습 기법을 많이 사용한다.

Fine tuning을 하게 되면, 단순히 우리가 가지고 있는 dataset으로 학습을 시작하는 것보다 일반화된 성능 향상을 기대할 수 있다.

사전 학습 모델은 엄청난 수의 텍스트, 이미지 등을 학습하여 데이터의 의미, 패턴 등을 파악한 상태이기 때문에 이러한 layer들을 유지하는 것이 좋다.

또한, 랜덤 가중치로 시작하는 것보다 의미를 이미 해석하는 가중치로 시작하는 것이므로 학습 속도가 빠르고, 

적은 데이터셋으로도 의미있는 결과를 뽑아낼 수 있고,

일부 Weight(일반적으로는, 마지막의 Classifier만 unfreeze한다) 만 학습시키므로 학습 시간 단축의 효과가 있다.

<img width="1320" height="1860" alt="image" src="https://github.com/user-attachments/assets/047acd5e-7f22-4740-8609-383ad9d1e250" />
일반적으로 NLP에서는 Transformer 기반 모델이 많기에, 이를 사용한 GPT나 BERT 등의 모델을 사전 학습 모델로 사용하게 된다.

## GPT

Auto-Regressive하게 문장을 생성해 준다. Transformer의 Decoder 단만 따온 것으로, 이전 값을 다시 input으로 넣어 새로운 값을 만드는 방식으로 CLS 토큰이 나올 때까지 반복하게 된다.

GPT-2의 경우에는 opensource이지만, 최신 모델들은 가중치가 공개되지 않아 API로만 사용이 가능하고, FINE-TUNE된 모델을 사용하려면 유료 파인튜닝을 해야 하는 문제가 있다.

## LLama

GPT와 유사한 방식이지만, 여러 자원 요구량이 다양하고 연구 목적으로 공개되어있는 특징을 가지고 있어 연구 목적으로 쉽게 사용 가능하다.
*상업적 사용은 제한된다.

## BERT

Transformer의 Encoder 단만 따온 것으로, 문장을 이해하는 능력에 특화되어 있다. 많은 pretrained된 모델들이 존재하여 쉽게 사용할 수 있으며, 한국어 전용 모델도 많이 있다.

MLM 패턴(마스크된 단어 맞추기)에만 집중한 RoBERTa라는 모델, 파라미터를 줄인 ALBERT, Teacher-student 방식으로 지식 증류(학생의 정답, pretrained된 모델의 정답의 차이를 loss로 학생을 가르침)를 통한 경량화 모델 등이 있다. 
