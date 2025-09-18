# 강의정리

## Attention

RNN-based model은 어떻게 LSTM 등으로 Long-Term memory를 저장하든,

"긴 sequence"에서 앞의 정보가 소실되는 것을 방지할 수 없다.

Seq2Seq 모델에서도,

인코더의 "마지막" HIDDEN STATE 를 사용하기 때문이다.

따라서, input step의 "모든 hidden state"를 고려하는 방식을 고려한다.

### Mechnism

$$ Attention Value = softmax(\frac{QK^T}{\sqrt{d_k}})V $$

Idea: Query와 Key들의 유사도를 비교 -> softmax 이후 Value와 곱해 준다.

여기서 Query를 Context,

Key - Value를 references라고 보면 된다.

Attention 모델, Transformer 등에서는 Key와 Value가 같다. (Attention에서는 Input의 "모든 step의" hidden state들)

즉, Query와 hidden states를 dot-product해서 유사도를 계산하여, 유사한 만큼 Value를 곱해 준다는 것이다.

이렇게 계산한 Attention Value를 원래 vector에 concat해준 뒤 output(next input)을 계산한다.

<img width="953" height="740" alt="image" src="https://github.com/user-attachments/assets/05c4c0ff-0f02-4da5-a867-763e60160c62" />

https://en.wikipedia.org/wiki/Seq2seq

<img width="655" height="579" alt="image" src="https://github.com/user-attachments/assets/02b56c59-d316-459b-9fc5-9f736310dbc9" />
https://www.tensorflow.org/tutorials/text/nmt_with_attention?hl=ko

Attention mechnism의 또다른 장점은 input vector와 output vector들의 관계를 시각화할 수 있다는 것이다. 스페인어 todavia는 "you"에 대응되는 식으로, 관계를 시각화할 수 있다.

## Transformer

기본적으로 RNN처럼 시계열 데이터를 처리하기 위함이었지만.. 아래에 나올 ViT처럼 이미지 데이터에도 성능이 좋다고 알려졌다.



### Self-Attantion

Q,K,V를 자기혼자 만드는 아이디어.

$W_Q$ , $W_K$, $W_V$ 를 각각 input에 곱해 주어 Q,K,V를 만든 뒤 Attention을 진행한다!

즉, i번째 input의 경우 $Q_i$ 를 모든 $K$ 와 dot product한 후, 이를 softmax하여 모든 $V$ 에 곱해 준다. 이렇게 하면, 원래의 embedding과 유사하긴 하겠지만 다른 값들의, context의 영항도 받는다.


### Multi-Head attention

Self-Attention에서 $W_Q$ , $W_K$, $W_V$ 를 head 개수만큼(머리가 많다!) 만들어 준 뒤, concat해주는 단순한 방식이다.

그 뒤에 있는 linear transform layer 단에서 어차피 차원을 맞추어 줄 것이기 때문에, 차원은 걱정하지 않아도 된다.

### Transformer 구조

<img width="510" height="754" alt="image" src="https://github.com/user-attachments/assets/eba655c8-eb37-44b8-9f2d-90d7a104454f" />

여기서 N은 세로로 쌓았다는 것이다.. 나처럼 인코더와 디코더가 N 쌍이 있다고 생각하지 말자.

이러한 context를 학습시키기 위해 이런 단계를 N번 반복하는 것이다.

Decoder의 경우에는, 먼저 Decoder의 Input을 Multi-head attention해준다. 이때, 뒤의 값들은 사용하면 안되기 때문에(문맥 고려) Masking해준다. 그 뒤,
다음 Encoder와의 Attention에서 Q,K를 Encoder에서 가져와 문맥을 학습하고, Decoder에서 V를 가져온다.

마지막 단을 통과한 뒤, Linear와 Softmax를 통해 어떤 단어가 될지 그 확률과 함께 예측한다.

### 분류 문제

Transformer Block들을 통과한 값의 평균 등 집계 함수를 사용한 뒤,

Classifier 혹은 Regressor layer 한 단을 쌓아주면 끝이다!

그러나.. 이렇게 평균 Embedding 등을 쓰면 길어질 경우 좀 이상해질 수 있기 때문에, 하나의 output을 활용하기로 한다.

어떻게? Transformer block에서 하나의 x에 대해 하나의 output이 나올 텐데.. 그 x에 따라 결과가 달라지기 때문에, 더미 토큰 classification token CLS를 사용한다.

### Positional Encoding

Transformer는 순서가 없다! 그렇기 때문에, 순서가 중요한 데이터의 경우에 이를 학습시키기 위해 Positional Encoding을 사용한다.

Positional encoding은 sin과 cos 함수를 이용하여, 이전 위치의 데이터와 유사한 값을 가지게 해 준다.

<img width="500" height="425" alt="image" src="https://github.com/user-attachments/assets/7c699dbb-7cc0-4069-8f88-48cbeac64f03" />
source: Stanford CS 224n

이렇게, index가 비슷한 값의 색깔도 유사함을 볼 수 있다. 또한, index 차이가 많이 난다면 거의 무관한 수준임을 알 수도 있다.

## Bert( Bidirectional Encoder Representations from Transformers )

## ViT
