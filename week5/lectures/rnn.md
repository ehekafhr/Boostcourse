# RNN

3주차 강의 내용에 +a 정도로 요약하겠다.

NN이나 CNN과 달리, "시계열 데이터"를 처리 가능한 모델로,

Hidden state(weight과는 다름) 을 모델이 가지고 있다.

Output을 뱉을 때 hidden state가 작동하며,

모델이 내뱉는 Output 외에도, input에 따라 Hidden state가 업데이트된다.

$$ h_t = f_W(h_{t-1},x_t) $$

이렇게 하면, 가변적인 길이의 Input을 처리 가능하다(t번 반복하면 된다!)

이전 정보들을 Hidden stater가 기억하기에, 이전 정보들을 사용 가능하다.

다만.. "Input"의 길이만큼 이 모델을 돌려 주어야 하기 때문에 계산이 느리다.

게다가, 이 작업은 "순서"를 지켜야 하기에 병렬화도 불가능하다!

또다른 단점으로는 Vanishing Gradient, Long-range dependence 모델링 실패 문제가 있다.

또한, "입력과 출력의 길이가 같다"는 문제 존재. ->번역할 때 단어 수가 같지는 않다.

<img width="1644" height="521" alt="image" src="https://github.com/user-attachments/assets/b2e0c97a-6184-4061-bfba-0e67f5e14ba0" />

https://karpathy.github.io/2015/05/21/rnn-effectiveness/

물론, many-to-many 외에도 여러 가지 구조를 통해 단어 수 등이나 여러 태스크에 활용할 수 있다.

데이터를 전부 넣기 전에는 output이 나오지 않지만, hidden network를 두 단 쌓아 양방향으로 학습 가능하게 할 수도 있다.
(첫번째 hidden layer를 forward 방향으로 통과시키고, 그 다음 hidden layer를 backward 방향으로)

## BPTT

전체 sequence에 대한 loss를 계산하면.. 앞부분 loss는 제대로 계산이 되지 않을 수 있으므로, Backpropagation Through Time을 사용하여 Chunk 단위로 Forward/Backward를 진행한다.

## Vanishing Gradient

Backpropagation 시, $h_t$ 에서 $h_{t-1}$ 로 갈 때 hidden state를 위한 weight $W_{hh}$ 를 곱해 주게 된다.

그런데, "똑같은" $W_{hh}$ 가 t번 곱해지기 때문에 값이 불안정해진다.

만약 $|W_{hh}|$ 가 1보다 크다면 Gradient가 explode할 것이고,

1보다 작다면 Vanishing할 것이다! 대각행렬로 분해를 한다고 생각해 보면 이해가 쉬울 것이다. 대각행렬의 각 요소의 값이 1보다 큰 값은 무한대로 갈 것이고, 1보다 작은 값은 0으로 vanishing할 것이다.

앞에서 배운 적절한 초기화 기법도, "충분히 깊지 않은" 레이어에만 동작한다. RNN은? input의 길이 t가 길다면 이것이 depth로 작용하여 엄청나게 깊은 모델이기 때문에 이런 걸로도 조절이 불가능하다.

Exploding gradient는 클리핑으로 처리 가능하지만(그냥 특정 값을 최대로 해버리면 된다), Vanishing gradient는 해결이 되지 않는다.

# RNN 기반 알고리즘

Vanishing Gradient가 사라짐을 보장하지는 않는다. 하지만, 적어도 Vanila RNN보다는 낫고, 특히 Long-range dependence가 없는 문제는 해결 가능하다.

## LSTM

<img width="1055" height="586" alt="image" src="https://github.com/user-attachments/assets/27a627c1-231c-416d-a18b-b67f21de9383" />

https://en.wikipedia.org/wiki/Long_short-term_memory

LSTM은 long-term으로 기억하는 Cell state를 추가했다. Cell state는 FC를 통과하며 업데이트되지 않는다.

가장 왼쪽의 sigmoid를 통과한 X가 Forget gate로, 얼마나 이전의 cell state를 기억할지를 결정한다. 

tanh 게이트와 2번째 sigmoid를 통과한 값을 곱하는 것을 input gate로, 이것과 

마지막으로는 cell state와 무관한 값을 넣어주기 위해 sigmoid를 취해 Output gate로 사용하여,

cell state를 input gate 값과 더해 update하고 이를 tanh 게이트로 넘겨 Output gate로 곱해 $h_t$ 를 업데이트하고 output을 뱉는다.

Resnet의 residual처럼 생각하면 될 것 같다. 다만, 그 비율을 정하는 두 함수가 있다.(input gate와 forget gate로, 두 합이 1이 되는 기존의 moving average와는 조금 다름)

## GRU

<img width="500" height="250" alt="image" src="https://github.com/user-attachments/assets/fe33af0b-db67-4c5e-a386-642ab071017c" />

https://en.wikipedia.org/wiki/Gated_recurrent_unit

LSTM의 경량화된 모델로, Cell state와 Hidden state를 하나의 Vector로 통합하여 사용.


## Seq2Seq

<img width="1548" height="763" alt="image" src="https://github.com/user-attachments/assets/32990dd0-a479-41f3-8da8-186d1fbd2d34" />

https://medium.com/data-science/sequence-to-sequence-model-introduction-and-concepts-44d9b41cd42d

RNN을 인코더 - 디코더 구조로 만든다.

인코더 RNN에서 y를 뱉는 대신에, 마지막 hidden state를 디코더의 input vector로 사용한다!

디코더 RNN 는 이 hiddent state를 첫번째 hidden state로 받아, [<SOS>: 시작 토큰]을 첫 input으로 넣고 output을 뱉고 그 뱉은 output을 다시 input으로 사용한다. "EOS" 토큰이 나올 떄까지 반복한다.

이때, 완전히 정보를 주지 않으면 학습이 제대로 진행되지 않기 때문에, 학습 시에는 output을 다시 input으로 넣지 않고, "정답"을 input으로 넣어 준다.(Teacher Forcing)
