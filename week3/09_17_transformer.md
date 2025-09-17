# 9_17 Transformer

## RNN

NN이나 CNN과 달리, "시계열 데이터"를 처리 가능한 모델로,

Hidden state(weight과는 다름) 을 모델이 가지고 있다.

Output을 뱉을 때 hidden state가 작동하며,

모델이 내밷는 Output 외에도, input에 따라 Hidden state가 업데이트된다.

$$ h_t = f_W(h_{t-1},x_t) $$

이렇게 하면, 가변적인 길이의 Input을 처리 가능하다(t번 반복하면 된다!)

이전 정보들을 Hidden stater가 기억하기에, 이전 정보들을 사용 가능하다.

다만.. "Input"의 길이만큼 이 모델을 돌려 주어야 하기 때문에 계산이 느리다.

게다가, 이 작업은 "순서"를 지켜야 하기에 병렬화도 불가능하다!

또다른 단점으로는 Vanishing Gradient, Long-range dependence 모델링 실패 문제가 있다.

또한, "입력과 출력의 길이가 같다"는 문제 존재. ->번역할 때 단어 수가 같지는 않다.

### Vanishing Gradient

Backpropagation 시, $h_t$ 에서 $h_{t-1}$ 로 갈 때 hidden state를 위한 weight $W_{hh}$ 를 곱해 주게 된다.

그런데, "똑같은" $W_{hh}$ 가 t번 곱해지기 때문에 값이 불안정해진다.

만약 $|W_{hh}|$ 가 1보다 크다면 Gradient가 explode할 것이고,

1보다 작다면 Vanishing할 것이다!

앞에서 배운 적절한 초기화 기법도, "충분히 깊지 않은" 레이어에만 동작한다. RNN은? input의 길이 t가 길다면 이것이 depth로 작용하여 엄청나게 깊은 모델이기 때문에 이런 걸로도 조절이 불가능하다.

Exploding gradient는 클리핑으로 처리 가능하지만(그냥 특정 값을 최대로 해버리면 된다), Vanishing gradient는 해결이 되지 않는다.

## RNN 기반 알고리즘

Vanishing Gradient가 사라짐을 보장하지는 않는다. 하지만, 적어도 Vanila RNN보다는 낫고, 특히 Long-range dependence가 없는 문제는 해결 가능하다.

### LSTM

### GRU

## Seq2Seq

## Attention

## Transformer

## Bert

## ViT
