# Assignment 1

Attention mechnism과, 이를 통한 VIT를 사용해보는 실습 과제.

데이터의 Dimension을 고려하여, 컨볼루션을 통해 패치로 나눈다.

처음에는 patch로 나누기를 진행 후 컨볼루션을 진행하기 위해

torch.unfold 함수를 이용해 가로, 세로로 분할 후 컨볼루션 연산을 수행하였지만.. 무언가 오류가 났다.

몇 번의 오류 후, 위에서 이미 Convolution layer에서 stride와 kernel size를 같게 함으로써 분할 역할을 함께 수행하게 정의했다는 것을 깨닫게 되었다. unfold 한 patch들을 위한 convolution layer가 아니었기 때문에 사이즈 오류가 난 것이다.

1D embedding Vector에 대한 오해가 풀렸다. 가로 세로를 고려하지 않고 idx를 받기 때문에, 세로 방향으로는 locality를 무시할 것이라 생각했지만.. Matrix를 보니 vector가 1D일 뿐이고 모든 patch들에 대해 유사도를 학습하기 때문에 가로뿐만 아니라 세로 방향 지역 유사성도 이용한다.

# Assignment 2

Data를 Augmentation해서 pretrain된 모델을 Fine-Tuning해보는 과제


또한 pyplot의 `imshow()` 함수는 (x,y,channel) 차원을 받는데, 일반적으로 torch에서는 (batch,channel,x,y) 형식으로 이미지를 표시한다. batch는 무시하더라도, `permute()` 함수를 통해 channel 위치를 조정해 주어야 image를 show할 수 있다.

또한 이번 과제에서는 사용되지 않았지만 언급이 된 내용으로는, BGR과 RGB의 차이가 있다. 이 과제의 내용에서는 RGB -> BGR로 바꾸는 과정이 없었기 때문에 그대로 사용해도 되었지만, 모델에 들어가서 학습 중인 상황에서 image를 보기 위해서는 순서를 바꾸어 줄 필요가 있겠다.

`resize()` 만 한 것보다는 여러 가지 Augmentation을 넣은 게 정확도가 살짝 높았다. 

특이한 점으로는, Train data에 normalize를 적용했더니 오히려 정확도가 줄어들었다. normalize를 하는 행위가 이미지가 가지고 있는 데이터를 손실시키는 것 같다. 마지막 질문에 대답하는 과정에서 생각이 났는데, 이미 ViT 모델은 normalize를 하지 않은, 데이터의 밝기의 편차 등을 조금이라도(엄청나게는 아니지만) 사용하기 때문에 이를 방해하는 Data를 넣어버린 게 아닌가 싶다.

곰곰히 생각해보니 normalize는 Augmentation도 아니다.

+ 피어 세션에서 나온 이야기인데, Augmentation의 순서가 생각보다 영향을 끼치는 것 같다. rotate 등의 선형 변환과의 순서는 상관 없다고 생각했는데, "빈 자리"를 검은색으로 채우는 과정에서 색 변환과의 순서에 따른 결과 차이가 발생하는 것 같다.

# Assignment 3

# Assignment Advanced

Visualization을 하는 과정이었는데..

모델을 디버깅하기 위해 중간에 hooking하는 법을 알게 된게 가장 큰 수확 같다.

모델의 파라미터 갯수 등을 가져오고, CNN Visualization을 위해 중간 커널, 그래디언트 등을 가져오는 방법을 보았다. parameters의 인ㄴ자들이나..

saliency를 그리는 과정에서는 softmax를 통과하지 않는 게 신기했다. 이전 과제에서도 그렇고, visualize를 잘 보여주기 위해서는 softmax 단을 거치지 않는 것으로 보인다.

## hooking

tensor 혹은 module에 걸 수 있다.

forward, backward 모두 걸 수 있으며, 해당 텐서 혹은 모듈을 지나갈 때 해당 hooking function 또한 지나가게 된다(input, output도 받는 hooking function을 정의해야 한다)

여기서 내가 실수한 점이, module의 forward에는 `register_forward_hook` 으로 정의를 잘 했는데,

backward를 할 때 tensor 그 자체에 `register.hook` 을 걸지 않고 `tensor.grad_fn` 에 `register.hook` 을 걸었다.

이 경우, `grad_fn` 에 흘러 들어오는 모든 gradient에 대해 따로 계산을 하기 때문에.. "여러 인자"를 받는 알맞은 hook function을 만들어 주었어야 했다. 나의 경우에는 `grad_fn.register.hook` 의 함수가 하나의 인자가 아니라 두 개의 인자를 받아야 해서 오류를 냈는데, `tensor`에 바로 걸어 주니 문제가 해결되었다.

참고로, `grad_fn.register.hook` 에 흘러온 두 개의 인자는 grad_output, grad_input으로 보인다( 이전 단으로 가는 gradient, 다음 단에서서 들어온 gradient 순서로...) ReLU였기 때문에 큰 차이는 없었지만, 둘의 차이를 뺀 값을 본 결과 굉장히 이상한 위치에 값이 있었던 걸로 보아, 음의 값 부분만큼 차이가 나는 것으로 보인다.

# Weekly mission & pair review

직접 resnet을 계층적으로 구현해 보는 과제.

convblock -> resblock -> resnet을 계층적으로 만들고,  resnet은 resblock 등을 nn.sequential로 묶고, 마지막엔 `nn.AvgPool2d`를 통해 커널 수 만큼의 output을 뱉어 fully connected layer로 보낸다.

이렇게 학습한 모델과, 같은 구조를 갖는 pre-trained model의 마지막 fc layer만 `require_grad` 상태로 만들고 나머지는 `require_grad = False`를 준 fine-tune model과 비교를 진행했다.

우리의 모델은 train의 경우에는 어느 정도 비슷하지만 "안정적이지 않은" 상태를 보였고, validation에서는 안정적이지 않은 것 뿐만 아니라 fine-tune model에 비해 accuracy도 떨어지는 모습을 보였다.
