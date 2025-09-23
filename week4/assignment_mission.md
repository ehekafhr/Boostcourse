# Assignment 1

Attention mechnism과, 이를 통한 VIT를 사용해보는 실습 과제.

데이터의 Dimension을 고려하여, 컨볼루션을 통해 패치로 나눈다.

처음에는 patch로 나누기를 진행 후 컨볼루션을 진행하기 위해

torch.unfold 함수를 이용해 가로, 세로로 분할 후 컨볼루션 연산을 수행하였지만.. 무언가 오류가 났다.

몇 번의 오류 후, 위에서 이미 Convolution layer에서 stride와 kernel size를 같게 함으로써 분할 역할을 함께 수행하게 정의했다는 것을 깨닫게 되었다. unfold 한 patch들을 위한 convolution layer가 아니었기 때문에 사이즈 오류가 난 것이다.

1D embedding Vector에 대한 오해가 풀렸다. 가로 세로를 고려하지 않고 idx를 받기 때문에, 세로 방향으로는 locality를 무시할 것이라 생각했지만.. Matrix를 보니 vector가 1D일 뿐이고 모든 patch들에 대해 유사도를 학습하기 때문에 가로뿐만 아니라 세로 방향 지역 유사성도 이용한다.

# Assignment 2

Data를 Augmentation해서 pretrain된 모델을 Fine-Tuning해보는 과제.



또한 pyplot의 `imshow()` 함수는 (x,y,channel) 차원을 받는데, 일반적으로 torch에서는 (batch,channel,x,y) 형식으로 이미지를 표시한다. batch는 무시하더라도, `permute()` 함수를 통해 channel 위치를 조정해 주어야 image를 show할 수 있다.

또한 이번 과제에서는 사용되지 않았지만 언급이 된 내용으로는, BGR과 RGB의 차이가 있다. 이 과제의 내용에서는 RGB -> BGR로 바꾸는 과정이 없었기 때문에 그대로 사용해도 되었지만, 모델에 들어가서 학습 중인 상황에서 image를 보기 위해서는 순서를 바꾸어 줄 필요가 있겠다.

`resize()` 만 한 것보다는 여러 가지 Augmentation을 넣은 게 정확도가 살짝 높았다. 

특이한 점으로는, Train data에 normalize를 적용했더니 오히려 정확도가 줄어들었다. normalize를 하는 행위가 이미지가 가지고 있는 데이터를 손실시키는 것 같다. 마지막 질문에 대답하는 과정에서 생각이 났는데, 이미 ViT 모델은 normalize를 하지 않은, 데이터의 밝기의 편차 등을 조금이라도(엄청나게는 아니지만) 사용하기 때문에 이를 방해하는 Data를 넣어버린 게 아닌가 싶다.

# Assignment 3

# Assignment Advanced

# Weekly mission & pair review
