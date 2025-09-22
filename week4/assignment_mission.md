# Assignment 1

Attention mechnism과, 이를 통한 VIT를 사용해보는 실습 과제.

데이터의 Dimension을 고려하여, 컨볼루션을 통해 패치로 나눈다.

처음에는 patch로 나누기를 진행 후 컨볼루션을 진행하기 위해

torch.unfold 함수를 이용해 가로, 세로로 분할 후 컨볼루션 연산을 수행하였지만.. 무언가 오류가 났다.

몇 번의 오류 후, 위에서 이미 Convolution layer에서 stride와 kernel size를 같게 함으로써 분할 역할을 함께 수행하게 정의했다는 것을 깨닫게 되었다. unfold 한 patch들을 위한 convolution layer가 아니었기 때문에 사이즈 오류가 난 것이다.

1D embedding Vector에 대한 오해가 풀렸다. 가로 세로를 고려하지 않고 idx를 받기 때문에, 세로 방향으로는 locality를 무시할 것이라 생각했지만.. Matrix를 보니 vector가 1D일 뿐이고 모든 patch들에 대해 유사도를 학습하기 때문에 가로뿐만 아니라 세로 방향 지역 유사성도 이용한다.

# Assignment 2

# Assignment 3

# Assignment Advanced

# Weekly mission & pair review
