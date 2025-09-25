# Visualization

딥러닝 모델의 경우, 중간 과정을 설명하기 어렵다는 문제가 있다(Black box).

모델이 잘 돌아가더라도, "왜 잘 돌아가는가?"에 대한 설명이 부실했던 문제가 있었다.

이러한 중간 과정과 어떻게 모델이 결과를 뽑아내는지에 대한 이해가 없다면, 모델을 발전시키거나 모델의 문제점을 찾기 힘들어진다.

## Embedding feature analysis

딥러닝 모델을 거치며, 레이어들은 low-level부터 high-level feature를 추출하게 된다.

### High-level feature

#### NN in feature space

모델을 통과한 이미지의 마지막 단의 feature vector를 보고, (미리 계산해 놓은) 다른 이미지들의 feature vector와 비슷한 image를 찾는다.

즉, 모델이 어떠한 이미지들을 "유사한 이미지"라고 판단하는지를 볼 수 있다.

다만, 다른 이미지들의 vector와 비교하기 때문에 미리 이미지들을 넣어서 DB를 만들어 놓아야 한다.

#### Dimensionality reduction

위의 방법을 쓰는 이유는, feature들이 너무 고차원이기 때문에 visualize하기 어렵기 때문이다.

따라서, 이러한 feature들을 2차원, 혹은 3차원으로 차원 축소해서 시각화하는 방식이 있다.

벡터들의 유사도를 이용해 cluster하듯이 데이터를 표현하는 T-sne(t-distributed stochastic neighbor embedding)이 있다.

<img width="500" height="483" alt="image" src="https://github.com/user-attachments/assets/e45d2e28-d7d1-48d3-b620-fdf70673d67c" />
https://en.wikipedia.org/wiki/T-distributed_stochastic_neighbor_embedding

MNIST dataset에 대한 T-SNE embedding이다.

### Mideum-high level feature - Layer activation

중간 레이어의 channel을 하나 정하고, 이미지들을 넣어 그 channel이 image의 어떤 부분을 activation하는지 crop해서 확인한다.

어떤 이미지가 무엇을 찾아내는지를 Image segmantation을 통해 기계적으로 찾아낼 수도 있다.

미리 Image segmentation을 해서 구역을 나눠서 라벨링을 해 둔 뒤,

각 채널별로 activation map에 대해 thresholding을 해서 mask를 구한다. 

mask와 labeling된 segmantation을 비교서 어떤 feature를 추출하는지 확인해 보면..

<img width="638" height="206" alt="image" src="https://github.com/user-attachments/assets/0a5f137f-16e5-4349-ace2-1270409a5410" />

이렇게, 중간 layer의 어떤 채널은 머리를 찾는다는 것을 알 수 있다.

### output:  Class visualization - Gradient ascent

$$ I^* = argmax_If(I) - Reg(I) $$

가 되는 image $I^*$ 를 찾는다.

입력 이미지를 아웃풋의 특정 클래스가 되도록 학습시킨다. 즉, 입력을 학습시킨다. $Reg(I)$ 는 변화를 줄이기 위해, L2 규제를 적용한다.

"모델"을 학습하는 것이 아니라, "이미지"를 학습하는 것이기 때문에 모델의 weight는 건드리지 않는다.

Backpropagation을 하는데, 여기서는 minimize 문제가 아니라 maximize 문제이기 대문에 Gradient acent를 사용한다.

이를 반복하면 "모델이 가장 라벨에 알맞다고 생각하는" 이미지를 얻을 수 있다. (+regulation 때문에 조금 흐릿할 수도)

### 영역도출

어떠한 부분이 모델의 결정에 중요한 역할을 했는지 찾아내는 visualization

#### CAM: Class activation mapping

원래 Image의 어떤 영역이 결과에 영향을 많이 주었는지를 이미지화한다. heatmap 형식으로 표시한다.

<img width="523" height="357" alt="image" src="https://github.com/user-attachments/assets/d9ab4133-d7dd-4e5c-9e3a-9eb61646bc97" />

CNN 모델을 통과하고, 마지막에는 선형 레이어를 통과하게 된다.

이를 식으로 나타내면, Global average pooling feawture, GAP을 통과한 faeture $F_k$ 에 대해 

$$ S_c = \sum_k w^{c}_{k}F_k $$

이를 GAP 이전 데이터로 나타내면

$$ \sum_k w^{c}_{k} \sum_{(x,y)}f_k(x,y) $$

순서를 바꾸면

$$  \sum_{(x,y)} \sum_k w^{c}_{k} f_k(x,y) $$

여기서 x,y에 따라 결정되는 오른쪽 $sum_k w^{c}_{k} f_k(x,y) 를  CAM_c(x,y)$ 로 하여 heatmap을 만든다.

명확한 결과를 준다는 장점이 있지만,

마지막 레이어를 통과하기 전에 flatten 대신 GAP이 필요하기 때문에 모델 전체의 재학습이 필요하다.

ResNet, GoogLeNet의 경우는 이미 AveragePooling이 FC 전에 있기 때문에, 이를 GAP로 해석하면 추가 학습이 필요없다.

#### Grad-CAM

CAM과 다르게, 추가 학습이 필요없게 하는 방식이다.

<img width="1327" height="466" alt="image" src="https://github.com/user-attachments/assets/64aa5812-781d-46b5-a4cf-74193ba93908" />

https://arxiv.org/pdf/1610.02391

$$ \alpha^{c}_{k} = \frac{1}{Z}\sum_i \sum_j \frac{\partial y^c}{\partial A^{k}_{ij}} $$

를 통해 특정 클래스 c에 대한 각 channel k에 곱할 값들을 구하고,

$$ L^{c}_{Grad-CAM} = ReLU(\sum_k \alpha^{c}_{j}A^{j}) $$

로 선형 결합을 한 뒤, ReLU로 음의 영향력을 무시하게 한다. Target에서 Conv layer가 나올때까지 Backpropagate. (모델의 재학습이 없다. 한번만 Back해서 확인하는 것)

그렇다면 "특정 클래스"로 판단하는 데에 어떤 부분이 영향을 끼쳤는지 파악할 수 있다.

위의 그림에도 나와 있듯이, Gradient를 Feature MAps까지만 전달하면 되기 때문에 어떠한 task에도 적용 가능하다는 것도 장점이다.

#### ViT: Self-attention layer visualization

CLS 토큰(우리가 MLP를 통과해서 결과를 뽑아낼)에 대한 마지막 self-attention의 self-attention map을 통해 visualize한다.

게다가, ViT에서는 multi-head attention을 하기 때문에 head마다 다른 의미를 가지고 있는 영역을 보는 것을 가능하게 한다!

<img width="571" height="564" alt="image" src="https://github.com/user-attachments/assets/17932c20-c78d-4f54-a43a-e9b253cfc1f5" />

https://arxiv.org/pdf/2104.14294

head마다 다른 영역을 보는 것을 확인할 수 있다. 어떤 놈은 당근, 어떤 녀석은 칼..

하지만, gradcam과는 다르게 이 경우에는 특정 클래스를 targeting할 수 없다. attention matrix에서 판단하기 때문..

### GAN dissection

<img width="768" height="383" alt="image" src="https://github.com/user-attachments/assets/1e90eb1c-4b23-4fd0-a519-2d788cfde2d8" />
https://arxiv.org/pdf/1811.10597

이러한 "해석 가능한" feature들을 이용하여, GAN과 같은 생성 모델에서 특정 부분을 수정할 수 있다.
