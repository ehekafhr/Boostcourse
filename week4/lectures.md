# Computer Vision이란?

시각은 인간의 감각 정보의 대부분을 차지한다. 이러한 시각 정보를 "처리"하기 위해 절반 가량의 두뇌 자원이 사용된다.

*사실, 우리의 눈이 너무 구려서 인공지능이 pre-processing하듯이 떨림과 혈관을 제거해 주어야 하는데, 여기서 많은 자원이 사용된다! 인공 눈이 만들어진다면 우리의 뇌가 조금 더 자유로워지지 않을까.

이러한 시각 인지와 관련된 것들을 컴퓨터로 처리하는 것을 Computer Vision이라고 한다. 이를 위해, 단순히 컴퓨터에 대한 이해 뿐만 아니라 "인간의 시각 인지"에 대한 이해도 동반되어야 한다.

하지만 기준이 되어야 할 인간의 시각 인지부터가 완벽하지 않기 때문에, 컴퓨터 비전은 난해한 문제가 된다.

최근에는 이미지 분석뿐만 아니라, 거꾸로 이미지를 만드는 모델들도 나오고 있다. 

과거에는 정보를 통해 이미지를 만드는 Computer Graphics, Rendering에 반대되는 개념으로 이미지를 통해 정보를 추론하는 것을 Computer Vision, Inverse rendering이라고 했다.

현재는 Inverse rendering 뿐만 아니라 Rendering과 Img 2 Img process 또한 Computer Vision의 한 분야로 취급한다.

## 이미지 생성

컴퓨터는 카메라와 같은 도구를 통해 3D 세상을 2D로 project한다. (물론 인간의 눈과, 최근 카메라들은 여러 2D image를 통해 세상을 3D로 인식하는 듯 하다) 이렇게 투영된 이미지를 디지털화하여 우리가 사용하는 디지털 이미지가 완성된다. 디지털화 과정은 카메라 회사마다 다르다..

이미지 데이터는 위치, 값으로 이루어진다. Grayscale의 경우에는 하나의 채널만 사용하지만,

일반적으로 사용하는 우리의 RGB 이미지는 3개의 채널을 사용하고, 채도, 명도 등을 활용하는 다른 기법을 사용하는 이미지는 채널 구조가 다를 수 있다.



## CV 강의 개요

이번 주 강의는 이러한 Computer Vision의 발전사와, 데이터를 늘려 학습을 돕는 Data augmentation과 경량화와 self-supervised learning을 위한 knowledge distillation과 cv task를 다룬다.

### CV Tasks

#### Image segmentation

Object dection: 물체 인식 - 네모박스!

Segmentation: 물체 인식 - 픽셀 단위로 labeling

Instance segmentation : 서로 다른 인스턴스들을 찾아냄.

Semantic segmentation : "같은 클래스"의 인스턴스들을 묶음.

-> Panoptic segmentation : 모든 인스턴스들을 찾아내고, 그 클래스까지 분류한다!

#### Computation photography: 떨림 제거, 채도 조정, 화질 개선 등등..

#### 3D understanding: "이미지" to 3D

#### Generative model: 생성

Text - to - texture: 텍스쳐 입히기

3D human: 사람 모델링

# CV의 발전사

## CNN

linear layer의 경우.. 모든 픽셀이 레이어를 통과해야 해서 엄청나게 많은 연산량을 필요로 한다!

게다가, 이러한 linear layer는 하나의 이미지 맵을 형성하게 되는데, 그렇다는 것은 이미지가 돌아가거나, 옆으로 이동하거나 하는 등의 문제에 robust하지 못하다는 문제도 갖고 있다.

그렇기 때문에, kernel을 layer로 사용해 convolution(사실은 correlation이다!) 연산을 해 feature map을 추출하는 모델을 CNN이라고 한다.

물론, CNN layer의 끝에는 똑같이 NN을 단다.(classification을 위해!)

이러한 CNN의 발전은, 더 깊어지고, 더 커지는 방향으로 이루어졌다. 

AlexNet의 경우.. 1.2million이라는 거대한 데이터를 활용하여, 6천만 개의 파라미터를 가진다!

발전 과정에서, cnn뿐만 아니라 다른 곳에서 적용된 방법으로 dropout, ReLU 함수 등이 사용되게 되었다.

### Receptive Field

한 픽셀이 담고 있는, 이전 이미지의 범위이다. 즉, 한 픽셀의 표현력을 의미한다. $(P+K-1) \times (P+K-1)$ 로 계산된다. P는 pooling layer(pixel P개를 합친다, max 등등의 방식으로), K는 kernel(convolution으로 곱해지는 layer)의 크기이다. 

### VGGNet

엄청 큰 모델이다! local response normalization을 지우고(특정 채널의 값이 커지는 것을 방지하는 normalization), 
3 $\times$ 3 convolution filters block과 2 $\times$ 2 max pooling layer를 깊게 쌓았다.

아직도 쓴다. 그 자체를 모델로 쓰지는 않고.. perceptual loss를 계산하는 곳에.

### Residual - ResNet

deeper하게 쌓으면.. Vanishing/exploding gradient 문제가 발생한다. 이를 leaky ReLU 등으로 어느 정도 해결해도,

"Overfitting"하게 되는 문제가 발생하게 되는 것으로 예전 사람들이 착각했다.

<img width="1526" height="573" alt="image" src="https://github.com/user-attachments/assets/3cc2e385-6c2c-4ea9-8206-3ad8f347e57a" />

하지만 이 그림에서 알 수 있듯이, 깊은 레이어가 overfitting된다기보단 그냥 못한다! overfitting되려면 training에서는 훨씬 잘했어야 한다.

그래서 이 문제가 Overfitting이 아니라 degradation 문제라는 것을 알게 되었다.

F(x) 대신 F(x) + x 를 통해, 레이어를 통과하지 않는 값들을 레이어를 통과한 값에 더해 준다.

그렇게 하면, backpropagation에서 "함수를 통과하지 않은, identity 값"이 backpropagate된다! 이를 통해 vanishing gradient 문제도 해결하였다.

이런 방식을 사용한 것이 ResNet이다.

## Transformer

이전 강의에서 다루었지만, Computer vision에서도 이 모델을 쓴다!

### 기원

RNN에서, Long-term dependency와 vanishing gradient 문제를 해결하기 위해 나온 모델이다.

Self-Attention을 통해 연관성을 학습하고, 이 값을 디코더에 넣어 결과를 뱉는 모델이다.

<img width="496" height="749" alt="image" src="https://github.com/user-attachments/assets/ce5d31db-922f-4356-9dbd-132dbd2df5f7" />

여기에 Positional encoding을 통해 (sin, cos를 이용) 순서 정보까지 넣어 주는 모델이다.

### ViT

이 모델이, 놀랍게도 사이즈가 크기만 하면 Computer Vision에서도 잘 먹힌다!

<img width="1024" height="539" alt="image" src="https://github.com/user-attachments/assets/01a3f803-aba1-47c2-8eca-2ef747f1788c" />

먼저 이미지들을 Patch로 자르고, positional "1D embedding"을 붙혀 주어 Transformer에 넣어 준다. (맨 앞에는 cls 토큰, output을 내뱉기 위함을 붙여 준다)
Transformer의 Encoder 뒤에, cls 토큰 전용 MLP head를 달아서 Class로 분류하게 된다.

이전 강의에서 다루었듯이, "큰 파라미터,  데이터"에 대해서는 CNN보다 잘 동작한다.

#### Swin Transformer

<img width="822" height="528" alt="image" src="https://github.com/user-attachments/assets/7d04f976-6232-4548-a02d-ab7446ec7200" />

작은 블록(local window) 내에서만 Attention을 하는 식으로, 효율적으로 진행한다(계층적으로, 작은 이미지 패치에 대해 ViT를 적용하고 더 큰 것에 적용하는 식으로 반복한다.

Self-Attention의 연산량이 줄어들기 때문에, 총 연산량을 $O(n^2)$ 에서 $O(n)$ 으로 획기적으로 줄인다.



<img width="864" height="417" alt="image" src="https://github.com/user-attachments/assets/14b1467d-52b1-4c11-bf7e-1860c4dbe80e" />

Window를 자르는 형태마다 다른 결과가 나올 수 있기 때문에, 여러 가지 결과를 얻기 위 레이어마다 window를 x,y 둘다 이동시켜 준다.

### MAE : Masked Autoencoders

Object: Masking Token들 만으로 빈 Token 부분 채우기.

원본 이미지에서 75% 정도를 Masking하고, encoder - decoder 구조를 거쳐서 복구시키는 것을 학습시킨다. (self-supervised)

Transformer는 Input data를 Masking하면 연산을 하지 않아(CNN의 경우에는 레이어 형태를 유지하기 때문에 연산이 줄어들지 않는다) Masking을 하게 되면 빠르게 연산이 가능하다는 장점이 추가되었다.

Fine-tuning에서도 좋은 성능이 나온다.

### DINO

<img width="657" height="605" alt="image" src="https://github.com/user-attachments/assets/3c5568ea-f856-4287-884d-b7015c247aa7" />


Studnet - Teacher 구조로 이루어진다.

Input $x$ 를 $x_1$ , $x_2$ 로 input을 나누어 student와 teacher가 output을 뽑고, teacher network(학습하지 않음)의 output을 따라가도록 student를 학습시킨다.

그 후, exponential moving average를 통해 teacher를 학습시킨다. 

#### Moving Average - Exponential Moving Average
https://www.investopedia.com/terms/m/movingaverage.asp 

어떤 hyperparameter l에 대해, teacher의 paramer을 (1-l), student의 parameter를 l만큼 moving-average로 계산해 준다) - Teacher의 parameter는 이전 student들의 average와 같게 된다.(정확히는 아니다, l에 따라..)

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

를 통해 특정 클래스 c에 대한 각 channel k를 학습시키고,

$$ L^{c}_Grad-CAM = ReLU(\sum_k \alpha^{c}_{j}A^{j}) $$

로 선형 결합을 한 뒤, ReLU로 음의 영향력을 무시하게 한다. 이 과정을 converge할 때까지 반복. (모델의 재학습이 아니다. 최종 단에 적용될 $alpha$ 를 학습하는 것이다)

그렇다면 "특정 클래스"로 판단하는 데에 어떤 부분이 영향을 끼쳤는지 파악할 수 있다.

위의 그림에도 나와 있듯이, Gradient를 Feature MAps까지만 전달하면 되기 때문에 어떠한 task에도 적용 가능하다는 것도 장점이다.

#### ViT: Self-attention layer visualization

CLS 토큰(우리가 MLP를 통과해서 결과를 뽑아낼)에 대한 self-attention map을 통해 visualize한다.

게다가, ViT에서는 multi-head attention을 하기 때문에 head마다 다른 의미를 가지고 있는 영역을 보는 것을 가능하게 한다!

<img width="571" height="564" alt="image" src="https://github.com/user-attachments/assets/17932c20-c78d-4f54-a43a-e9b253cfc1f5" />

https://arxiv.org/pdf/2104.14294

head마다 다른 영역을 보는 것을 확인할 수 있다. 어떤 놈은 당근, 어떤 녀석은 칼..

하지만, gradcam과는 다르게 이 경우에는 특정 클래스를 targeting할 수 없다. attention matrix에서 판단하기 때문..

### GAN dissection

<img width="768" height="383" alt="image" src="https://github.com/user-attachments/assets/1e90eb1c-4b23-4fd0-a519-2d788cfde2d8" />
https://arxiv.org/pdf/1811.10597

이러한 "해석 가능한" feature들을 이용하여, GAN과 같은 생성 모델에서 특정 부분을 수정할 수 있다.

# Augmenatation

데이터셋은 "모든"데이터를 얻을 수 없고, 얻는 과정이 독립추출이기 어렵기 때문에 bias된 데이터를 얻게 되고,

이러한 데이터를 사용하게 되면 실전 데이터에서 robust하지 않게 동작할 수 있다.

예를 들어, 해바라기를 해가 떴을 때만 찍으면 어두운 해바라기는 제대로 판별하기 어려울 수 있다.

따라서, 이러한 gap을 줄이기 위해 데이터를 이리저리 흔들어서 실제 데이터 domain처럼 만들어 주는 것을 Augmentation이라고 한다.

## Technic

Crop: 짜르기

Shear: 기울이기-평행사변

Brightness: 밝기 조

Perspective Trnasformation: 원근 변환. 사진을 사다리꼴처럼 기울인다..

Rotate : 돌리기

flip : 뒤집기

Mixing: 이미지 두 개를 합친다!!! 이렇게 합친 이미지는 라벨도 그만큼 섞어 준다. one-hot label이 아니게 된다.

Augmentation
