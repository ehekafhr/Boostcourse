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

Semantic segmentation : "같은 클래스"의 인스턴스들을 묶음.

Instance segmentation : 클래스의 인스턴스들에 번호까지 매겨, 인스턴스끼리도 분류

-> Panoptic segmentation : 배경까지 찾아냄

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

# Augmenatation

데이터셋은 "모든"데이터를 얻을 수 없고, 얻는 과정이 독립추출이기 어렵기 때문에 bias된 데이터를 얻게 되고,

이러한 데이터를 사용하게 되면 실전 데이터에서 robust하지 않게 동작할 수 있다.

예를 들어, 해바라기를 해가 떴을 때만 찍으면 어두운 해바라기는 제대로 판별하기 어려울 수 있다.

따라서, 이러한 gap을 줄이기 위해 데이터를 이리저리 흔들어서 실제 데이터 domain처럼 만들어 주는 것을 Augmentation이라고 한다.

## Technique - 정적

Crop: 짜르기

Shear: 기울이기-평행사변

Brightness: 밝기 조

Perspective Trnasformation: 원근 변환. 사진을 사다리꼴처럼 기울인다..

Rotate : 돌리기

flip : 뒤집기

Mixing: 이미지 두 개를 합친다!!! 이렇게 합친 이미지는 라벨도 그만큼 섞어 준다. one-hot label이 아니게 된다.

mixup: Beyond Empirical Risk Minimization: https://arxiv.org/abs/1710.09412 

여기서는 모든 pixel을 섞어 준다는 개념이고,

CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features: https://arxiv.org/abs/1905.04899

여기서는 개 하반신, 고양이 상반신처럼 crop해서 붙이는 방식이다.

이러한 단순한 조합들이 생각 외로 Test Accuracy를 높여 준다.

Copy-paste: Segmentation된 부분들을 다른 데이터 위에 그려넣어 데이터를 만들어 준다.

## Learning-based Video Motion Magnification

copy-and paste를 통해 Augmentatiopn된 데이터를 만들고, (Object + Background)

Motion을 추가하기 위해서 단순히 Translation만 사용한다.(Random한 방향으로)

이렇게 "움직임"을 학습하면, 모션의 크기를 크게 하거나 하면서 흔들림, 이동 등을 과장할  수 있다.

# Segmentation & Detection

## Sementic segmentation 

각각의 픽셀을 카테고리로. 어떤 Instance인가?를 따지는 것은(서로 다른 자동차인지..)는 Instance segmentation에서 다룬다.

의료, 자율주행 등 다양한 CV 분야에 사용된다.

### FCN

CNN과 다르게, flatten을 하지 않고 마지막까지 CNN 단을 유지하여.. pixelwise하게 결과를 뽑아낸다.

그런데, 이런 경우 CNN에서 max pooling이나, padding 없는 convolution에 의해 이미지 사이즈가 줄어들기 때문에 "작은" 이미지가 나오게 된다.

하지만, 마지막 단의 레이어는 많은 정보를 담고 있지만, 이미 "합쳐진" 데이터이기 때문에 이것만으로 upsampling하기에는 무리가 있다.

<img width="1636" height="383" alt="image" src="https://github.com/user-attachments/assets/7ce82f10-454c-4bef-914f-5ac97672a303" />
https://arxiv.org/pdf/1411.4038

그래서, 마지막 커널을 통과한 값들 뿌남ㄴ 아니라 다른 값들도 사용한다.

그림에 나와 있는 FCN-32s는 마지막 단만 사용한 것으로, FCN-16s, FCN-8s 등은 그 전 단, 그 전 전 단까지 사용한 결과이다.

너무 앞쪽 단에는 feature extraction이 다 되어 있지 않을 수 있기 때문에 8x가 좋은 결과를 뱉는 것으로 보인다.

### U-net

<img width="1697" height="757" alt="image" src="https://github.com/user-attachments/assets/a3a43331-7b48-45b4-9e39-c20360521dc4" />
[ Ronneberger etal.,U-Net:ConvolutionalNetworksforBiomedical ImageSegmentation,MICCAI2015](https://arxiv.org/abs/1505.04597)

U-net은 FCN과 비슷하게, Contracting path(왼쪽)에서 절반씩 downsizing을 하면서 feature channel의 수를 두배로 늘리고,

*이때, upscaling을 할 때 "2배"의 이미지가 되기 때문에, downsizing을 할 때 홀수인 경우 사이즈가 이상해진다. 따라서, 커널의 사이즈 등을 조절해서 항상 downsizing을 할 때 짝수가 되도록 해 준다.

오른쪽의 Expanding path에서는 반대로 채널을 줄여 가며 이미지를 upscaling한다.

논문을 살펴보면.. 왼쪽과 오른쪽의 이미지 사이즈가 다르다. upscaling할 때 어떻게 그러면 이어붙이냐?고 하면..

그냥 가운데 중심으로 자른다고 한다.얼탱.

## Object Detection

Bounding box를 찾는 과정이다. (+Classification)

보통 박스 하나는 $(p_{class}, x_1,x_2,y_1,y_2)$ 꼴로 나타난다.

### Two-stage(R-CNN)

Bounding box를 먼저 찾고(Selective search를 사용한다. 이미지들로부터, 후보 이미지들을 모두 뽑아내는 Pre-trained된 모델이고, CNN 구조는 아니다)

그 Bounding box들을 동일 사이즈로 warp 한 후(CNN에 넣기 위해) CNN으로Clasification하는 두 단계로 이루어진 아키텍쳐이다.

기본적이 R-cnn은, 인풋 이미지에서 수많은 region 제안을 뽑고, 그 제안들에 대해 모두... CNN을 돌린다.

당연하겠지만, 무척 오래 걸린다.. 오래 걸리고, 데이터도 모자라서 그런지 SVM object classifier를 사용했다.

<img width="1372" height="707" alt="image" src="https://github.com/user-attachments/assets/381017fb-0dae-4583-961d-b095c6e24abb" />
https://lilianweng.github.io/posts/2017-12-31-object-recognition-part-3.html

Fast R-CNN은 "미리" 이미지를 통과시켜 RoI map를 투영할 feature map 만들어 놓고, proposal들을 그 위에 proejct한 후 max pooling을 통해 고정된 크기의 값을 뱉도록 한다. 이렇게 하면, fully connected layer에 넣기 위한 vector의 길이를 맞출 수 있다.

Fast R-CNN은 이러한 RoI pooling을 통해 2000번의 CNN 계산을 한 번으로 줄였다.

Faster R-cnn은 RPN을 통해 Region proposal마저 CNN으로 계산하게 했다. (이전의 Selective search는 굉장히 느린 작업이었다)

### One-stage(YOLO)

RoI pooling을 거치지 않고, 한 번에 classification과 bounding box를 찾는 방식이다.

Yolo는 이미지를 그리드로 나눈 뒤, 각각 Class probability map과 Bounding box(Confidence를 포함한)을 만든 뒤, 두 개를 합쳐 detection을 한다.

그리드 갯수 $S \times S$ , 그리드당 bounding box $B$ , 클래스 갯수 $C$ 에 대해,

$S \ times S \times B \times (C+5)$ 

크기의 output이 나온다! 5는 바운딩 박스의 크기와 confidence이다.

다라서 후처리가 필요하다.. confidence score를 기준으로 하든.. 좀 시간이 걸리고,

이렇게 후처리 끝나고 결정된 값들만 사용된다.

#### Focal loss

사실 대부분의 영역은 의미없는 값들이다. 대부분의 box들은 의미없는 배경만을 잡을 것이고, 중요한 정보가 아닌데도 너무 많은 숫자를 가지고 있어, 학습이 느려지는 문제가 있다.

그래서 크로스 엔트로피 대신, Focal Loss를 사용한다.

크로스 엔트로피에 $(1-p_t)^{\gamma}$ 를 곰한 값이다. ( $\gamma$ 는 hyperparameter)

이렇게 하면, 잘 판별된 결과들에 대한 loss가 굉장히 줄어들어, "확실히 배경이여!" 하는 부분에서 쓸모없이 학습이 진행되지 않게 한다. 

이를 활용한 것이 RetinaNet이다.

<img width="1727" height="537" alt="image" src="https://github.com/user-attachments/assets/9d657bea-07e7-473b-8117-189847ed348f" />
https://arxiv.org/abs/1708.02002

U-net처럼 추출하고, skip connection을 통해 합쳐 주는 것은 같지만, class와 box subnet을 나누어 object detection에 사용될 수 있게 하고, Focal loss를 통해 배경의 영향을 줄인다.

## Instance segmentation: Mask R-CNN

같은 클래스더라도, 다른 인스턴스인 경우 "다른 인스턴스임을 알리는 번호"를 매기는 작업까지 한다.

Mask R-CNN은 Faster R-CNN 구조의 classifier 단에 Mask FCN predictor를 더한 구조이다.

여기서 RoI pooling 대신 RoI align을 사용한다.

RoI pooling에서는 양자화된 feature map을 사용하는데, 여기서 양자화되어 있기 때문에 RoI와 격자의 크기가 맞지 않으면 버려지는 문제가 있다. 이러한 문제는 masking task에서는 치명적일 수 있기 때문에, 양자화를 하지 않고, RoI를 나눈 cell들에 대해 4개의 point를 잡아 Bilinear interpolation을 통해 Cell의 값을 추출한다. (중간의 값을 선형적으로 찾는 방법, 선형회귀라고 생각하면 된다)

여기서 Faster R-CNN에 Mask 대신 3D surface를 회귀시켜 3D 정보를 추출하는 Mesh R-CNN이나 DensePose R-CNN을 만들 수도 있다.

## Transformer-based

### DETR: End-to-End Object DEtection with TRansformers

Yolo 등의 모델에서, 의미없는 bounding box를 지우는 작업은 꽤나 시간이 걸린다. 그러한 문제를 해결하기 위해.. End-to-End 로 object detion을 하는 Transformer model을 DETR이라고 한다.

<img width="1527" height="413" alt="image" src="https://github.com/user-attachments/assets/aa11ea3f-6ca6-4b10-a3e6-672226730aa1" />

기본적으로 이미지의 feature들을 추출하기 위해 CNN을 사용하고, 여기에 positional encoding을 더해 transformer에 넣어 준다.

encoder의 경우는 동일한데, decoder의 경우에는 이전 output을 활용한 auto-regressive한 값이 아니라 학습되는 object queries를 대신 넣어 준다. 이렇게 decoder에서 나온 값들을 FFN을 통과해서, 정답과 예측 값이 1:1이 되도록 하는 bipartite matching을 구해 loss를 계산한다.

이때, 최적의 bipartite matching을 찾기 위해 matching loss와 box loss를 이용한 Hungarian algorithm 등을 사용한다.

### MaskFormer

<img width="1493" height="386" alt="image" src="https://github.com/user-attachments/assets/88df3db9-d51f-4ff5-8499-b80cae8d649d" />

<img width="1423" height="449" alt="image" src="https://github.com/user-attachments/assets/94459b4e-9d8b-4c3e-9803-cf6800265ba4" />

Maskformer에서는 Mask R-CNN처럼, Mask를 따로 학습하여 픽셀 단위 분류 대신 사용한다.

https://arxiv.org/abs/2107.06278

Backbone CNN에서의 feature를 Transformer에 넣고, 이미지 해상도로 decoding하여 다음 step의 계산을 준비한다.

transformer decoder의 결과를(Query가 학습되며 position도 학습한다) MLP를 거쳐서 class prediction과 mask embedding로 나눈다.

이전에 계산한 pixel embedding에 mask embedding을(softmax를 거쳐) dot-product해준 것을 mask prediction으로 사용한다.

N개의 Query가 있기 때문에, class predictions(class 수 +1(아무것도 아님))와 mask prediction은 N개가 존재한다. 이 두 개를 합쳐 예측을 수행하고, loss를 구한다.



### Uni-DVPS

Backbone에서 Transformer Decoder의 output을 Transformer decoder의 key-value로 하고, Depth를 따로 계산하는 MLP를 추가해 주는 모델이다.

Pixel decoder에서 나온 값을  Mask, Depth 계산에 사용할 수 있게 옮겨 주는 Feature gate가 있어, 이를 mask와 Depth에도 사용하는 모델이다.

이때, Query matching을 통해 Object를 추적할 수 있다. 영상에서, 같은 object의 경우 비슷한 query를 갖기 때문에 이를 통해 유사도를 계산해 같은 object인지 확인할 수 있다.

## SAM, Grounded SAM

<img width="500" height="358" alt="image" src="https://github.com/user-attachments/assets/2e66bd73-2d3a-4b57-8173-dda304d08ce3" />

[SegmentAnything,](https://arxiv.org/abs/2304.02643#:~:text=We%20introduce%20the%20Segment%20Anything%20%28SA%29%20project%3A%20a,masks%20on%2011M%20licensed%20and%20privacy%20respecting%20images.)

image에, 우리가 원하는 추가적인 프롬프트(포인트, 텍스트, 마스크, 박스 등등..)을 넣어서 masking을 하는 마법같은 모델이다.


<img width="922" height="1397" alt="image" src="https://github.com/user-attachments/assets/09b62421-6f1d-43e0-b772-8a436c865460" />

<img width="581" height="788" alt="image" src="https://github.com/user-attachments/assets/4ea19de9-b469-46c0-b294-27e4ad761976" />


https://maucher.pages.mi.hdm-stuttgart.de/orbook/deeplearning/SAM.html


ViT Encoder를 사용하고, 임베딩된 데이터에 mask를 추가한 뒤 prompt encoder를 통과시킨 prompt와 mask image를 mask decoder에 넣는데,

prompt를 query로 받고, mask image를 key-value로 하는 token to image attention 뿐만 아니라 image to token을 하는 decoder 두 단을 쌓는다. 

image의 positional embedding과 prompt embedding은 다음 단에도 더해져서 들어간다는 것을 기억하자.

이렇게 image to token쪽을 통과한 값은 upscale와 CNN을 거쳐, token to image가 final attention을 지난 값의 mask token을 MLP를 통과시켜서 product해서 mask를 만들게 되고, iou token은 따로 MLP를 통과시켜 confidence score로 받게 된다.

내부 구조는 position encoding이 계속 더해지고, key, value를 구하는 과정에서 이것저것 더해져서 조금 복잡하다..

이 SAM을 학습시키기 위해 데이터 생성 과정이 재밌는데, 처음에는 조그마한 라벨링된 데이터로 시작해서,

어느 정도 이후에는 모델이 뱉은 label을 사람이 수정해서 조금 더 쉽게 labeling하고ㅗ,

그 다음에는 모델이 "확실하다"고 하는 것들은 그냥 모델의 라벨링을 믿고 나머지를 사람들이 labeling하고,

그 다음에는 완전히 자동적으로 모델이 label을 추가하며 자동적으로 학습하게 했다.

특정 데이터에는 안 좋은 결과가 나왔지만, 대부분의 결과에 좋은 데이터가 나왔다. (게다가, 이것은 특정 데이터로 fine-tuning하지 않은 모델이다!)

Grounded-SAM은 G-DINO, detection을 한 뒤 context를 주는 모델과 결합시켜 Segmentation에 더해 Open-Vocab detection까지 추가한 모델이다.

# Computational imaging

## Training

### Denosing

### Super resolution

### Image deblurring

### Video motion magnification

### Image restoration

## Advanced loss functions

### Adversarial loss

### Perceptual loss

## Video

### Problem

### Processing
