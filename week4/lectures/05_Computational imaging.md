# Computational imaging

카메라가 사진, 혹은 영상을 찍는 과정은 단순히 빛을 감지하여 RGB 값으로 변환하는 것 뿐만 아니라, Noise를 제고하는 등의 Computation photography가 들어간다.

대부분의 최신 카메라, 휴대폰 카메라 등에는 이러한 기술이 적용되어 있지만, 더 무거운 모델을 통해 원래 이미지로 복원하거나, 이미지를 고해상도로 변환하는 등의 작업을 할 수 있다.

## Training

기본적인 방법은 다음과 같다.

1) 원본 데이터 이미지를 준비한다.

2) 원본 데이터에 "일부러" Degrade(어둡게 하기, noise 추가 등등..)을 추가한다.

3) 이를 복원하는 DNN 구조를 통과시킨다(일반적으로 skip-connection을 이용한 U-Net 구조가 많이 사용된다)

4) 원본 이미지와 비교하고, Loss를 backpropaget한다. 뒤에서 다루겠지만, 일반적인 L2 loss나 L1 loss는 feature를 잘 잡아내지 못하기 때문에 perceptual loss 등을 사용하게 된다.

### Denosing


<img width="1001" height="308" alt="image" src="https://github.com/user-attachments/assets/82ac2915-52d2-447d-9440-51af4af5a5d8" />

https://arxiv.org/pdf/2003.12751

이렇게 사진을 찍는 과정에서, Noise가 발생하게 된다. 이 Noise는 Gaussian noise를 따른다고 가정하고, 위의 방법으로 denosing을 진행하게 된다.

Gaussian noise를 추가하는 것은 아주 간단하기 때문에, 쉽게 적용할 수는 있다..

### Super resolution

저해상도의 사진을 고해상도로 변환하는 과정.

실제 고해상도의 사진을 의도적으로 Down-sampling 후 noise를 추가하여, 이를 다시 복원하는 네트워크를 학습시킨다. 

#### RealSR

위처럼 그냥 이미지 사이즈를 변경한 데이터셋은 실제 카메라의 특성을 잘 반영하지 못하기 때문에, RealSR에서는 "초점이 다른, 같은 사진" 두 개를 High-resolution과 Low-resolution으로 사용한다.

<img width="1355" height="490" alt="image" src="https://github.com/user-attachments/assets/97c208df-1758-4acd-8415-f77bc78e75c0" />


물론, 두 사진이 완전히 같기는 어렵기 때문에 두 사진의 특정 부분을 Align하는 iterative step이 필요하다.


<img width="1733" height="850" alt="image" src="https://github.com/user-attachments/assets/f877798c-c971-434f-8bdd-25dd63637b3b" />

<img width="1548" height="527" alt="image" src="https://github.com/user-attachments/assets/4a319be0-d038-4d62-b7bd-41bb258db951" />

https://arxiv.org/abs/1904.00523#:~:text=In%20this%20paper%2C%20we%20build%20a%20real-world%20super-resolution,progressively%20align%20the%20image%20pairs%20at%20different%20resolutions.

RealSR의 모델은 U-net과 비슷하게, skip connection을 사용하지만 피라미드 구조로 작은 이미지들과 점점 합쳐가며 upsampling하는 구조를 사용했다.

이러한 RewalSR을 사용한  경우, 조금 더 blur 가 없고 "뾰족한" 데이터를 뱉어낼 수 있다.

### Image deblurring

사진을 찍는 동안 카메라가 흔들리거나, 물체가 움직여 발생하는 blur를 제거하는 과정.

기본적으로 그냥 blur를 추가하게 되면, "모든 부분"이 흔들리기 때문에 실제 카메라의 흔들림, 혹은 물체의 움직임과는 거리가 있다.

높은 Fram rate를 가진 카메라를 이용해서, 여러 프레임을 평균낸 image를 blur된 이미지로 사용한다. 

이러한 방식으로 image를 blur하면, 결국 "연속적이지 않은" 사진들의 평균을 내기 때문에 현실적이지 않은 패턴들이 나타나게 된다.

#### RealBlur

RealBlur는 이 문제를 해결하기 위해서 재밌는 발상을 했는데, 빛을 두 방향으로 투과시켜주는 Beam splitter를 통해, shutter speed가 빠른 카메라와 느린 카메라 둘이 똑같은 사진을 찍도록 했다! 그러다면, shutter speed가 느린 쪽의 이미지는 우리가 실제로 얻는 blur 이미지가 된다.

물론, 이 경우에도 두 이미지의 위상이 정확하게 일치하지는 않기 때문에 Align을 하기 위한 작업이 필요하다.

### Video motion magnification

Frame t와, t+1의 Data가 있지만, 움직임을 과장한 Data는 존재하지 않기 때문에,

두 프레임을 CNN에 Magnification factor $\alpha$ 와 함께 넣어 Magnified frame를 넣고, 랜덤한 움직임을 추가한 이미지를 Ground truth로 사용하게 된다.(실존하지 않는 이미지이다)

### General Image restoration

이미지를 복구하거나, blur를 지우는 등의 작업은 "특정 blur level, 특정 자르기 level" 등의 trianig할 때의 이미지 손상 강도에 과적합되는 경향이 있다.

예를 들어, 과하게 손상된 이미지를 잘 복구하는 모델은 오히려 별로 노이즈가 없는 이미지를 망가뜨릴 수 있다.

그렇기 때문에, On-demand learning에서는 이러한 TASK의 난이도에 따라 dataset을 나누고,

각각의 Dataset마다 PSNR(이미지 품질) - $10 * log_{10}(\frac{MAX^2}{MSE})$ 를 측정해 PSNR이 낮은 데이터셋에 대해 더 많이 학습하도록 하여 모든 TASK를 잘 처리하도록 한다.

## Advanced loss functions

<img width="815" height="621" alt="image" src="https://github.com/user-attachments/assets/b876fa5a-7dc0-4b64-ba0b-9e420acc0417" />

https://arxiv.org/pdf/2104.12034

L1 loss와 L2 loss는 사람이 느끼는 이미지의 차이와 다르다. 단순히 "픽셀별로"차이를 측정하기 때문에, 위의 그림처럼 단순히 조금 밝아진 사진과, 모아이 석상이 되어버린 사진이 똑같은 로스를 가지게 된다.

픽셀이 실수로 한 칸씩 오른쪽으로 당겨지기만 해도, blur한 데이터에서는 끔찍한 loss를 만들어 낼 수 있다!

그렇기 때문에, MAE loss, MAE loss를 만들게 되면, 질감 등을 정확히 표현하지 못하고 뭉뚱그려서 표현하게 된다. 

### Adversarial loss

Adversarial loss는 이를 해결하기 위해, GAN 모델을 사용한다. MSE loss 대신, Discriminator model을 만들어서 이 discriminator model이 "잘 판별한 정도"를 loss로 활용한다.

물론, MSE loss도 같이 사용하긴 한다.

이러한 모델의 문제점은 GAN의 문제점과 같이, 어떻게 Generator와 Discriminator의 학습 속도를 맞출 것인가?라는 문제가 있다.

(두 모델 중 하나가 너무 잘해서 Discriminator의 정답률이 1이 되거나, 0.5가 되어 버리면 모델의 학습이 멈추어 버린다!)

### Perceptual loss

Visualization 부분에서도 다루었듯이, CNN의 중간 kernal들은 이미지의 feature들을 추출하는 역할을 한다.

<img width="982" height="371" alt="image" src="https://github.com/user-attachments/assets/eb90a8a6-e94c-4201-9ced-324b28843c19" />

<img width="855" height="479" alt="image" src="https://github.com/user-attachments/assets/5fd4727a-d829-470a-8778-c9513369c0ab" />

[Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/abs/1603.08155)

미리 학습된 VGGnet을 뒤에 달아서, 여기서 추출한 Feature들간의 MSE loss를 대신 사용한다! 논문에서는 relu3.3을 통과한 값들을 비교했다.

원본 이미지인 Content Target 말고도 Style Target이라는 것이 있는데, 여기에 다른 이미지를 넣어서 그 이미지의 "Style"을 따라가는 원본 이미지의 변형 형태를 만들어내도록 할 수 있다!

이때, Style loss를 Gram matrix를 통해(MSE 대신) 계산할 수 있다.

$H\times W\times C$ 형태를갖는 Featuremaps를 이미지에 대해 flatten해서, $C\times (H*W)$ 의 형태로 만든 뒤 Transpose해서 두 matrix를 곱해, $C\times C$ Gram-Matrix를 만든다.

Feature 뿐만 아니라, Feature들의 관계도 포함하는 Gram Matrix를 MLP를 통과해 두 값을 비교하는 방식으로 사용한다.

## Video

비디오에 색을 입히거나, 스타일을 추가하는 과정이다.

### Problem

각각 사진별로 색깔을 추가하거나 하면, 이전 사진과는 다른 색을 입히거나 해서 연속성이 떨어진다. Temporal inconsistency가 발생한다..

### Processing: Learning Blind Video Temporal Consistency

<img width="1083" height="691" alt="image" src="https://github.com/user-attachments/assets/bfe3a420-c293-4775-9b9f-a00a90a87eb1" />

<img width="848" height="372" alt="image" src="https://github.com/user-attachments/assets/7a2564a3-307b-43a2-a888-92be00023801" />

https://arxiv.org/abs/1808.00449#:~:text=In%20this%20paper%2C%20we%20present%20an%20efficient%20end-to-end,as%20inputs%20to%20produce%20a%20temporally%20consistent%20video.

이 모델은 시간 t의 이미지에 대한 원본 이미지와의 perceptual loss 뿐만 아니라, 이전 time의 값들과의 Temporal loss도 측정한다.

t Frame을 생성하는 방법은 다음과 같다.

t-1 frame Output 값 $O_{t-1}$ 그리고 그냥 프로세싱한(이미지 하나에 대해) $P_t$ 를 Network의 입력으로 넣고, Network 중간에 original video(t-1 frame, t frame)를 넣어 Concat해 준 뒤 LSTM을 포함한 U-net 구조로 다음 값을 뱉는다. 이 이미지들은 ground truth와 perceptual loss를 측정하게 된다.

<img width="1098" height="618" alt="image" src="https://github.com/user-attachments/assets/28dfdbae-ae6f-4989-8dcc-6b2484c96516" />

이렇게 생성된 image들 간의 Long-term과 Short-term temporal loss도 측정하게 된다.

Shor-term temporal loss는 앞의 image와의 차이를 측정하는 loss로,

이전 이미지와 현재 이미지의 실제 차이를 사용해 만든 mask (0과 1을 갖는다)

$$ M_{t\Rightarrow t-1}^{(i)} == exp(-\alpha \vert I_t - \hat{I_{t-1} \vert^{2}_{2} $$ 

에 대해,

$$ L_{st} = \sum_{t=2}^{T} \sum_{i=1}^{N} M_{t\Rightarrow t-1}^{(i)} \vert O_{t}^{(i)} - \hat O_{t-1}^{(i)} \vert_1 $$

를 loss로 사용한다.

hat이 붙은 부분은 $I_t$와 $I_{t-1}$ 을 flowNet을 통과시켜 추출한 F layer와 함께 Warping Layer를 통과했다는 뜻이다. 또한, N은 픽셀 수이다.

(왜 t=2부터 시작하는가? 어차피 long-term temporal loss에서 계산될 것이기 때문.)

이때, Long-term temporal loss는 모든 image pair들에 대해 계산하지 않고, "첫번째" 이미지와의 loss만을 계산한다.

$$ L_{lt} = \sum_{t=2}^{T} \sum_{i=1}^{N} M_{t\Rightarrow t-1}^{(i)} \vert O_{1}^{(i)} - \hat O_{t-1}^{(i)} \vert_1 $$


