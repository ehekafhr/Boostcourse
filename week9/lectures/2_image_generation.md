# GAN

<img width="500" height="584" alt="image" src="https://github.com/user-attachments/assets/80c4cb4a-507a-4923-9f69-a6d539cc4a19" />
https://en.wikipedia.org/wiki/Generative_adversarial_network


판별자 Discriminator, 생성자 Generator가 서로 적대적으로 학습하는 모델 구조.

Discriminator는 생성자가 만든 image인지, 원본 image인지를 판별하고 생성자는 원본이라고 속일 image를 만든다.

## cGAN

GAN 학습 시에 조건을 주입하여(Generator, Discriminator), 주어진 조건에 따라 이미지를 생성하고 그 조건 하에서 Discriminate를 하는 모델이다.

## Pix2Pix

Image to Image로, input을 Image로 받는 모델이다. 이 경우에는 Input과 Ouptut의 paired Image가 필요하다는 문제가 있더.

## CycleGAN

Pix2Pix의 paired Image를 해결하기 위해, 두 개의 GAN 모델을 사용한다.

두 개의 GAN 모델의 generator은 서로 역함수 역할을 하게 학습되어, 두 모델의 각각의 adversarial training loss 외에도 두 모델을 전부 통과시켰을 때 원래의 image로 돌아오는지에 대한 cycle loss를 추가한다.

## StarGAN

CycleGAN의 확장판으로, 두 개의 도메인 뿐만 아니라 여러 개의 도메인을 사용할 수 있도록 가운데에 중앙 도메인을 둔다. (다른 도메인으로 이동할 떄는 중앙 도메인을 거친다) -> 도메인을 판단하는 loss도 추가된다.

## ProgressiveGAN
<img width="740" height="466" alt="image" src="https://github.com/user-attachments/assets/a0b61f9e-e2e3-4560-aa09-fb3985904e44" />
https://arxiv.org/abs/1710.10196

고해상도 Image를 생성하기 위해서는 많은 비용이 발생한다. 그렇기 때문에, 고해상도 Image를 바로 만드는 모델이 아니라

저해상도 Image를 만드는 모델에서 점점 layer를 쌓아가는 모델이다.

## StyleGAN

ProgressiveGAN 구조에서, Latent $z$ 를 바로 사용하는 것이 아니라 MLP $f$ 를 통과시킨 $W$ 를 입력으로 사용한다.

style $y$ 은 convolution 후 noise 삽입 후, 

$$ AdaIN(x_i,y) = y_{s,i} \frac{x_i -\mu (x_i)}{\sigma (x_i)} + y_{b,i} $$

로 모델에 넣어 준다. 이를 통해, 저해상도부터 고해상도까지 모두 style을 반영하게 되어(ProgressiveGAN 구조이므로) 원하는 정도에 따른 style 변환이 간편하다.

# AE

Encoder와 Decoder로 구성되어, Encoder가 만든 $z$ 를 원래 이미지로 Decoder가 복원하도록 만든 모델 구조. 두 image의 차이를 Loss로 한다.

## VAE
Encoder에서 $z$ 를 바로 만드는 것이 아닌, $z$ 분포의 $\mu$ 와 $\sigma$ 를 만들어서, 

gaussian Noise $\epsilon$ 을 $N(0,I)$ 에서 추출하여 $\epsilon * \sigma + \mu$ 를 $z$ 로 사용하는 모델이다.

잠재 공간에 대한 분포를 학습에 반영하기 위해서, 원하는 잠재 분포와의 KL Divergence를 추가적으로 Loss로 사용한다.

### VQ-VAE

연속적 잠재 공간이 아니라, 이산적 잠재 공간을 가정한다.

Encoder가 만든 $z_e$ 를 Codebook을 통해 이산화하여 $z_q$ 로 보내어 Decoder로 보낸다. 이미지뿐만 아니라, 텍스트, 음성과 같은 데이터에 적합하다.

# Diffusion

## DDPM

입력 이미지를 forward process를 통해 noise를 넣어 주며 잠재 공간으로 변환하고, 이를 역변환하는 모델을 학습하는 구조.

Forward에서는 가우시안 노이즈를 점점 넣어주어 최종적으로는 random noise가 되게 하고,

Reverse process는 이 노이즈를 추정하여 제거하는 모델을 학습한다.

## DDIM

DDPM의 경우에는 노이즈를 넣고 빼는 많은 step 수로 인해, 이미지 하나를 생성하는데에 많은 시간이 소요된다.

그렇기 때문에, 복원 시에 Markov chain을 건너뛰어 중간 step을 건너뛰면서, 일부에만 reverse process를 적용한다.

속도가 엄청나게 빠르다는 장점이 있고, 성능이 그렇게 떨어지지 않는다. 

## CFG

데이터의 품질(fidelity)를 보이기 위해, Classifier Guidance를 활용한다.

### Classifier Guidance

noise -> data로 가는 과정에서, score function $\delta_x logp_t(x)$ 를 $\delta_x logp_t(x|y)$ 로, class y를 조건부로 반영하도록 주입한다.

이를 별도로 추가하고, 모든 step에 classifier를 넣어 주려면 굉장히 힘들기 때문에..

$\nabla_{x_t} logp(x_t|y)$ 를 

$\gamma \nabla_{x_t} logp_{\theta}(x_t|y) + (1-\gamma) \nabla_{x_t} logp(x_t)$ 로 분해한다.

왼쪽은 conditional score, 오른쪽은 unconditional score(class의 영향이 없는)으로 분해하여  class에 대한 가중치 $w$ 를 이용해 $\gamma$ = 1+w$ 로 정의하여 넣어준다.

## LDM

Diffusion 학습 시, Image 자체에 noise를 추가하고 복원하는 것이 아니라

encoder를 통해 추출된 저차원의 latent vector를 noise 추가하고 복원한다. 여기서도, Classifier-free guidance 방식을 통해 이미지 생성에 condition 반영 가능.(latent to latent로)


# Stable Diffusion

2022년 8월, Stability AI에서 발표한 Open-Source Text to Image 모델.

LDM에서 일부 구조가 개선된 모델로, 대량의 이미지와 텍스트 쌍으로 학습되었다.

LDM과 같이, Latent를 활용하는 Autoencoder 구조이고, U-net 구조를 통해 Noise를 Predict한다.(U-net의 Block은 Attention)

또한, Noise Scheduler를 통해 얼마나 Noise한 latent를 만들지를 결정한다.

위의 Noise Predict하는 U-net에, U-net Block의 Attention은 Text Encoder의 결과물인 Token embedding과의 Cross-Attention이다(Token embedding이 Key, Value)

Text embedding을 하기 위해서는, CLIP Text Encoder를 사용.

LDM에서는 BERT를 사용하였지만, Stable Diffusion에서는 CLIP Text Encoder로 발전된 Encoder를 사용한다.

더 큰 언어모델(CLIP)을 사용하기에, 이미지 품질이 더 좋아졌다. 후속 SD2에서는 OpenCLIP(파라미터 수가 2배이다!)을 사용해 모델을 향상시켰다.

## SD training

큰 크기의 모델, 많은 데이터를 사용하기에 많은 연산량을 요한다.

각각의 Image와 Text를 Latent로 변환(Image Encoder, Text Encoder)한 후, 여러 Timestep에 따라 Noise를 추가하고 이를 예측하는 것을 통해 모델을 학습한다.

## Inference

Token embedding과 Gaussian noise를 가지고, U-Net에 (이전의 output 혹은 초기 Gaussian noise)와 Token embedding을 step별로 넣어 주어 깨끗한 Latent를 만든 뒤, Decoder에 넣어 준다.

Inpainting의 경우에는 Input image의 latent에 noise를 추가한 것으로부터 inference 과정을 시작한다. Input image의 영향을 높게 가져가려면 noise step을 적게, 반대의 경우 noise step을 크게 가져간다.


# After SD

## SD2

생성 이미지 해상도를 높이고, Text Encoder를 더 큰 모델로 바꾸었다.

Super-resolution Upscaler Diffusion Model을 pipeline에 넣어 주어, 고해상도 upscaling도 가능하게 하였다.

이미지의 Depth map 또한 예측 가능하다.

## SDXL

2023년 7월 26일 공개된 이미지 생성 모델로, 높은 색정확도, 정확한 색 대비 등이 가능하다.
기본 이미지 해상도가 1024 * 1024로 가장 높고, Base 모델과 Refiner 모델로 이루어지는 것이 특징이다.

Base 모델을 통과한 Latent (128*128)을 Refiner에 prompt와 함께 넣어 주는 형식이다.

Text Encoder가 따라서 2개가 되고, paramter 수도 늘리고 여러 비율의 이미지를 생성할 수 있게 하였다(기존의 모델들은 정방형)

# Evaluation

## Inception Score

Fidelity: 생성된 이미지가 특정 class를 높게 표현하는가? 생성된 이미지가 특정 클래스에 속할 확률(cross-entropy loss를 생각하면 쉽다!)

Diversity: 다양한 class가 생성되고 있는지. 여러 생성된 image에 대해, 생성된 image들의 예측 클래스의 합이 균일해야 한다.

Inception Score은 marginal 분포(Uniform)과의 KL Divergence를 통해 산출하고, 서로 다를수록 KL Divergence가 높다. 즉, 하나의 Label로만 inference되면(Fidelity에서) 제일 좋다는 것이다.

## FID score

원본 이미지와, 생성 image의 벡터간의 거리를(벡터는 pretrained Inception Network를 통해 구한다) cost로 구한다.

거리는 프레쳇 거리(Frechet distance)로, 평균과 표준편차의 제곱합의 합으로 나타낸다.

## CLIP Score

Image와 Caption 사이의 상관관계를 평가하는 지표이다.

CLIP을 사용한다.

CLIP(Image)와 CLIP(Caption)을 한 embedding 사이의 cosine 유사도를 성능 지표로, 높을수록 좋은 지표이다.
