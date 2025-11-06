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

$\delta_x_t logp(x_t|y)$ 를 

$\gamma \delta_x_t logp_{\theta}(x_t|y) + (1-\gamma) \delta_x_t logp(x_t)$ 로 분해한다.

왼쪽은 conditional score, 오른쪽은 unconditional score(class의 영향이 없는)으로 분해하여  class에 대한 가중치 $w$ 를 이용해 $\gamma$ = 1+w$ 로 정의하여 넣어준다.

## LDM

Diffusion 학습 시, Image 자체에 noise를 추가하고 복원하는 것이 아니라

encoder를 통해 추출된 저차원의 latent vector를 noise 추가하고 복원한다. 여기서도, Classifier-free guidance 방식을 통해 이미지 생성에 condition 반영 가능.(latent to latent로)
