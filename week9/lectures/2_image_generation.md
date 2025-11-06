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

# Diffusion
