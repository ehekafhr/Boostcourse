# Recsys 동향

Shallow Model -> Deep Model -> Large-sacale Generative Model

## Deep Model

[<img width="1266" height="400" alt="image" src="https://github.com/user-attachments/assets/0cbe4692-0edb-4536-8e28-ecbcd3fe32ee" />](https://users.cecs.anu.edu.au/~akmenon/papers/autorec/autorec-paper.pdf https://dl.acm.org/doi/10.1145/3132847.3132972  https://www.width.ai/post/neural-collaborative-filtering )

Autoencoder을 기반으로 한 모델들. 

## Large-scale Generative Models

Attention 메커니즘 등을 활용한 목잡한 형태의, 생성형 모델.

prompt 등을 input으로 추가로 받아 생성 가능하다. Chat=Gpt의 Chat-Rec 등이 있다. 또한, diffusion을 활용한 생성형 모델도 있다.

## Explainabililty

"왜" 추천을 했는지 설명을 해줄수 있는 모델. 기존의 Markov chain처럼 "유사한 사람들이 선택해서"를 벗어나서, Feature-level과 Sentence-level, 그리고 "유사한 Visual"을 가지고 설명을 해 줄 수 있다.

## Debiasing ans Causality

일반적인 추론 모델은 "상관관계"를 추적한다. 이때, 보이지 않는 다른 변수에 의해 영향을 받는 상황이 생길 수 있다. 이를 제고하는 것을 Debgias, 이를 통해 상관관계가 아닌 인과관계를 분석하는 것을 Causality라고 한다.

# OverView

## Generative model: Autoencoder / Variational Autoencoder 기반.

## 변분추론(VI)

model $p_\theta(x)$ 과

Data $D = {x_1,...,x_n}$ 을 통해

Maximum likelihood $\theta \leftarrow argmax_{\theta} \frac{1}{N} \sum_i log p_{\theta}(x_i)$ 

를 찾는 과정. "Posterior"를 찾아가는 과정이 된다.

## Markov monte carlo: 확률들을 chain에 걸쳐 곱해 가며 가장 가까운 분포를 찾는다.

## Analysis: 설명 가능성

## Causality and ML 


