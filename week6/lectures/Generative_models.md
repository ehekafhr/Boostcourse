# 생성모델

Data (x,y) 혹은 (x)에 대해

$$ P_{data}(x) \cong P_{model}(x) $$

$$ P_{datat}(x,y) \cong P_{model}(x,y) $$

를 만족하는 model을 만들자!

모델의 distribution을 찾아내고, 그 구조를 이해하여 generative model을 만드는 것이 목적.

# VAE: Variational AutoEncoders

<img width="693" height="282" alt="image" src="https://github.com/user-attachments/assets/f6d27a6a-b8d3-4f46-b06f-d37d614ac196" />
https://en.wikipedia.org/wiki/Variational_autoencoder

Enbcoder - Dedcoder 구조를 만들어(각각은 보통 NN이다), decoder를 generator로 쓰게 된다.

이때, hidden layer를 $z$ 로 보통 표기한다.

이 hidden layer $z$ 의 확률분포 $p(z)$ 를 $N(0,1)$ , 즉 표준정규분포를 따르도록 하는 것이 목표이기 때문에 KL distance $KL(z\vert p(z))$ ㄹ들 loss로 사용하게 된다.

이때, KL divergence를 위해서는 $z$ 를 확률분포로 바꾸어 주어야 한다.. 그렇기 때문에,

$z$ 를 encoder 마지막 단에서 $\mu$ 와 $\Sigma$ 를 구해 $z ~ N(\mu, \Sigma)$ 로 sampling하게 된다.

<img width="501" height="356" alt="image" src="https://github.com/user-attachments/assets/fa3d5b01-0029-4956-9fa2-83af4b38a822" />
https://en.wikipedia.org/wiki/Variational_autoencoder

이때, 그냥 sampling을 하게 되면 backprop가 불가능하므로  reparameterization trick을 써서
$\epsilon ~ N(0,1)$ 을 sampling한 뒤
$z ~ \mu + \Sigma * \epsilon$ 을 사용하여 backprop이 가능하게 한다.

여러 가지 trick과 moment에 대한 복잡한 계산을 거쳐..

KL divergence 

$$ KL(q\vert p) = \frac{1}{2}(log\frac{|\Sigma_p|}{|\Sigma_q|} - d + (\mu_p - \mu_q)^T \Sigma_{p}^{-1}(\mu_p - mu_q) + tr(\Sigma_{p}^{-1}\Sigma_q)) $$ 

으로 정리된다.

그리고 Reconstructi9on loss는 보통 MSE를 쓰게 된다.
