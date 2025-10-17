# Assignment 1 

기본적인 분포를 확인하는 과제. 딱히 특이점은 없었다.

# Assignment 2

GPT 생성모델을 사용해 보는 문제. 추천을 생성하긴 하는데.. 뭔가 이상하다. GPT-2여서 그런가?

# Assignment 3

Autorec과 VAE 구현 과제.

저기 있는 Minimize하는 KL divergence $\frac{1}{2} \sum_{i=1}^{N}(exp(\sigma_i)-(1+\sigma_i)+{\mu_i}^2)$ 식을 다시 보아야 했다.

loss에서 mean을 안쓰고 sum을 썼는데.. lr이 늘어나는 효과라 더 빠르게 된 것 같지만 MSE의 정의에는 맞지 않는 것 같다.

# Assignment 4

3번 과제와 비슷한 과제. Clustering을 진행했는데, K-means clustering 때문에 cluster가 잘 나오지 않는다고 생각했는데 다른 클러스터링 방법을 사용하거나, k를 바꾸더라도 클러스터링이 잘 되지 않았따.

영화 데이터이기 때문에 사람의 취향을 몇 개로 묶기에는 힘든 부분이 있기 때문에 이런 결과가 나온 것 같다.

# Weekly mission

KL divergence를 분해한다는 생각이 신박했다. 변수 사이의 독립성 부분만 추출하는 것을 보았다.

그러나 $p(n)$ 등의 notation들이 헷갈린다.
