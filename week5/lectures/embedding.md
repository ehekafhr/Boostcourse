# Tokenization

자연어가 들어왔을 때, 이를 Token 단위로 분리하는 방법

미리 Vocablulary(단어 사전)을 만들어 놓고, 그 사전에 대응되는 token으로 입력을 처리해야 한다.

그렇지 않으면, <UNK> 등으로 토큰을 처리해야 함. (OOV)

## Word-level Tokenization

단순히, 띄어쓰기를 기준으로 단어를 분리한다.

Vocabulary size가 커지는 문제가 있다.

## Character-level tokenization

모든 철자를 쓴다. 
Vocabulary size는 굉장히 작다.

하지만, token에 어떠한 의미를 부여하기 어렵다.

## Subword-level Tokenization

Word-level과 Character-level의 사이로, "Subword"-의미를 가지는 작은 단위로 단어들을 나눈다.

Subword의 범위는 Tokenization 방법론에 따라 결정. 

Character-level tokeinzation은 이미 Vocabulary에 들어가 있기 때문에 OOV 문제는 발생하지 않는다.

### BPE

Iteration을 돈다. 이미 있는 vocabulary의 sub-word 쌍에서, "가장 빈도가 높은" 쌍을 vocabulary에 추가한다.

Vocabulary의 크기가 충분히 커질 때까지 할 수도 있고, 빈도에 따라 충분히 빈도가 낮으면 voccabulary에 추가하지 않을 수도.

### WordPiece

BPE와 마찬가지로 병합 방식을 사용한다. 다만, 빈도를 통해 쌍을 선택할 경우 많이 나오는 단어들이 그냥 여러 가지로 합쳐질 수 있기 때문에, (un, able이 합쳐질 필요는 없다) 각각의 빈도를 나누어 준다.

$$ score = \frac{fraq of pair}{freq of first element \times freq of second element} $$

분자만 보는 것이 BPE, 분모도 고려하는 것이 WordPiece이다.

### [SentencePiece](https://github.com/google/sentencepiece)

여러 가지 방법론을 이용하여, subword를 구분한다.

# Embedding

이렇게 tokenization을 한 word들을 특정 dimension으로 embedding해 주어, input과 output으로 활용하게 된다.

## One-hot encoding

하나의 차원이 하나의 단어를 의미하는 존재하는 단어 갯수 길이의 vector에 대해, 해당하는 단어가 1이고 나머지가 0인 벡터로 임베딩한다.

Sparse representation이기 때문에, index만으로도 단어를 표현할 수 있어 메모리가 좋지만, 의미론적 유사성을 나타낼 수 없음

## Word2Vec

Word를 Dense Vector로 표현한다. 주변 단어의 정보들을 이용하여 단어 벡터를 표현하며, 

Skip-gram과 CBoW 방법론을 사용할 수 있다.

Word2Vec은 One hot-vector에서 One hot-vector 레이블을 맞추는 선형 모델이다.

이때, 두 개의 weight matrix가 있어 딥러닝 모델과 유사하다고 생각할 수 있지만, 단순히 히든 레이어를 만들고 연관성을 학습시키기 위한 것으로 Activation function은 없다.

여기서 앞쪽의 Weight를 Word embedding으로 사용한다. 앞쪽 Weight의 dimension은 (vocab_size,dim) 이기 때문에, 각각의 vocab을 dim으로 보내 주는 하나의 vector가 존재하게 된다. 이를 embedding vector로 사용하고, 이 vector는 두번째 Weight의 각각의 vector와의 유사성을 학습한다. A와 B가 비슷하고, C와 B가 비슷하다면 A와 C도 비슷하다고 유추하는 식으로, 비슷한 embedding vector는 높은 유사도를 갖는다.

게다가, 단어 벡터는 단어들 간의 관계를 학습한다. 예를 들어, vec[queen]-vec[king] 은 "남자-여자" vector를 의미해, vec[women]-vec[man]과 비슷한 결과가 나온다.

즉, 이를 통해 단어 간 연산을 통해 다른 단어를 sementic하게 추론해내는 것도 가하다.
