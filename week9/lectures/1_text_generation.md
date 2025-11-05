# LLM Pretrained Models

## LLM

LLM(Large Language Model)은 범용적인 Task를 수행할 수 있는 Language Model이다.

사전학습에 사용되는 데이터 수와, 모델의 파라미터 수가 매우 큰 경우를 종합적으로 지칭한다.

이전의 Pretrained LM인 GPT-1/2, BERT 등은 사전학습된 LLM을 Task별로 Fine-tuning을 진행하여 원하는 모델을 구축한다.

하지만, GPT 3/4, LLaMA 등 큰 모델은 한 모델에서 다양한 Task를 수행할 수 있다.

여기에 "조금의" 예시 데이터셋을 만을 넣어 주는 Few-Shot Learning과, 추가적인 학습 없이 새로운 태스크를 수행하게 하는 Zero-shot learning도 가능하다. 

추가적인 Learning을 하는 대신, Prompt를 넣어 주기만 하면 Zero-shot learning이 가능하다. 물론, 모델의 크기가 크다면 Few-shot의 Demonstration으로 성능이 좋아질 수 있다.

### Prompt

이러한 LLM에 입력으로 들어가는 구성을 Prompt라고 한다.

Prompt는 Task에 대한 묘사, Task에 대한 예시(입-출력 쌍), 입력 데이터로 구성된다.

이러한 Prompt를 어떻게 구성하느냐에 따라 모델 성능이 달라질 수 있다.

### 모델 구조

일반적으로 Transformer의 구조를 변형한 모델들을 사용한다.

Transformer 전체를 사용하는 Encoder - Decoder 구조에서는 입력에 대한 이해와 문장 생성을 분리하고, Decoder Only 구조에서는 단일 모델을 사용한다.(Bert 등)

### Corpus

Corpus란 사전학습을 위해 준비해야 할 대량의 텍스트 데이터이다.

블로그, 뉴스, 커뮤니티 등에서 최대한 많은 데이터를 수집한다.

LLM의 크기가 크면 코퍼스 내의 "데이터 자체"를 암기하는 Memorization 현상이 발생한다. 

따라서, 개인정보를 모델이 학습하거나, 욕설이나 혐오하는 글을 모델이 학습하는 것을 방지하기 위한 섬세한 데이터 정제 작업이 필요하다.

## Instruction Tuning

이렇게 학습된 LLM은 존재하지 않는 어휘를 생성하거나, 프롬프트만으로 Zero-few shot learning이 가능한 좋은 성능을 보이지만, 혐오 표현이나 잘못된 조언을 할 수 있는 문제가 있다.

그렇기 때문에, 입력에 대해 안전하면서 도움이 되는 답변을 하도록 Instruction Tuning을 진행해야 한다.

Instruction Tuning은 SFT -> Reward Modeling -> RLHF의 3단계로 구성된다.

### SFT (Supervised FineTuning)

Prompt + Demonstration (입력 + 출력) 쌍을 가지고, LLM에게 입력에 알맞은 답변을 하도록 학습시킨다.

### Reward Modeling

RL을 위해, LLM이 생성한 답변에 대한 Reward를 만든다.

Reward는 사용자에게 얼마나 Helpful한지, 얼마나 Safety한지를 점수로 산출한다.

이렇게, Prompt와 Demonstration (입력 + 출력) 쌍으로 "점수"를 출력하는 모델을 만든다.

### RLHF (Reinforcement Learning with Human Feedback)

LLM 모델을 Agent로 사용하여, 강화학습 알고리즘인 PPO 알고리즘을 차용해 LLM model이 생성한 답변을 Reward Model이 평가하는 방식으로 학습한다.

# Parameter-Efficient Tuning

## 필요성의 대두

Trnasformer 기반의 LLM 모델들은 General-purpose로도 좋은 성능들을 보이기 시작했다. 특히, 데이터 크기와 파라미터 사이즈는 점점 증가하며 성능을 높이고 이다.

이러한 LLM 모델을 사용하는 기존 방식은 Finetuning이다.

Finetune는 전체 layer, 혹은 Output layer, 혹은 Classifier layer만 학습시키는 방식으로 진행된다.

모든 layer를 학습하는 것이 가장 성능이 좋지만, 그만큼 시간도 많이 든다는 문제가 있다.

또다른 방식으로는 in-context learning으로, 위에서 이야기한 방식대로 몇 개의 예시를 들어 주는 식으로 모델을 원하는 task에 적용할 수 있다.

이러한 방식들의 문제는(특히 모든 weight을 update하는 finetuning) , 추가학습 과정에서 과거의 정보를 잊어버리는 forgetting이 문제가 된다.

그렇기 때문에, 일부 파라미터를 효율적으로 fine-tune 하는 PEFT 방법론들이 대두되고 있다.

일반적으로는 아래의 네 가지 방법론을 사용한다.

### Adapter

기존에 이미 학습이 완료된 모델의 각 레이어 뒤에 학습 가능한 FFN을 삽입한다.

FFN은 MHA, FFN 등의 기존 아키텍쳐의 블록을 통과한 값들을 작은 값으로 압축하고, 비선형 변환(ReLU 등)한 뒤 원래 차원으로 복원하는 방식으로 사용한다.

Target task에 대해 최적화되고, 작은 학습 파라미터 만으로도 finetuning에 근접한 성능을 기록하지만, Inference time이 증가한다는 문제가 있다 (Model 자체에 추가적인 feedforward layer들을 쌓는 것이므로)

### Prefix Tuning

Transformer의 각각의 Layer에 learnable한 vector를 input으로 추가로 넣어주는 방식이다.

Task를 의미하는 특정 embedding을 Transformer에 넣어준다고 생각하면 될 듯하다.

이 경우 또한, Inference time이 어느 정도는 증가할 수밖에 없다 (Transformer의 MHA에 참여하는 embedding이 하나 더 생기므로)

### Prompt Tuning

직접적인 자연어 Prompt 대신, learnable한 Prompt(자연어가 아닌, embedding)를 Input에 붙혀 주는 방식.

Task를 의미하는 Prompt를 학습하는 모델이다.

### LoRA

사전 학습된 모델의 파라미터를 고정하고, 학습가능한 rank decomposition 행렬을 삽입한다.

$d_{model}$ 에서 $d_{FFW}$ 로 가는 Layer Block이 있을 때, (NN layer라면) 이 layer의 parameter 개수는 $d_{model} \times d_{FFW}$ 일 것이다.

LoRA의 경우에는 이를 중간에 $r$ 차원으로 보내는 과정을 거쳐, $d_{model}, r$ 차원의 matrix 하나와 $r, d_{FFN}$ 차원의 matrix 하나로, $r$ 에 따라 learnable parameter를 엄청나게 줄여서 빠르게 학습시킬 수 있다. 이를 Low-rank Decomposition이라고 한다.

이후에, LoRA를 일정 비율로(여러가지 LoRA를 합칠 수도 있다) 원래 Layer의 Parameter에 합쳐 주기만 하면 기존 모델과 parameter 갯수가 같은 task에 맞는 새로운 모델을 얻을 수 있다 (그저, 두 matrix를 곱해 주면 차원이 맞는 weight 행렬이 나온다)

# sLLM models

