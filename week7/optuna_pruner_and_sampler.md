# Optuna

https://optuna.org/

Optuna는 하이퍼파라미터 튜닝을 위한, 여러 프레임워크와 툴에서 사용할 수 있는 오픈소스 프레임워크.

먼저, 하이퍼파라미터 탐색 범위를 제시하는 함수를 만든다.
```
def hyperparameter_space(trial):
    return{
            "learning_rate": trial.suggest_float("learning_rate", 5e-6, 5e-5, log=True),
            "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [ 64, 128,256]),
            "weight_decay": trial.suggest_float("weight_decay", 0.005, 0.05)
    }
```
이렇게 `dict` 형태로 범주형은 `trial.suggest_categorial`, 연속형은 `trial.suggest_float` 로 하이퍼파라미터 탐색 범위를 정해 준다.

`transformers.Trainer` 또한 Optuna를 통한 searching을 지원한다.

```
best_trial = trainer.hyperparameter_search(
    direction="maximize",
    backend="optuna",
    n_trials=5,
    hp_space=hyperparameter_space,
    compute_objective=lambda metrics: trainer.state.best_metric if trainer.state.best_metric is not None else metrics["eval_accuracy"]
)
```
으로 `backend = "optuna"` 로 지정하면 optuna framework를 사용할 수 있다.
 `optuna.study.create_study`, `optuna.study.Study.optimize` 에 사용되는 인자를 넣을 수 있다.
 `optina.study.create_study`는 metric을 최대화할지 최소화할지 정하는 `direction`, hyperparameter set을 sampling하는 `sampler`, 가망이 없는 학습을 중단하는 `pruner`가 있고,

`optuna.study.Study.optimize` 에는 몇 번의 searching을 할 지 정하 `n_trials`, trial이 끝날 때마다 호출되는 `callbacks` 함수를 정의할 수 있다.

보다시피, 나의 코드에는 optimization에 중요한 `pruner`, `sampler`가 비어 있다.

이 경우에는 `TPESampler`와 `MedianPruner`가 지정되게 된다. *pruning 설정 하지 않았는데 갑자기 pruning을 하길래, 퍼플렉시티가 설정하지 않으면 pruning하지 않는다고 우겨서 싸우다가 결국 공식문서를 찾았다.


## Sampler

Sampler는 `hyperparameter_space`에서 정한 하이퍼파라미터 범위에서 어떤 하이퍼파라미터를 고를지 정하는 알고리즘이다.

default는 TPESampler이며, GridSampler, RandomSampler, TPESampler, CmaEsSampler, BruteForceSampler 등이 있다.

어떤 Sampler가 최적인지 모르겠으면, [AutoSampler](https://hub.optuna.org/samplers/auto_sampler/)를 통해 자동으로 Sampler를 고를 수 있다.

|Sampler                       |Description                                        |Time Complexity         |Sequential/Parallel Support|Multivariate Support|Note                             |
|------------------------------|---------------------------------------------------|------------------------|---------------------------|--------------------|---------------------------------|
|RandomSampler                 |Samples parameters uniformly at random             |O(1)                    |Both                       |No                  |Baseline                         |
|TPESampler                    |UsesTPEbased on Parzen estimator for Bayesian opt  |O(n log n)              |Both                       |Yes                 |Probabilistic model based        |
|CmaEsSampler                  |Covariance Matrix Adaptation Evolution Strategy    |O(n^2)                  |Sequential only            |No                  |Evolutionary algorithm           |
|NSGAIISampler                 |NSGA-II multi-objective evolutionary algorithm     |O(m n^2)                |Sequential only            |No                  |m: number of objectives          |
|MOTPESampler                  |Multi-objective TPE                                |O(n log n)              |Both                       |Yes                 |Combines multiobj and TPE        |
|GridSampler                   |Grid search over predefined discrete values        |O(k^d)                  |Both                       |No                  |k: grid points per dim, d: dims  |
|BruteForceSampler             |Samples every point by step size in parameter range|O(k^d)                  |Both                       |No                  |Step-size based exhaustive search|
|PartialFixedSampler           |Fixes some parameters, samples others              |Depends on reduced space|Both                       |Depends             |For partial search spaces        |
|IntersectionSearchSpaceSampler|Samples intersection of search spaces              |Depends                 |Both                       |Depends             |For combined search spaces       |
|TPESamplerMultivariate        |Multivariate TPE model                             |More than O(n log n)    |Both                       |Yes                 |Models param dependencies        |


### GridSampler

serach space만 주어지면 Grid search를 수행한다. 

"모든 경우의 수"를 추적해야 하기 때문에, 하이퍼파라미터 범위는 범주형으로 지정되어야 한다.

### BruteForceSampler

Grid Search와 유사하지만, BruteForceSampler는 suggest_float에서 `step`을 지정해 놓으면 categorial이 아니더라도 사용 가능하다.

### RandomSampler

seed에만 영향을 받아, 랜덤한 하이퍼파라미터를 뽑아 trial을 만든다.

### GPSampler

Gaussian Process를 사용하는 기본적인 Bayesian optimization Sampler이다.

### [TPE Sampler](https://arxiv.org/abs/2304.11127)

Bayesian sampling에 Tree 구조를 더한 알고리즘이다.

### CmaEsSampler

[CmaEs](https://github.com/CyberAgentAILab/cmaes?tab=readme-ov-file) 를 backbone으로 하는 Sampler.

<img width="1000" height="500" alt="image" src="https://github.com/user-attachments/assets/85936089-56c1-414c-bce9-0bff547a381d" />

### PartialFixedSampler

한 번 study를 진행한 후, 특정 hyperparameter를 고정하고 싶을 때 쓰는 Sampler.

`fixed_params`와 `sampler`를 인자로 받아, `fixed_params`를 고정시킨다.

## Pruner



Sampler가 고른 하이퍼파라미터를 검증하던 도중, 가능성이 없는 trial을 가지치기하는 방법이다.

Prune의 default는 `MedianPruner`이며, Prune을 하지 않으려면 `NopPruner`를 지정해 주어야 한다.

### MedianPruner

현재 학습 중간 결과가, 같은 step의 이전 학습들의 중간값보다 좋지 않다면 prune한다.
인자로 `n_startup_trials = 5`, `n_warmup_steps = 0`를 받는다.
`n_startup_trials`는 몇 번째 trial부터 Pruner가 시작할지(기본은 5이다)
`n_warmup_steps`는 몇 번째 step부터 Pruner가 시작할지를 정한다.

warmup이나 startup을 설정하지 않으면, Sampler가 explor를 하지 못해 hyperparameter들의 local minima에 빠질 수 있다.

### PercentilePruner

Median Pruner에서 이전 값들 Median 대신 percentile 값을 받는 Pruner이다.

### ThresholdPruner

`lower`, `upper`, `n_warmup_steps`를 받아, warmup step 이 평가 지표가 lower와 upper를 넘어서면 prune.

### WilcoxonPruner

[Wilcoxon signed-rank test
](https://en.wikipedia.org/w/index.php?title=Wilcoxon_signed-rank_test&oldid=1195011212)

현재의 trial과 현재까지 best trial 간의 Wilcoxon signed-rank test를 수행한다.

`p_threshold = 0.1`을 인자로 받아, 
Wilcoxon signed-rank test를 통과했을 때 p값보다 작으면 prune한다.

즉, Wilcoxon signed-rank test가 best trial이 현재의 trial보다 좋을 확률이 (1-`p_threshold`)보다 높다고 판별하면 prune하는 방식이다.

### PatientPruner

PatientPruner는 다른 Pruner를 감싸는 Pruner로,

인자로 다른 pruner와 `patience`, `min_delta`를 받는다.

Early stopping과 유사하게, `patience` 기간 동안 성능이 `min_delta`만큼 개선되지 않으면 Prune한다.

다른 Pruner를 감싸는 Wrapper Pruner이므로, 또다른 Pruner(MedianPruner ..)와 함께 동작한다.



