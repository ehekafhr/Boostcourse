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

### PatientPruner

PatientPruner는 다른 Pruner를 감싸는 Pruner로,

인자로 다른 pruner와 `patience`, `min_delta`를 받는다.

Early stopping과 유사하게, `patience` 기간 동안 성능이 `min_delta`만큼 개선되지 않으면 Prune한다.

다른 Pruner를 감싸는 Wrapper Pruner이므로, 또다른 Pruner(MedianPruner ..)와 함께 동작한다.



