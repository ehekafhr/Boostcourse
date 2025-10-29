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



## Pruner
