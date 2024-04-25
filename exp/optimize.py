import optuna
from typing import Dict, Tuple, Any

import backtest
from policy import Policy


def tpe(func, lower_bound, upper_bound, max_iter):

    def objective(trial):
        params = []

        for i in range(len(lower_bound)):

            param_name = f"param_{i}"
            param_value = trial.suggest_int(param_name, lower_bound[i], upper_bound[i])

            params.append(param_value)

        return func(params)

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler()
    )

    study.optimize(objective, n_trials=max_iter)

    best_params = [study.best_params[f"param_{i}"] for i in range(len(lower_bound))]
    best_value = study.best_value

    return best_params, best_value


def optimize(
    policy: Policy,
    params: Dict[str, Tuple[float, float]],
    max_iter: int = 100,
) -> Tuple[Dict[str, Any], float]:

    param_names = list(params.keys())

    lb = [params[param][0] for param in param_names]
    ub = [params[param][1] for param in param_names]

    def wrapped_function(parameters: Any) -> float:
        param_values = {param_names[i]: parameters[i] for i in range(len(param_names))}
        return backtest.eval(policy, param_values)

    best_param, best_profit = tpe(wrapped_function, lb, ub, max_iter=max_iter)

    best_param_dict = {param_names[i]: best_param[i] for i in range(len(param_names))}

    return best_param_dict, best_profit
