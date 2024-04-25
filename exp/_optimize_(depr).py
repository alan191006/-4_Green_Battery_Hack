import numpy as np


def binary_search_optimization(func, lower_bound, upper_bound, max_iter):

    def objective(params):
        return func(params)

    best_params = [(low + high) / 2 for low, high in zip(lower_bound, upper_bound)]
    best_value = objective(best_params)

    for _ in range(max_iter):

        step_size = max((high - low) / 4 for low, high in zip(lower_bound, upper_bound))

        improved = False

        for i in range(len(lower_bound)):
            for direction in [-1, 1]:

                params = best_params.copy()
                params[i] += direction * step_size

                if all(
                    lb <= p <= ub for lb, p, ub in zip(lower_bound, params, upper_bound)
                ):

                    value = objective(params)

                    if value > best_value:
                        best_value = value
                        best_params = params
                        improved = True

        if not improved:
            break

    return best_params, best_value


def optimize(func, params, max_iter=100):

    param_names = list(params.keys())
    lower_bound = []
    upper_bound = []
    for param in param_names:
        lower_bound.append(params[param][0])
        upper_bound.append(params[param][1])

    def wrapped_function(parameters):
        param_values = {}
        for i in range(len(param_names)):
            param_values[param_names[i]] = parameters[i]
        return func(param_values)

    best_param_dict = {}
    best_profit = float("-inf")

    for _ in range(max_iter):

        parameters = []
        for i in range(len(lower_bound)):
            parameters.append(np.random.uniform(lower_bound[i], upper_bound[i]))

        value = wrapped_function(parameters)

        if value > best_profit:
            best_profit = value
            best_param_dict = {}
            for i in range(len(param_names)):
                best_param_dict[param_names[i]] = parameters[i]

    return best_param_dict, best_profit
