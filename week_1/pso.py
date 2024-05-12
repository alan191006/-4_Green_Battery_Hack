from typing import Callable, Dict, Any, Tuple
import numpy as np

# pso is outdated and dont support integer
# from pyswarm import pso


class PSOAlgorithm:
    def __init__(
        self,
        obj_func: Callable[..., float],
        out_func: Callable[..., float],
        **kwargs: Any
    ) -> None:
        self.obj_func = obj_func
        self.out_func = out_func
        self.kwargs = kwargs
        self.best_params = None
        self.best_output = float("inf")

    def obj_func_wrapper(self, params: np.ndarray) -> float:
        output = self.obj_func(params, **self.kwargs)
        return output

    def callback(self, params: np.ndarray, output: float) -> None:
        if output < self.best_output:
            self.best_params = params.copy()
            self.best_output = output

    def optimize(
        self,
        bounds: Dict[str, Tuple[float, float]],
        maxiter: int = 100,
        swarmsize: int = 100,
        integer_params: Dict[str, bool] = None,
    ) -> Tuple[np.ndarray, float]:

        bounds_list = [(bounds[key][0], bounds[key][1]) for key in bounds]

        if integer_params is None:
            integer_params = {key: False for key in bounds}

        # PSO init
        num_params = len(bounds_list)
        swarm = np.random.uniform(
            low=np.array([bound[0] for bound in bounds_list]),
            high=np.array([bound[1] for bound in bounds_list]),
            size=(swarmsize, num_params),
        )

        # Main PSO loop
        velocity = np.zeros_like(swarm)
        best_swarm_pos = swarm.copy()
        best_swarm_value = np.full(swarmsize, np.inf)
        best_pos = np.zeros(num_params)
        best_value = np.inf

        for _ in range(maxiter):
            for i in range(swarmsize):
                # Evaluate current position
                swarm_value = self.obj_func_wrapper(swarm[i, :])

                # Update personal best
                if swarm_value < best_swarm_value[i]:
                    best_swarm_value[i] = swarm_value
                    best_swarm_pos[i, :] = swarm[i, :]

                    # Update global best
                    if swarm_value < best_value:
                        best_value = swarm_value
                        best_pos = swarm[i, :]

            for i in range(swarmsize):
                r1, r2 = np.random.rand(), np.random.rand()
                velocity[i, :] = (
                    velocity[i, :]
                    + r1 * (best_swarm_pos[i, :] - swarm[i, :])
                    + r2 * (best_pos - swarm[i, :])
                )
                swarm[i, :] = swarm[i, :] + velocity[i, :]

                # Clamp values to bounds
                for j in range(num_params):
                    if integer_params[list(bounds.keys())[j]]:
                        swarm[i, j] = int(np.round(swarm[i, j]))
                    swarm[i, j] = np.clip(
                        swarm[i, j], bounds_list[j][0], bounds_list[j][1]
                    )

        self.best_params = best_pos
        self.best_output = best_value
        return self.best_params, self.out_func(self.best_params, **self.kwargs)
