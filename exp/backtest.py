import numpy as np
import pandas as pd

from pathlib import Path

import evaluate as E
from policy import Policy
from environment import BatteryEnv

from typing import Any, Dict, Union

root = Path(__file__).parent.parent
default_data = root / "data/no_missing_training_data.csv"


def eval(
    policy_class: Policy,
    params: Dict[str, Any],
    seed: int = 42,
    data_path: Union[str, Path] = default_data,
    initial_soc: float = 7.5,
    initial_profit: float = 0,
) -> float:
    """Evaluate the performance of a given policy on a battery environment."""
    external_states = pd.read_csv(data_path)

    E.set_seed(seed)
    start_step = 0

    historical_data = external_states.iloc[:start_step]
    future_data = external_states.iloc[start_step:]

    battery_environment = BatteryEnv(
        data=future_data, initial_charge_kWh=initial_soc, initial_profit=initial_profit
    )

    policy = policy_class(**params)
    policy.load_historical(historical_data)
    trial_data = E.run_trial(battery_environment, policy)

    total_profits = trial_data["profits"]
    rundown_profit_deltas = trial_data["rundown_profit_deltas"]

    mean_combined_profit = total_profits[-1] + np.sum(rundown_profit_deltas)

    return mean_combined_profit
