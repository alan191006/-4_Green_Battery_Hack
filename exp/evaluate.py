import json
import random
import argparse
import numpy as np

from typing import List, Dict, Any, Union

from policy import Policy
from environment import BatteryEnv, PRICE_KEY, TIMESTAMP_KEY


def float_or_none(value: str) -> Union[float, None]:

    if value.lower() == "none":
        return None
    try:
        return float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value} is not a float or 'None'")


def load_config(file_path: str) -> Dict[str, Any]:
    """Load configuration from a JSON file."""
    with open(file_path, "r") as file:
        return json.load(file)["policy"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def run_down_battery(
    battery_environment: BatteryEnv, market_prices: List[float]
) -> List[float]:
    """Simulate battery rundown until empty."""

    rundown_profits = []

    last_day_prices = market_prices[-288:]
    assumed_rundown_price = np.mean(last_day_prices)

    while battery_environment.battery.state_of_charge_kWh > 0:

        kWh_removed = battery_environment.battery.discharge_at(
            battery_environment.battery.max_charge_rate_kW
        )

        rundown_profits.append(
            battery_environment.kWh_to_profit(kWh_removed, assumed_rundown_price)
        )

    return rundown_profits


def run_trial(battery_environment: BatteryEnv, policy: Policy) -> Dict[str, Any]:
    """Run a trial of the battery environment with a policy."""
    (
        profits,
        socs,
        market_prices,
        battery_actions,
        solar_actions,
        pv_inputs,
        timestamps,
    ) = ([], [], [], [], [], [], [])

    external_state, internal_state = battery_environment.initial_state()

    while True:

        pv_power = float(external_state["pv_power"])
        solar_kW_to_battery, charge_kW = policy.act(external_state, internal_state)

        market_prices.append(external_state[PRICE_KEY])
        timestamps.append(external_state[TIMESTAMP_KEY])
        battery_actions.append(charge_kW)
        solar_actions.append(solar_kW_to_battery)
        pv_inputs.append(pv_power)

        external_state, internal_state = battery_environment.step(
            charge_kW, solar_kW_to_battery, pv_power
        )

        profits.append(internal_state["total_profit"])
        socs.append(internal_state["battery_soc"])

        if external_state is None:
            break

    rundown_profits = run_down_battery(battery_environment, market_prices)

    return {
        "profits": profits,
        "socs": socs,
        "market_prices": market_prices,
        "actions": battery_actions,
        "solar_actions": solar_actions,
        "pv_inputs": pv_inputs,
        "final_soc": socs[-1],
        "rundown_profit_deltas": rundown_profits,
        "timestamps": timestamps,
    }


def parse_parameters(params_list: List[str]) -> Dict[str, Any]:
    params = {}

    for item in params_list:
        key, value = item.split("=")
        params[key] = eval(value)

    return params
