import pprint
import numpy as np
import pandas as pd
from collections import deque

from optimize import optimize
from policy import Policy


class SMAPolicy(Policy):
    def __init__(
        self, short_window: int = 5, medium_window: int = 10, long_window: int = 20
    ):
        super().__init__()
        self.short_window = short_window
        self.medium_window = medium_window
        self.long_window = long_window

        self.short_prices = deque(maxlen=short_window)
        self.medium_prices = deque(maxlen=medium_window)
        self.long_prices = deque(maxlen=long_window)

    def act(self, external_state, internal_state):
        market_price = external_state["price"]
        self.short_prices.append(market_price)
        self.medium_prices.append(market_price)
        self.long_prices.append(market_price)

        if len(self.short_prices) == self.short_window:
            short_ema = np.mean(self.short_prices)
            medium_ema = np.mean(self.medium_prices)
            long_ema = np.mean(self.long_prices)

            if short_ema > medium_ema and short_ema > long_ema:
                charge_kW = -internal_state["max_charge_rate"]  # Buy Signal
            elif short_ema < medium_ema and short_ema < long_ema:
                charge_kW = internal_state["max_charge_rate"]  # Sell Signal
            else:
                charge_kW = 0  # No Signal
        else:
            charge_kW = 0

        return 0, charge_kW

    def load_historical(self, external_states: pd.DataFrame):
        for price in external_states["price"].values:
            self.short_prices.append(price)
            self.medium_prices.append(price)
            self.long_prices.append(price)


if __name__ == "__main__":
    params = {
        "short_window": (12, 36),  # 1h - 3h
        "medium_window": (24, 288),  # 2h - 1D
        "long_window": (72, 2016),  # 6h - 7D
    }

    best_param_dict, best_profit = optimize(SMAPolicy, params)

    print("Best parameters:")
    pprint.pprint(best_param_dict)

    print(f"Best profit: {best_profit}")
