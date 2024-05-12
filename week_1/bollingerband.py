import pprint
import pandas as pd
from collections import deque
from optimize import optimize
from policy import Policy


class BollingerBandPolicy(Policy):
    def __init__(
        self,
        window: int = 213,
        num_std: int = 3,
    ):
        super().__init__()
        self.window = window
        self.num_std = num_std

        self.prices = deque(maxlen=window)

    def calc_sma(self):
        if len(self.prices) < self.window:
            return None
        return sum(self.prices) / self.window

    def calc_std(self):
        if len(self.prices) < self.window:
            return None
        sma = self.calc_sma()
        variance = sum((price - sma) ** 2 for price in self.prices) / self.window
        return variance**0.5

    def act(self, external_state, internal_state):
        market_price = external_state["price"]

        self.prices.append(market_price)

        sma = self.calc_sma()
        std = self.calc_std()

        if sma is not None and std is not None:
            upper_band = sma + self.num_std * std
            lower_band = sma - self.num_std * std

            if market_price > upper_band:
                return 0, internal_state["max_charge_rate"]
            elif market_price < lower_band:
                return 0, -internal_state["max_charge_rate"]
            else:
                return 0, 0

        return 0, 0

    def load_historical(self, external_states: pd.DataFrame):
        for price in external_states["price"].values:
            self.prices.append(price)


if __name__ == "__main__":
    params = {"window": (12, 2016), "num_std": (1, 5)}  # 1h - 7D

    best_param_dict, best_profit = optimize(BollingerBandPolicy, params)

    print("Best parameters:")
    pprint.pprint(best_param_dict)

    print(f"Best profit: {best_profit}")
