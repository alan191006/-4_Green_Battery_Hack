"""
* Note: doesn't laways work?
"""

import pprint
import pandas as pd
from collections import deque
from optimize import optimize
from policy import Policy


class MeanReversionPolicy(Policy):
    def __init__(
        self,
        rsi_window: int = 14,
        rsi_oversold: float = 76,
        rsi_overbought: float = 118,
    ):
        super().__init__()

        self.rsi_window = rsi_window
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought

        self.prices = deque(maxlen=rsi_window)

    def calculate_rsi(self):
        if len(self.prices) < self.rsi_window:
            return None

        deltas = pd.Series(self.prices).diff(1)

        gain = deltas.where(deltas > 0, 0)
        loss = deltas.where(deltas < 0, 0)

        avg_gain = gain.rolling(window=self.rsi_window).mean()
        avg_loss = loss.rolling(window=self.rsi_window).mean().abs()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi.iloc[-1]

    def act(self, external_state, internal_state):
        market_price = external_state["price"]

        self.prices.append(market_price)

        rsi = self.calculate_rsi()

        if rsi is not None:

            if rsi < self.rsi_oversold:
                return 0, -internal_state["max_charge_rate"]

            elif rsi > self.rsi_overbought:
                return 0, internal_state["max_charge_rate"]

            else:
                return 0, 0

        return 0, 0

    def load_historical(self, external_states: pd.DataFrame):
        pass


if __name__ == "__main__":
    params = {
        "rsi_window": (14, 2016),
        "rsi_oversold": (0, 40),
        "rsi_overbought": (60, 100),
    }

    best_param_dict, best_profit = optimize(MeanReversionPolicy, params)

    print("Best parameters:")
    pprint.pprint(best_param_dict)

    print(f"Best profit: {best_profit}")
