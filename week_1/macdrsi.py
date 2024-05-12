"""
* Note: Slow, should be a better way to implement this...
"""

import pprint
import pandas as pd
from collections import deque
from optimize import optimize
from policy import Policy


class MACDRSIPolicy(Policy):
    def __init__(
        self,
        short_window: int = 12,
        long_window: int = 26,
        signal_window: int = 9,
        rsi_window: int = 14,
        rsi_oversold: float = 30,
        rsi_overbought: float = 70,
    ):
        super().__init__()
        self.short_window = short_window
        self.long_window = long_window
        self.signal_window = signal_window
        self.rsi_window = rsi_window
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought

        self.short_prices = deque(maxlen=long_window)
        self.long_prices = deque(maxlen=long_window)
        self.macd_values = deque(maxlen=signal_window)
        self.signal_values = deque(maxlen=signal_window)
        self.rsi_values = deque(maxlen=rsi_window)

    def calc_ema(self, prices, window):
        alpha = 2 / (window + 1)
        ema = prices[0]
        prices_list = list(prices)
        for price in prices_list[1:]:
            ema = alpha * price + (1 - alpha) * ema
        return ema

    # def calc_sma(self, prices, window):
    #     if len(prices) < window:
    #         return None
    #     return sum(prices[-window:]) / window

    def calc_macd(self):
        short_ema = self.calc_ema(self.short_prices, self.short_window)
        long_ema = self.calc_ema(self.long_prices, self.long_window)
        macd = short_ema - long_ema
        return macd

    def calc_signal_line(self):
        # macd = self.calc_macd()
        signal_line = self.calc_ema(self.macd_values, self.signal_window)
        return signal_line

    def calc_rsi(self):
        prices_list = list(self.long_prices)
        if len(prices_list) < self.rsi_window:
            return None
        deltas = pd.Series(prices_list).diff(1)
        gain = deltas.where(deltas > 0, 0)
        loss = deltas.where(deltas < 0, 0)
        avg_gain = gain.rolling(window=self.rsi_window).mean()
        avg_loss = loss.rolling(window=self.rsi_window).mean().abs()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1]

    def act(self, external_state, internal_state):
        market_price = external_state["price"]

        self.short_prices.append(market_price)
        self.long_prices.append(market_price)
        self.macd_values.append(self.calc_macd())
        self.signal_values.append(self.calc_signal_line())
        self.rsi_values.append(self.calc_rsi())

        if len(self.long_prices) >= max(
            self.short_window, self.long_window, self.signal_window, self.rsi_window
        ):
            macd = self.macd_values[-1]
            signal_line = self.signal_values[-1]
            rsi = self.rsi_values[-1]

            if macd > signal_line and rsi > self.rsi_overbought:
                return 0, internal_state["max_charge_rate"]
            elif macd < signal_line and rsi < self.rsi_oversold:
                return 0, -internal_state["max_charge_rate"]
            else:
                return 0, 0

        return 0, 0

    def load_historical(self, external_states: pd.DataFrame):
        # for price in external_states["price"].values:
        #     self.short_prices.append(price)
        #     self.long_prices.append(price)
        pass


if __name__ == "__main__":
    params = {
        "short_window": (3, 576),  # 15m - 2D
        "long_window": (72, 2016),  # 6h - 7D
        "signal_window": (3, 576),  # 15m - 2D
        "rsi_window": (3, 576),  # 15m - 2D
        "rsi_oversold": (1, 100),
        "rsi_overbought": (1, 100),
    }

    max_iter = 500

    best_param_dict, best_profit = optimize(MACDRSIPolicy, params, max_iter)

    print("Best parameters:")
    pprint.pprint(best_param_dict)

    print(f"Best profit: {best_profit}")
