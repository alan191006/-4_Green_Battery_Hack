import pprint
from collections import deque

from optimize import optimize
from policy import Policy

# Submission
class EMAPolicy(Policy):
    def __init__(
        self,
        short_window: int = 32,
        medium_window: int = 244,
        long_window: int = 105,
        threshold_lower: int = 45,
        threshold_higher: int = 214,
    ):
        super().__init__()
        self.short_window = short_window
        self.medium_window = medium_window
        self.long_window = long_window

        self.short_prices = deque(maxlen=short_window)
        self.medium_prices = deque(maxlen=medium_window)
        self.long_prices = deque(maxlen=long_window)

        self.short_alpha = 2 / (short_window + 1)
        self.medium_alpha = 2 / (medium_window + 1)
        self.long_alpha = 2 / (long_window + 1)

        self.threshold_lower = threshold_lower
        self.threshold_higher = threshold_higher

        # self.previous_action = []

    # EMA component
    def calc_ema(self, prices, alpha):
        ema = prices[0]
        prices_list = list(prices)
        for price in prices_list[1:]:
            ema = alpha * price + (1 - alpha) * ema
        return ema

    # Thresholding componentv
    def thresholding(self, price, charge):
        if price < self.threshold_lower:
            charge = max(0, charge)
        elif price > self.threshold_higher:
            charge = min(0, charge)
        return charge

    # Logic component
    def control(self, price, charge):
        max_charge_rate = 5
        if price < self.threshold_lower or charge >= 0:
            return max_charge_rate
        elif price > self.threshold_higher or charge <= 0:
            return -max_charge_rate
        return 0

    def act(self, external_state, internal_state):
        market_price = external_state["price"]
        # pv_power = external_state["pv_power"]

        self.short_prices.append(market_price)
        self.medium_prices.append(market_price)
        self.long_prices.append(market_price)

        if len(self.short_prices) == self.short_window:
            short_ema = self.calc_ema(self.short_prices, self.short_alpha)
            medium_ema = self.calc_ema(self.medium_prices, self.medium_alpha)
            long_ema = self.calc_ema(self.long_prices, self.long_alpha)

            if short_ema > medium_ema and short_ema > long_ema:
                charge_kW = -internal_state["max_charge_rate"]  # Buy Signal
            elif short_ema < medium_ema and short_ema < long_ema:
                charge_kW = internal_state["max_charge_rate"]  # Sell Signal
            else:
                charge_kW = 0  # No Signal
        else:
            charge_kW = 0

        charge_kW = self.thresholding(market_price, charge_kW)

        return self.control(market_price, charge_kW), charge_kW

    def load_historical(self, external_states):
        # for price in external_states["price"].values:
        #     self.short_prices.append(price)
        #     self.medium_prices.append(price)
        #     self.long_prices.append(price)
        #
        pass


if __name__ == "__main__":
    # params = {
    #     "short_window": (12, 36),  # 1h - 3h
    #     "medium_window": (24, 288),  # 2h - 1D
    #     "long_window": (72, 2016),  # 6h - 7D
    # }
    params = {
        "threshold_lower": (-50, 50),
        "threshold_higher": (20, 300),
    }

    max_iter = 100

    best_param_dict, best_profit = optimize(EMAPolicy, params, max_iter)

    print("Best parameters:")
    pprint.pprint(best_param_dict)

    print(f"Best profit: {best_profit}")
