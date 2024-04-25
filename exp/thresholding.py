# Fit on clamped test set
import pprint

from policy import Policy
from optimize import optimize


class ThresholdingPolicy(Policy):
    def __init__(self, lb=-8, hb=80, exc_threshold=50):
        super().__init__()
        self.lb = lb
        self.hb = hb
        print(rf"({lb}, {hb})")
        self.exc_threshold = exc_threshold
        # self.position = 0

    def act(self, external_state, internal_state):
        market_price = external_state["price"]
        charge_rate = internal_state["max_charge_rate"]
        # soc = internal_state["battery_soc"]

        # if self.position == 0:
        if market_price < self.lb:
            # self.position = 1
            return 0, charge_rate
        # elif self.position == 1:
        if market_price >= self.hb:
            # self.position = 0
            return 0, -charge_rate

        # if market_price > self.exc_threshold:
        #     if soc > charge_rate:
        #         return 0, charge_rate - soc

        return 0, 0

    def load_historical(self, external_states):
        pass


if __name__ == "__main__":

    params = {
        "lb": (-100, 200),
        "hb": (0, 2000),
    }

    max_iter = 300

    best_param_dict, best_profit = optimize(ThresholdingPolicy, params, max_iter)

    print("Best parameters:")
    pprint.pprint(best_param_dict)

    print(f"Best profit: {best_profit}")
