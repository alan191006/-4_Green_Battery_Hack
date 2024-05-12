"""
File: thresholding.py
Author: Alan Huynh
Group: ω4
Date: 6-Apr-2024

Description:
    Defines ThresholdingPolicy class. Buys below lower bound, sells above upper
    bound, and sells if price surpasses threshold to prevent negative returns 
    on negative prices.
    
Additional information:
    - Version: 0.0.2
    - Lower bound: -8
    - Upper bound: 80
"""

from policies.policy import Policy


class ThresholdingPolicy(Policy):
    def __init__(self, lb=-8, hb=80, exc_threshold=50):
        super().__init__()
        self.lb = lb
        self.hb = hb
        print(rf"({lb}, {hb})")
        self.exc_threshold = exc_threshold
        self.position = 0

    def act(self, external_state, internal_state):
        market_price = external_state["price"]
        charge_rate = internal_state["max_charge_rate"]
        soc = internal_state["battery_soc"]

        # if self.position == 0:
        if market_price < self.lb:
            self.position = 1
            return 0, charge_rate
        # elif self.position == 1:
        if market_price >= self.hb:
            self.position = 0
            return 0, -charge_rate

        # if market_price > self.exc_threshold:
        #     if soc > charge_rate:
        #         return 0, charge_rate - soc

        return 0, 0

    def load_historical(self, external_states):
        pass
