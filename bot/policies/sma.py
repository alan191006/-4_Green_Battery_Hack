"""
File: ema.py
Author: Alan Huynh
Group: ω4
Date: 14-Apr-2024

Description:
    Implements MultiEMAPolicy class, based on 3 Exponential Moving Averages.
    Buys when the short-term EMA crosses above both the medium and long-term 
    EMAs, and sells when the short-term EMA crosses below both the medium and 
    long-term EMAs
    
Additional information:
    * Note: Works but worse than an SMA
"""

import pandas as pd
from collections import deque
import numpy as np
from policies.policy import Policy


class SMAPolicy(Policy):
    def __init__(self, short_window=20, medium_window=271, long_window=434):
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
