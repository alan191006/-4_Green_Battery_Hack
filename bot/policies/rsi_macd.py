"""
File: ema.py
Author: Alan Huynh
Group: ω4
Date: 6-Apr-2024

Description:
    Define RSIMACDPolicy class. Using Relative Strength Index (RSI) and Moving 
    Average Convergence Divergence (MACD) indicators. Buys when RSI is below 30
    and MACD crosses above Signal, and sells when RSI is above 70 and MACD 
    crosses below Signal, aiming to capitalize on oversold and overbought 
    market conditions.
    
Additional information:
    * Note: This doesn't seem to work
"""

import pandas as pd
from collections import deque
import numpy as np
from policies.policy import Policy


class RSIMACDPolicy(Policy):
    def __init__(
        self,
        rsi_period=14,
        short_ema_period=12,
        long_ema_period=26,
        signal_ema_period=9,
    ):
        super().__init__()
        self.rsi_period = rsi_period
        self.short_ema_period = short_ema_period
        self.long_ema_period = long_ema_period
        self.signal_ema_period = signal_ema_period

        self.rsi_values = deque(maxlen=rsi_period)
        self.price_values = deque(maxlen=max(long_ema_period, signal_ema_period) + 1)
        self.macd_values = deque(maxlen=long_ema_period + 1)
        self.signal_values = deque(maxlen=signal_ema_period + 1)

    def calculate_rsi(self):
        delta = np.diff(np.array(self.price_values))
        gains = delta.clip(min=0)
        losses = -delta.clip(max=0)
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)

        if avg_loss == 0:
            return 100
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def calculate_macd(self):
        short_ema = np.mean(list(self.price_values)[-self.short_ema_period :])
        long_ema = np.mean(list(self.price_values)[-self.long_ema_period :])

        macd_line = short_ema - long_ema

        self.macd_values.append(macd_line)

        if len(self.macd_values) >= self.signal_ema_period:
            signal_line = np.mean(list(self.macd_values)[-self.signal_ema_period :])
            self.signal_values.append(signal_line)
        else:
            self.signal_values.append(np.nan)

    def act(self, external_state, internal_state):
        market_price = external_state["price"]
        self.price_values.append(market_price)

        if (
            len(self.price_values)
            >= max(self.long_ema_period, self.signal_ema_period) + 1
        ):
            rsi = self.calculate_rsi()
            self.rsi_values.append(rsi)
            self.calculate_macd()

            if rsi < 30 and self.macd_values[-1] > self.signal_values[-1]:
                charge_kW = -internal_state["max_charge_rate"]  # Buy Signal
            elif rsi > 70 and self.macd_values[-1] < self.signal_values[-1]:
                charge_kW = internal_state["max_charge_rate"]  # Sell Signal
            else:
                charge_kW = 0  # No Signal
        else:
            charge_kW = 0

        return 0, charge_kW

    def load_historical(self, external_states: pd.DataFrame):
        for price in external_states["price"].values:
            self.price_values.append(price)
