# doesnt work..

import pprint
import pandas as pd
from optimize import optimize
from policy import Policy
from sma import SMAPolicy
from ema import EMAPolicy
from macdrsi import MACDRSIPolicy
from bollingerband import BollingerBandPolicy
from meanrev import MeanReversionPolicy

policies = [
    SMAPolicy(),
    EMAPolicy(),
    MACDRSIPolicy(),
    BollingerBandPolicy(),
    MeanReversionPolicy(),
]


class EnsemblePolicy(Policy):
    def __init__(self, weights=None):
        super().__init__()
        self.policies = [
            SMAPolicy(),
            EMAPolicy(),
            MACDRSIPolicy(),
            BollingerBandPolicy(),
            MeanReversionPolicy(),
        ]
        if weights is None:
            self.weights = [1] * len(self.policies)
        else:
            self.weights = weights

    def act(self, external_state, internal_state):
        votes = {}

        # Get weighted votes
        for i, policy in enumerate(self.policies):
            _, vote = policy.act(external_state, internal_state)
            vote *= self.weights[i]
            votes[i] = vote

        total_weighted_vote = sum(votes.values())
        total_weight = sum(self.weights)
        final = total_weighted_vote / total_weight

        max_rate = internal_state["max_charge_rate"]

        if final > max_rate // 2:
            return 0, max_rate

        elif final < -max_rate // 2:
            return 0, -max_rate

        else:
            return 0, 0

    def load_historical(self, external_states: pd.DataFrame):
        for policy in self.policies:
            policy.load_historical(external_states)


if __name__ == "__main__":

    # Define initial weights for optimization
    initial_weights = [1] * len(policies)

    # Define parameter ranges for optimization
    params = {f"weight_{i}": (0.0, 2.0) for i in range(len(policies))}

    max_iter = 200

    best_param_dict, best_profit = optimize(EnsemblePolicy, params, max_iter)

    print("Best parameters:")
    pprint.pprint(best_param_dict)

    print(f"Best profit: {best_profit}")
