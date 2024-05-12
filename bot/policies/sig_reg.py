import pandas as pd
from policies.policy import Policy
import torch

from .modules.signature_regression_model import SigRegModel, BatteryEnv
from .modules.segment_dataset import (
    load_and_scale_data,
    learn_scaling_params,
)

active_columns = [
    "timestamp",
    "price",
    "demand",
    "demand_total",
    "pv_power",
    "pv_power_forecast_1h",
    "pv_power_forecast_2h",
    "pv_power_forecast_24h",
]


class RegressionPolicy(Policy):
    def __init__(self):
        super().__init__()
        battery = BatteryEnv(13, 5, 7.5)
        self.model = SigRegModel.load_from_checkpoint(
            "bot/data/checkpoints/sig-reg-epoch=25-with-val_loss=-0.020.ckpt",
            battery=battery,
            map_location="cpu",
        )
        self.model.beta = None
        self.historical_states = []
        self.scaler = learn_scaling_params("bot/data/no_missing_training_data.csv")
        self.means = pd.read_csv("bot/data/training_means.csv")[active_columns].iloc[0]
        self.previous_endpoint = None
        self.previous_sig = None

    def act(self, external_state, internal_state):
        current = external_state[active_columns]
        if len(self.historical_states) > 0:
            last = self.historical_states[-1]
            current.fillna(last, inplace=True)
        else:
            current.fillna(self.means, inplace=True)
        insert = pd.DataFrame([current])
        insert["timestamp"] = pd.to_datetime(insert["timestamp"])
        self.historical_states.append(current)
        state, _, _, _ = load_and_scale_data(df=insert, scaler=self.scaler)
        state = torch.tensor(state).float()[None, ...]
        self.model.eval()
        battery_state = (
            torch.tensor(internal_state["battery_soc"]).float()[None, None]
        )
        with torch.no_grad():
            grid_action, pv_action, self.previous_endpoint, self.previous_sig = (
                self.model.step_forward(
                    battery_state, state, self.previous_endpoint, self.previous_sig
                )
            )
        cr = internal_state["max_charge_rate"]
        pv_charge = pv_action[0, 0] * external_state["pv_power"]
        grid_charge = grid_action[0, 0] * cr
        return pv_charge.numpy().item(), grid_charge.numpy().item()

    def load_historical(self, external_states: pd.DataFrame):
        pass
