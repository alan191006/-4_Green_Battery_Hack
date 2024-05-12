from typing import List
import pandas as pd
import torch
from torch.utils.data import Dataset
from numpy.lib.stride_tricks import sliding_window_view
import numpy as np


class EnsembleOutputDataset(Dataset):
    def __init__(
        self,
        grid_action_series: np.ndarray,
        pv_action_series: np.ndarray,
        pv_power_series: np.ndarray,
        price_series: np.ndarray,
        peak_ind_series: np.ndarray,
        interval_len: int = 14 * 60 * (60 // 5),
        skip_step: int = (60 // 5),
    ):
        self.grid_action_series = grid_action_series  # (T, C)
        self.pv_action_series = pv_action_series  # (T, C)
        self.pv_power_series = pv_power_series  # (T,)
        self.price_series = price_series  # (T,)
        self.peak_ind_series = peak_ind_series  # (T,)
        self.interval_len = interval_len
        self.skip_step = skip_step

        self.grid_actions = sliding_window_view(
            grid_action_series, self.interval_len, 0
        )[:: self.skip_step, :].copy()
        self.pv_actions = sliding_window_view(pv_action_series, self.interval_len, 0)[
            :: self.skip_step, :
        ].copy()
        self.price = sliding_window_view(price_series, self.interval_len, 0)[
            :: self.skip_step, :
        ].copy()
        self.peak_indicator = sliding_window_view(
            peak_ind_series, self.interval_len, 0
        )[:: self.skip_step, :].copy()

    def __len__(self):
        return len(self.grid_actions)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.grid_actions[idx].T),
            torch.from_numpy(self.pv_actions[idx].T),
            torch.from_numpy(self.pv_power_series[idx : idx + self.interval_len].T),
            torch.from_numpy(self.price_series[idx : idx + self.interval_len].T),
            torch.from_numpy(self.peak_ind_series[idx : idx + self.interval_len].T),
        )  # shape will be converted to (batch, T) or (batch, T, C) once batched


class EnsembleLiveDataset(Dataset):
    def __init__(
        self,
        original_df: pd.DataFrame,
        processed_data: np.ndarray,
        pv_power: np.ndarray,
        price: np.ndarray,
        peak_indicator: np.ndarray,
        policies: List = [],
        interval_len=14 * 60 * (60 // 5),
        skip_step=(60 // 5),
    ):
        self.interval_len = interval_len
        # sliding widows with valid values only
        self.df_data = [
            original_df.iloc[s * skip_step : s * skip_step + interval_len]
            for s in range((len(original_df) - interval_len) // skip_step + 1)
        ]
        self.processed_data = sliding_window_view(processed_data, self.interval_len, 0)[
            ::skip_step, :
        ].copy()
        assert len(self.df_data) == len(self.processed_data)
        self.skip_step = skip_step
        self.price = sliding_window_view(price, self.interval_len, 0)[
            ::skip_step, :
        ].copy()
        self.pv_power = sliding_window_view(pv_power, self.interval_len, 0)[
            ::skip_step, :
        ].copy()
        self.peak_indicator = sliding_window_view(peak_indicator, self.interval_len, 0)[
            ::skip_step, :
        ].copy()

        self.grid_actions = np.zeros((len(self.df_data), self.interval_len, len(policies)))
        self.pv_actions = np.zeros((len(self.df_data), self.interval_len, len(policies)))

        for i in range(len(self.df_data)):
            actions = []
            df = self.df_data[i]
            dummy_internal_state = {"max_charge_rate": 5}
            policy_objects = [obj() for obj in policies]
            for policy in policy_objects:
                policy_actions = []
                for j, external_state in df.iterrows():
                    policy_actions.append(policy.act(external_state, dummy_internal_state))
                policy_actions = np.array(policy_actions)
                actions.append(policy_actions)
            actions = np.stack(actions, -1)
            self.grid_actions[i] = actions[:, 0, :]
            self.pv_actions[i] = actions[:, 1, :]

    def __len__(self):
        return len(self.df_data)

    def __getitem__(self, idx):
        # data = self.processed_data[idx]
        price = self.price[idx]
        pv_power = self.pv_power[idx]
        peak_ind = self.peak_indicator[idx]
        grid_actions = self.grid_actions[idx]
        pv_actions = self.pv_actions[idx]

        return (
            torch.from_numpy(grid_actions),
            torch.from_numpy(pv_actions),
            # torch.from_numpy(data.T),
            torch.from_numpy(pv_power.T),
            torch.from_numpy(price.T),
            torch.from_numpy(peak_ind.T),
        )
