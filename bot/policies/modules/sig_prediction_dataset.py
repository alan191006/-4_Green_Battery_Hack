from typing import Any, List, Optional, Tuple, Union
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer, PowerTransformer
import torch
from torch.utils.data import Dataset, ConcatDataset, TensorDataset
from numpy.lib.stride_tricks import sliding_window_view
import numpy as np
import pandas as pd


active_columns = [
    "timestamp",
    "price",
    "demand",
    # "temp_air",
    # "pv_power",
    # "pv_power_forecast_1h",
    # "pv_power_forecast_2h",
    # "pv_power_forecast_24h",
    # "pv_power_basic",
]


class SigPredictionTransformer(TransformerMixin, BaseEstimator):
    def __init__(
        self,
        active_columns: List[str] = active_columns,
        interpolate_missing: bool = True,
        daily_time_scale: float = 1.0,
        price_scaling: str = "power",
    ):
        self.active_columns = active_columns
        self.interpolate_missing = interpolate_missing
        self.daily_time_scale = daily_time_scale
        self.price_scaling = price_scaling

    def fit(self, X: pd.DataFrame, y=None):
        dat = X[active_columns].copy()
        if self.interpolate_missing:
            col_ind_without_time = [x for x in active_columns if x != "timestamp"]
            interp_values = dat[col_ind_without_time].interpolate(
                axis=0,
            )
            dat.loc[:, col_ind_without_time] = interp_values
        # this is to make the price data symmetric and appear more normal
        self.price_transformer = PowerTransformer().fit(
            dat["price"].values.reshape(-1, 1)
        )
        dat["price"] = self.price_transformer.transform(
            dat["price"].values.reshape(-1, 1)
        )
        self.scaler = MinMaxScaler((-1, 1))
        self.scaler.fit(dat[[x for x in active_columns if x not in ["timestamp"]]])
        self.time_origin = dat["timestamp"].min()

    def transform(self, X: pd.DataFrame):
        dat = X[active_columns].copy()
        if self.interpolate_missing:
            col_ind_without_time = [x for x in active_columns if x != "timestamp"]
            interp_values = dat[col_ind_without_time].interpolate(
                axis=0,
            )
            dat.loc[:, col_ind_without_time] = interp_values
        dat["price"] = self.price_transformer.transform(
            dat["price"].values.reshape(-1, 1)
        )
        dat[[x for x in active_columns if x not in ["timestamp"]]] = (
            self.scaler.transform(
                dat[[x for x in active_columns if x not in ["timestamp"]]]
            )
        )
        # convert timestamp to a sequence that increase by 1 tick every day
        dat["timestamp"] = (
            (dat["timestamp"] - self.time_origin).dt.total_seconds()
            / 3600
            / 24
            * self.daily_time_scale
        )
        return dat

    def inverse_transform(
        self,
        X: Union[pd.DataFrame, np.ndarray, torch.Tensor],
        base_datetime: Optional[pd.Timestamp] = None,
    ):
        if isinstance(dat, torch.Tensor):
            dat = dat.numpy()
        if isinstance(dat, np.ndarray):
            dat = pd.DataFrame(dat, columns=active_columns)
        else:
            dat = X[active_columns].copy()
        dat["price"] = self.price_transformer.inverse_transform(
            dat["price"].values.reshape(-1, 1)
        )
        dat[[x for x in active_columns if x not in ["timestamp"]]] = (
            self.scaler.inverse_transform(
                dat[[x for x in active_columns if x not in ["timestamp"]]]
            )
        )
        if base_datetime is None:
            base_datetime = self.time_origin
        seconds_passed = dat["timestamp"] / self.daily_time_scale * 24 * 3600
        dat["timestamp"] = base_datetime + pd.to_timedelta(seconds_passed, unit="s")
        return dat


def make_sig_prediction_dataset(
    dat: pd.DataFrame,
    interval_len: int,
    skip_step: int,
    interval_split: Union[float, int] = 0.75,
):
    dat_arr = dat.values
    data_windows = sliding_window_view(dat_arr, interval_len, axis=0)[
        ::skip_step
    ]  # (n_windows, n_features, interval_len)
    if isinstance(interval_split, float):
        split_ind = int(data_windows.shape[-1] * interval_split)
    else:
        split_ind = interval_split
    context_windows = data_windows[..., :split_ind]
    pred_windows = data_windows[..., split_ind:]
    ds = TensorDataset(
        torch.tensor(context_windows, dtype=torch.float32).permute(0, 2, 1),
        torch.tensor(pred_windows, dtype=torch.float32).permute(0, 2, 1),
    )
    return ds
