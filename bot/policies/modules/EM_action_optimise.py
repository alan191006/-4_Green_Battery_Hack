import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
import numpy as np
from functools import partial
from typing import Optional, Tuple, Any, Union
from .battery import SeqBatteryEnv


class EMActionModel(pl.LightningModule):
    def __init__(
        self,
        battery: SeqBatteryEnv,
        sequence_length: int,
        batch_size: int,
        increase_beta_per_n_epoch: int = 1,
        beta_min: float = 0.5,
        beta_increment: float = 0.1,
        beta_max: float = 4.0,
        learning_rate: float = 1e-1,
    ):
        super().__init__()
        self.battery = battery
        self.beta = beta_min
        self.beta_min = beta_min
        self.beta_increment = beta_increment
        self.beta_max = beta_max
        self.increase_beta_per_n_epoch = increase_beta_per_n_epoch
        self.batch_size = batch_size
        self.action_sequence = nn.Parameter(torch.zeros(batch_size, sequence_length, 2))
        self.learning_rate = learning_rate

    def reset_beta(self):
        self.beta = self.beta_min

    def hard_beta(self):
        self.beta = None

    def clamp_func(self, x, min_val, max_val, beta):
        if self.beta is not None:
            return (
                min_val + F.softplus(x - min_val, beta) - F.softplus(x - max_val, beta)
            )
        else:
            return torch.clamp(x, min_val, max_val)

    def e_step(
        self, pv: torch.Tensor, pr: torch.Tensor, peak_indicator: torch.Tensor = None
    ):
        grid_action = self.action_sequence[..., 0]
        pv_action = self.action_sequence[..., 1]
        grid_action = F.tanh(grid_action)
        pv_action = F.sigmoid(pv_action)
        battery_trace, costs = self.battery.forward(
            grid_action,
            pv_action,
            pv,
            pr,
            beta=None,
            is_peak_time_if_taxed=peak_indicator,
        )
        return battery_trace, costs

    def m_step(
        self,
        battery_state_minus_init: torch.Tensor,
        pv: torch.Tensor,
        pr: torch.Tensor,
        peak_indicator: torch.Tensor = None,
    ):
        grid_action = self.action_sequence[..., 0]
        pv_action = self.action_sequence[..., 1]
        grid_action = F.tanh(grid_action)
        pv_action = F.sigmoid(pv_action)
        battery_trace, costs = self.battery.static_forward(
            battery_state_minus_init,
            grid_action,
            pv_action,
            pv,
            pr,
            beta=self.beta,
            is_peak_time_if_taxed=peak_indicator,
        )
        return battery_trace, costs

    def forward(
        self, pv: torch.Tensor, pr: torch.Tensor, peak_indicator: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        battery_trace, costs = self.e_step(pv, pr, peak_indicator)
        battery_trace, costs = self.m_step(battery_trace[:, 1:], pv, pr, peak_indicator)
        return battery_trace, costs

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)

    def training_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int,
    ):
        if len(batch) == 2:
            pv_power, price = batch
            peak_indicator
        else:
            pv_power, price, peak_indicator = batch
        battery_trace, costs = self(pv_power, price, peak_indicator)
        loss = costs.mean()
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=False,
        )
        return loss

    def get_action(self) -> Tuple[torch.Tensor, torch.Tensor]:
        grid_action = self.action_sequence[..., 0]
        pv_action = self.action_sequence[..., 1]
        grid_action = F.tanh(grid_action)
        pv_action = F.sigmoid(pv_action)
        return grid_action, pv_action

    def validation_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int,
    ):
        if len(batch) == 2:
            pv_power, price = batch
            peak_indicator
        else:
            pv_power, price, peak_indicator = batch
        battery_trace, costs = self(pv_power, price, peak_indicator)
        loss = costs.mean()
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=False,
        )
        return loss

    def on_train_epoch_end(self) -> None:
        current_epoch = self.current_epoch
        if self.beta is not None:
            if current_epoch % self.increase_beta_per_n_epoch == 0:
                self.beta = min(self.beta + self.beta_increment, self.beta_max)
            if self.beta == self.beta_max:
                self.hard_beta()
        return super().on_train_epoch_end()

    def predict_step(
        self, batch: torch.Tensor, batch_idx: int, dataloader_idx: int = 0
    ):
        return self(batch)
