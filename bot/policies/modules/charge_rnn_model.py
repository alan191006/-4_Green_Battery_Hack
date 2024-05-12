import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
import numpy as np
from functools import partial
from typing import Optional, Tuple, Any, Union
from .battery import SeqBatteryEnv


class ChargeRNNModel(pl.LightningModule):
    def __init__(
        self,
        battery: SeqBatteryEnv,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        bidirectional: bool = False,
        increase_beta_per_n_epoch: int = 1,
        beta_min: float = 0.5,
        beta_increment: float = 0.1,
        beta_max: float = 5.0,
        cap_loss_weight: float = 0.01,
        spike_loss_weight: float = 0.1,
    ):
        super().__init__()
        self.rnn = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2),
        )
        self.battery = battery
        self.beta = beta_min
        self.beta_min = beta_min
        self.beta_increment = beta_increment
        self.beta_max = beta_max
        self.increase_beta_per_n_epoch = increase_beta_per_n_epoch
        self.cap_loss_weight = cap_loss_weight
        self.spike_loss_weight = spike_loss_weight

    def reset_beta(self):
        self.beta = self.beta_min

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h0 = torch.zeros(self.rnn.num_layers, x.size(0), self.rnn.hidden_size).to(
            x.device
        )
        out, _ = self.rnn(x, h0)
        out = self.fc(out)
        grid_action = F.tanh(out[..., 0])
        pv_action = F.sigmoid(out[..., 1])
        # pred_capacity = F.sigmoid(out[..., 2])
        return grid_action, pv_action

    def step_forward(
        self, x: torch.Tensor, h0: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out, h1 = self.rnn(x, h0)
        out = self.fc(out)
        grid_action = F.tanh(out[..., 0])
        pv_action = F.sigmoid(out[..., 1])
        return grid_action, pv_action, h1

    def configure_optimizers(self):
        return optim.Adam(self.parameters())

    def training_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int,
    ):
        state, pv_power, price = batch
        grid_action, pv_action = self(state)
        trace, cost = self.battery.forward(
            grid_action, pv_action, pv_power, price, self.beta
        )
        cost_loss = cost.mean()
        cap_loss = -(trace / self.battery.capacity_kWh).std(-1).mean()
        spike_ind = price > price.std() * 2 + price.mean()
        if spike_ind.sum() == 0:
            spike_loss = 0
        else:
            spike_loss = -torch.mean(
                (trace[:, :-1][spike_ind] / self.battery.capacity_kWh - 0) ** 0.5
            )
        loss = (
            cost_loss
            + self.cap_loss_weight * cap_loss
            + self.spike_loss_weight * spike_loss
        )
        self.log(
            "train_cost_loss",
            cost_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "train_cap_loss",
            cap_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "train_spike_loss",
            spike_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def validation_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int,
    ):
        state, pv_power, price = batch
        grid_action, pv_action = self(state)
        trace, cost = self.battery.forward(
            grid_action, pv_action, pv_power, price, None
        )
        cost_loss = cost.mean()
        cap_loss = -(trace / self.battery.capacity_kWh).std(-1).mean()
        spike_ind = price > price.std() * 2 + price.mean()
        if spike_ind.sum() == 0:
            spike_loss = 0
        else:
            spike_loss = -torch.mean(
                (trace[:, :-1][spike_ind] / self.battery.capacity_kWh - 0) ** 0.5
            )
        loss = (
            cost_loss
            + self.cap_loss_weight * cap_loss
            + self.spike_loss_weight * spike_loss
        )
        self.log(
            "val_cost_loss",
            cost_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "val_cap_loss",
            cap_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "val_spike_loss",
            spike_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def on_train_epoch_end(self) -> None:
        current_epoch = self.current_epoch
        if current_epoch % self.increase_beta_per_n_epoch == 0:
            self.beta = min(self.beta + self.beta_increment, self.beta_max)
        return super().on_train_epoch_end()

    # def test_step(
    #     self,
    #     batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    #     batch_idx: int,
    # ):
    #     state, pv_power, price = batch
    #     grid_action, pv_action = self(state)
    #     trace, cost = self.battery.apply_batch_action(
    #         grid_action, pv_action, pv_power, price
    #     )
    #     loss = cost.sum()
    #     self.log("test_loss", loss)
    #     return loss

    def predict_step(
        self, batch: torch.Tensor, batch_idx: int, dataloader_idx: int = 0
    ):
        return self(batch)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self(x)
