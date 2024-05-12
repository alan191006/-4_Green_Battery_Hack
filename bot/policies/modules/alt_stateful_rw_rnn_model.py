import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
import numpy as np
from functools import partial
from typing import Optional, Tuple, Any, Union
from .battery import BatteryEnv
import signatory


class AltStatefulRWRNNModel(pl.LightningModule):
    def __init__(
        self,
        battery: BatteryEnv,
        input_size: int,
        hidden_size: int,
        fc_size: int,
        num_encoder_layers: int = 1,
        dropout: float = 0.0,
        bidirectional: bool = False,
        increase_beta_per_n_epoch: int = 1,
        beta_min: float = 0.5,
        beta_increment: float = 0.1,
        beta_max: float = 5.0,
        augmenter: Optional[Union[nn.Module, Any]] = None,
        optim_params: Optional[dict] = dict(),
        simultaneous_trade_penalty: float = 0.0,
        critical_dimensions: Tuple[int, int, int, int] = (0, 1, 3),
        history_windows: Tuple[int, int, int] = (31, 222, 247),
        sig_depth: int = 3,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["battery", "augmenter"])

        self.battery = battery
        self.beta = beta_min
        self.beta_min = beta_min
        self.beta_increment = beta_increment
        self.beta_max = beta_max
        self.increase_beta_per_n_epoch = increase_beta_per_n_epoch

        self.augmenter = augmenter
        self.optim_params = optim_params
        self.simulateneous_trade_penalty = simultaneous_trade_penalty
        self.critical_dimensions = critical_dimensions
        self.history_windows = history_windows
        self.sig_depth = sig_depth
        self.sig_channels = signatory.logsignature_channels(
            len(critical_dimensions), sig_depth
        )

        self.encoder = nn.GRU(
            input_size=input_size
            + self.sig_channels * len(self.history_windows)
            + 1,  # +1 for battery state
            hidden_size=hidden_size,
            num_layers=num_encoder_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, fc_size),
            nn.ReLU(),
            nn.Linear(fc_size, 3),
        )

    def reset_beta(self):
        self.beta = self.beta_min

    def on_fit_start(self):
        self.battery.to(self.device)

    def on_train_start(self):
        self.reset_beta()

    def get_initial_h(self, batch_size: int) -> torch.Tensor:
        return torch.zeros(
            self.encoder.num_layers, batch_size, self.encoder.hidden_size
        ).to(self.device)

    def forward(
        self,
        x: torch.Tensor,
        pv: torch.Tensor,
        pr: torch.Tensor,
        peak_ind: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h0 = torch.zeros(
            self.encoder.num_layers, x.size(0), self.encoder.hidden_size
        ).to(x.device)
        x_critical = x[:, :, self.critical_dimensions]
        path = signatory.Path(x_critical, self.sig_depth, basepoint=True)

        if self.training:
            battery_state = self.battery.get_random_initial_state(x.size(0)).to(
                x.device
            )
        else:
            battery_state = self.battery.get_initial_state(x.size(0)).to(x.device)
        # grid_actions, pv_actions, battery_states, costs = [], [], [], []
        # (pre-allocate for efficiency)
        grid_actions = torch.zeros((x.size(0), x.size(1), 1)).to(x.device)
        pv_actions = torch.zeros((x.size(0), x.size(1), 1)).to(x.device)
        battery_states = torch.zeros((x.size(0), x.size(1), 1)).to(x.device)
        costs = torch.zeros((x.size(0), x.size(1), 1)).to(x.device)
        for i in range(x.size(1)):
            x_t = x[:, i : i + 1]  # (batch_size, 1, input_size)

            logsig_wins = torch.cat(
                [
                    path.logsignature(max(0, i + 2 - w), i + 2)[:, None, :]
                    for w in self.history_windows
                ],
                dim=-1,
            )  # (batch_size, 1, sig_channels * len(history_windows))

            pv_t = pv[:, i : i + 1]
            pr_t = pr[:, i : i + 1]

            x_t = torch.cat(
                [
                    x_t,
                    logsig_wins,
                    battery_state[:, None, :] / self.battery.capacity_kWh,
                ],
                dim=-1,
            )
            z_t, h0 = self.encoder(x_t, h0)
            out = self.fc(z_t)
            grid_action, pv_action, hold_gate = out[..., 0], out[..., 1], out[..., 2]
            hold_gate = F.sigmoid(hold_gate)
            grid_action = F.tanh(grid_action) * (1 - hold_gate)
            pv_action = F.sigmoid(pv_action) * hold_gate + torch.where(
                grid_action > 0, 1.0, 0.0
            ) * (1 - hold_gate)
            battery_state, cost = self.battery(
                battery_state,
                grid_action,
                pv_action,
                pv_t,
                pr_t,
                beta=self.beta,
                is_peak_time_if_taxed=peak_ind[:, i],
            )
            # accumulate actions, states, costs
            grid_actions[:, i, :] += grid_action
            pv_actions[:, i, :] += pv_action
            battery_states[:, i, :] += battery_state
            costs[:, i, :] += cost

        return grid_actions, pv_actions, battery_states, costs

    def step_forward(
        self,
        battery_state: torch.Tensor,
        x_t: torch.Tensor,
        h0: torch.Tensor,
        path: Optional[signatory.Path] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x_critical_t = x_t[:, :, self.critical_dimensions]
        if path is None:
            path = signatory.Path(x_critical_t, self.sig_depth, basepoint=True)
        else:
            path.update(x_critical_t)
        logsig_wins = torch.cat(
            [
                path.logsignature(start=max(0, path.size(1) - w))[:, None, :]
                for w in self.history_windows
            ],
            dim=-1,
        )  # (batch_size, 1, sig_channels * len(history_windows))

        x_t = torch.cat(
            [x_t, logsig_wins, battery_state[:, None, :] / self.battery.capacity_kWh],
            dim=-1,
        )
        z_t, h0 = self.encoder(x_t, h0)
        out = self.fc(z_t)
        grid_action, pv_action, hold_gate = out[..., 0], out[..., 1], out[..., 2]
        hold_gate = F.sigmoid(hold_gate)
        grid_action = F.tanh(grid_action) * (1 - hold_gate)
        pv_action = F.sigmoid(pv_action) * hold_gate + torch.where(
            grid_action > 0, 1.0, 0.0
        ) * (1 - hold_gate)
        return grid_action, pv_action, battery_state, h0

    def configure_optimizers(self):
        opt = optim.AdamW(self.parameters(), **self.optim_params)
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": optim.lr_scheduler.ReduceLROnPlateau(
                    opt, mode="min", factor=0.5, patience=5, min_lr=1e-6
                ),
                "monitor": "train_loss",
                "interval": "epoch",
            },
        }

    def penalty(self, grid_actions, pv_actions):
        if self.simulateneous_trade_penalty > 0:
            penalty = (
                torch.where(grid_actions < 0, -pv_actions * grid_actions, 0)
                .sum(-1)
                .mean()
            ) + (
                torch.where(grid_actions > 0, (1 - pv_actions) * grid_actions, 0)
                .sum(-1)
                .mean()
            )
            tag = "train" if self.training else "val"
            self.log(
                f"{tag}_ST_penalty",
                penalty,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
            return penalty
        return 0

    def training_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int,
    ):
        grid_actions, pv_actions, battery_states, costs = self(*batch)
        loss = costs.sum(1).mean()
        loss += self.simulateneous_trade_penalty * self.penalty(
            grid_actions, pv_actions
        )
        self.log(
            "train_loss",
            loss,
            on_step=True,
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
        grid_actions, pv_actions, battery_states, costs = self(*batch)
        loss = costs.sum(1).mean()
        loss += self.simulateneous_trade_penalty * self.penalty(
            grid_actions, pv_actions
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

    def on_predict_start(self):
        self.beta = None

    def predict_step(
        self, batch: torch.Tensor, batch_idx: int, dataloader_idx: int = 0
    ):
        return self(*batch)

    def on_before_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        if (self.training) & (self.augmenter is not None):
            state, pv_power, price, peak_ind = batch
            state = self.augmenter(state)
            batch = (state, pv_power, price, peak_ind)
        return batch
    
    def test_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int,
    ):
        grid_actions, pv_actions, battery_states, costs = self(*batch)
        loss = costs.sum(1).mean()
        self.log(
            "test_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss