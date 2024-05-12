import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from typing import Optional, Tuple, Any, Union
from .battery import SeqBatteryEnv


class SimpleStatelessModel(pl.LightningModule):
    def __init__(
        self,
        battery: SeqBatteryEnv,
        input_size: int,
        hidden_size: int,
        increase_beta_per_n_epoch: int = 1,
        beta_min: float = 0.5,
        beta_increment: float = 0.1,
        beta_max: float = 5.0,
        dropout: float = 0.25,
        augmenter: Optional[Union[nn.Module, Any]] = None,
        optim_params: Optional[dict] = dict(),
        simulateneous_trade_penalty: float = 0.0,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["battery", "augmenter"])
        self.fc = nn.Sequential(
            nn.Conv1d(input_size, hidden_size, 1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_size, hidden_size, 1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Conv1d(hidden_size, 4, 1),
        )
        self.battery = battery
        self.beta = beta_min
        self.beta_min = beta_min
        self.beta_increment = beta_increment
        self.beta_max = beta_max
        self.increase_beta_per_n_epoch = increase_beta_per_n_epoch
        self.augmenter = augmenter
        self.optim_params = optim_params
        self.simulateneous_trade_penalty = simulateneous_trade_penalty

    def reset_beta(self):
        self.beta = self.beta_min

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.fc(x.permute(0, 2, 1)).permute(0, 2, 1)
        pv_action = F.sigmoid(out[..., 0])
        grid_action = F.softmax(out[..., 1:], -1)
        grid_action = grid_action[..., 0] - grid_action[..., 2]
        return grid_action, pv_action

    def configure_optimizers(self):
        opt = optim.Adam(self.parameters(), **self.optim_params)
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    opt, T_0=10, T_mult=1, eta_min=1e-6, last_epoch=-1
                ),
                "interval": "step",
                "frequency": 5,
            },
        }

    def training_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int,
    ):
        state, pv_power, price, peak_ind = batch
        grid_actions, pv_actions = self(state)
        trace, cost = self.battery.forward(
            grid_actions,
            pv_actions,
            pv_power,
            price,
            beta=self.beta,
            random_initial_state=True,
            is_peak_time_if_taxed=peak_ind,
        )
        loss = cost.sum(-1).mean()

        if self.simulateneous_trade_penalty > 0:
            loss += (
                self.simulateneous_trade_penalty
                * torch.where(grid_actions < 0, -pv_actions * grid_actions.detach(), 0)
                .sum(-1)
                .mean()
            )
            loss += (
                self.simulateneous_trade_penalty
                * torch.where(
                    grid_actions > 0, (1 - pv_actions) * grid_actions.detach(), 0
                )
                .sum(-1)
                .mean()
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
        state, pv_power, price, peak_ind = batch
        grid_actions, pv_actions = self(state)
        trace, cost = self.battery.forward(
            grid_actions,
            pv_actions,
            pv_power,
            price,
            beta=None,
            is_peak_time_if_taxed=peak_ind,
        )
        loss = cost.sum(-1).mean()
        if self.simulateneous_trade_penalty > 0:
            loss += (
                self.simulateneous_trade_penalty
                * torch.where(grid_actions < 0, -pv_actions * grid_actions.detach(), 0)
                .sum(-1)
                .mean()
            )
            loss += (
                self.simulateneous_trade_penalty
                * torch.where(
                    grid_actions > 0, (1 - pv_actions) * grid_actions.detach(), 0
                )
                .sum(-1)
                .mean()
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
