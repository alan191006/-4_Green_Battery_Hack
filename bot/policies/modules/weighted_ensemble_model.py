import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from typing import Optional, Tuple, Any, Union
from .battery import SeqBatteryEnv


class WeightedEnsembleModel(pl.LightningModule):
    def __init__(
        self,
        battery: SeqBatteryEnv,
        num_child_predictors: int,
        increase_beta_per_n_epoch: int = 1,
        beta_min: float = 0.5,
        beta_increment: float = 0.1,
        beta_max: float = 5.0,
        lr: float = 1e-1,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["battery", "augmenter"])
        self.alpha = nn.Parameter(torch.ones(num_child_predictors))
        self.battery = battery
        self.beta = beta_min
        self.beta_min = beta_min
        self.beta_increment = beta_increment
        self.beta_max = beta_max
        self.increase_beta_per_n_epoch = increase_beta_per_n_epoch
        self.lr = lr

    def reset_beta(self):
        self.beta = self.beta_min

    def forward(
        self, child_grid_actions: torch.Tensor, child_pv_actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # input shape (batch_size, seq_len, num_child_predictors)
        # NOTE: even when passing one item at a time, seq_len=1 must be a dimension!
        grid_action = torch.sum(
            F.softmax(self.alpha, -1)[None, None, :] * child_grid_actions, -1
        )
        pv_action = torch.sum(
            F.softmax(self.alpha, -1)[None, None, :] * child_pv_actions, -1
        )
        return grid_action, pv_action

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)

    def training_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int,
    ):
        # NOTE: child_grid and child_pv are the outputs of the child predictors
        # corresponding dataloader needs to match this format
        child_grid, child_pv, pv_power, price, peak_ind = batch
        grid_action, pv_action = self(child_grid, child_pv)
        trace, cost = self.battery.forward(
            grid_action,
            pv_action,
            pv_power,
            price,
            beta=self.beta,
            random_initial_state=True,
            is_peak_time_if_taxed=peak_ind,
        )
        loss = cost.sum(-1).mean()
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
        child_grid, child_pv, pv_power, price, peak_ind = batch
        grid_action, pv_action = self(
            child_grid,
            child_pv,
        )
        trace, cost = self.battery.forward(
            grid_action,
            pv_action,
            pv_power,
            price,
            beta=None,
            is_peak_time_if_taxed=peak_ind,
        )
        loss = cost.sum(-1).mean()
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
        child_grid, child_pv = batch[0], batch[1]
        return self(child_grid, child_pv)
