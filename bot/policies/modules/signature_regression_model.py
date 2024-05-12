import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from typing import Optional, Tuple, Any, Union
from .battery import BatteryEnv
from .transforms import time_aug, unorm, max_abs_norm
import signatory
from einops.layers.torch import Rearrange


class SigRegModel(pl.LightningModule):
    def __init__(
        self,
        battery: BatteryEnv,
        input_size: int,
        hidden_size: int,
        feature_size: int,
        reg_size: int,
        increase_beta_per_n_epoch: int = 1,
        beta_min: float = 0.5,
        beta_increment: float = 0.1,
        beta_max: float = 10.0,
        sig_depth: int = 3,
        dropout: float = 0.0,
        logsig_history_window: int = 60 // 5 * 24,
        augmenter: Optional[Union[nn.Module, Any]] = None,
        optim_params: Optional[dict] = dict(),
        simultaneous_trade_penalty: float = 0.0,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["battery", "augmenter"])
        # compute the number of signature channels (feature numbers)
        self.gc_dim = feature_size
        self.gc_sig_channels = signatory.signature_channels(self.gc_dim, sig_depth)
        self.gc_logsig_channels = signatory.logsignature_channels(
            self.gc_dim, sig_depth
        )
        self.feature_size = feature_size
        self.sig_depth = sig_depth
        self.logsig_history_window = logsig_history_window
        # 1D convolutions (kernel size = 1 so no time step blending) to extract global and local features
        # "global" features are reduced to feature_size and summarised with signature map
        # representing "state up to now"
        self.global_conv = signatory.Augment(
            input_size,
            (hidden_size, feature_size),
            1,
            include_time=False,
            include_original=False,
        )
        # local features are reduced to hidden_size and directly used in the prediction network for current time step
        self.local_conv = nn.Sequential(
            Rearrange("b t c -> b c t"),
            nn.Conv1d(input_size, hidden_size, 1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_size, hidden_size, 1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            Rearrange("b c t -> b t c"),
        )
        # prediction network
        self.fc = nn.Sequential(
            nn.Linear(
                self.gc_logsig_channels + hidden_size + 1, reg_size
            ),  # +1 for battery state
            nn.ReLU(),
            Rearrange("b t c -> b c t"),
            nn.BatchNorm1d(reg_size),
            Rearrange("b c t -> b t c"),
            nn.Linear(reg_size, reg_size),
            nn.ReLU(),
            nn.Linear(reg_size, 2),
        )
        # self.sig_batch_norm = nn.BatchNorm1d(self.gc_logsig_channels)
        self.battery = battery
        # beta is the "softness" parameter for soft clamp function in battery capacity / charge rate cutoff
        # soft clamping is used to allow gradients to flow through the cutoff points
        self.beta = beta_min
        self.beta_min = beta_min
        self.beta_increment = beta_increment
        self.beta_max = beta_max
        self.increase_beta_per_n_epoch = increase_beta_per_n_epoch

        self.augmenter = augmenter
        self.optim_params = optim_params
        self.simulateneous_trade_penalty = simultaneous_trade_penalty

    def reset_beta(self):
        self.beta = self.beta_min

    def on_fit_start(self):
        self.battery.to(self.device)

    def on_train_start(self):
        self.reset_beta()

    def forward(
        self,
        x: torch.Tensor,
        pv: torch.Tensor,
        pr: torch.Tensor,
        peak_ind: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # compute learned augmentation through 1d convolutions
        x_global_feats = self.global_conv(x)  # (batch, seq_len, gc_dim)
        # conv requires (batch, channels, seq_len) format but x is (batch, seq_len, channels)
        x_local_feats = self.local_conv(x)

        path = signatory.Path(
            x_global_feats, self.sig_depth, basepoint=x_global_feats[:, 0, :]
        )
        # path length is original x_global_feats length + 1 (basepoint)
        # path endpoint is same as path length
        # for zero-based indexing, the endpoint position is index + 2

        # set initial battery state
        if self.training:
            battery_state = self.battery.get_random_initial_state(x.size(0)).to(
                x.device
            )
        else:
            battery_state = self.battery.get_initial_state(x.size(0)).to(x.device)

        # initialize grid actions, pv actions, battery states, costs
        # (pre-allocate for efficiency)
        grid_actions = torch.zeros((x.size(0), x.size(1), 1)).to(x.device)
        pv_actions = torch.zeros((x.size(0), x.size(1), 1)).to(x.device)
        battery_states = torch.zeros((x.size(0), x.size(1), 1)).to(x.device)
        costs = torch.zeros((x.size(0), x.size(1), 1)).to(x.device)
        # loop through time steps (sequence length)
        for i in range(x.size(1)):
            # take one time step of input data
            x_logsig_t = path.logsignature(
                max(0, i + 2 - self.logsig_history_window), i + 2
            )  # (batch, logsig_dim)

            x_logsig_t = x_logsig_t[:, None, :]
            x_local_t = x_local_feats[:, [i], :]

            pv_t = pv[:, [i]]
            pr_t = pr[:, [i]]

            # combine signature features, local features, and battery state
            x_t = torch.cat(
                [
                    x_logsig_t,
                    x_local_t,
                    battery_state[:, None, :] / self.battery.capacity_kWh,
                ],
                dim=-1,
            )
            # pass through prediction network and get grid action, pv action
            z_t = self.fc(x_t)
            grid_action, pv_action = z_t[..., 0], z_t[..., 1]
            grid_action = F.tanh(grid_action)
            pv_action = F.sigmoid(pv_action)
            # apply battery action and get new battery state and charging cost (or exporting profit)
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
        path: Optional[signatory.Path] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x_global_feats = self.global_conv(x_t)  # (batch, seq_len[1], gc_dim)
        x_local_feats = self.local_conv(x_t)

        if path is None:
            path = signatory.Path(
                x_global_feats, self.sig_depth, basepoint=x_global_feats[:, 0, :]
            )
        else:
            # update path with new global features
            path.update(x_global_feats)
        x_current_logsig = path.logsignature(
            start=max(0, path.size(1) - self.logsig_history_window)
        )
        x_t = torch.cat(
            [
                x_current_logsig[:, None, :],
                x_local_feats,
                battery_state[:, None, :] / self.battery.capacity_kWh,
            ],
            dim=-1,
        )  # (batch, seq_len[1], logsig_dim + hidden_size + 1)
        z_t = self.fc(x_t)

        grid_action, pv_action = z_t[..., 0], z_t[..., 1]
        grid_action = F.tanh(grid_action)
        pv_action = F.sigmoid(pv_action)

        return grid_action, pv_action, path

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
        dataloader_idx: int,
    ):
        grid_actions, pv_actions, battery_states, costs = self(*batch)
        loss = costs.sum(1).mean()
        loss += self.simulateneous_trade_penalty * self.penalty(
            grid_actions, pv_actions
        )
        if dataloader_idx == 0:
            self.log(
                "val_loss_sec",
                loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                add_dataloader_idx=False,
            )
        elif dataloader_idx == 1:
            self.log(
                "val_loss",
                loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                add_dataloader_idx=False,
            )
        return loss

    def on_train_epoch_end(self) -> None:
        current_epoch = self.current_epoch
        if (current_epoch % self.increase_beta_per_n_epoch == 0) & (
            self.beta is not None
        ):
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
