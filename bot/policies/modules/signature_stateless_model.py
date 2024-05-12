import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from typing import Optional, Tuple, Any, Union
from .battery import SeqBatteryEnv
from .transforms import time_aug, unorm, max_abs_norm
import signatory


class SigStatelessModel(pl.LightningModule):
    def __init__(
        self,
        battery: SeqBatteryEnv,
        input_size: int,
        hidden_size: int,
        feature_size: int,
        reg_size: int,
        increase_beta_per_n_epoch: int = 1,
        beta_min: float = 0.5,
        beta_increment: float = 0.1,
        beta_max: float = 5.0,
        sig_depth: int = 3,
        dropout: float = 0.25,
        logsig_history_window: int = 60 // 5 * 24,
        augmenter: Optional[Union[nn.Module, Any]] = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["battery", "augmenter"])
        # compute the number of signature channels (feature numbers)
        self.gc_dim = feature_size + 1  # (time augmentation is the extra feature)
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
        self.global_conv = nn.Sequential(
            signatory.Augment(
                input_size,
                (hidden_size, feature_size),
                1,
                include_time=False,
                include_original=False,
            ),
        )
        # local features are reduced to hidden_size and directly used in the prediction network for current time step
        self.local_conv = nn.Sequential(
            nn.Conv1d(input_size, hidden_size, 1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_size, hidden_size, 1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
        )
        # prediction network
        self.fc = nn.Sequential(
            nn.Linear(self.gc_logsig_channels + hidden_size, hidden_size),
            nn.ReLU(),
            # nn.BatchNorm1d(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, reg_size),
            nn.ReLU(),
            # nn.BatchNorm1d(reg_size),
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
        self.expanding_logsig_map = signatory.LogSignature(depth=sig_depth, stream=True)

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
        grid_actions, pv_actions = self.predict(x)
        battery_trace, costs = self.battery.forward(
            grid_actions,
            pv_actions,
            pv,
            pr,
            beta=None,
            random_initial_state=True,
            is_peak_time_if_taxed=peak_ind,
        )

        return grid_actions, pv_actions, battery_trace, costs

    def predict(self, x: torch.Tensor):
        # compute learned augmentation through 1d convolutions
        x_global_feats = self.global_conv(x)  # (batch, seq_len, gc_dim)
        # conv requires (batch, channels, seq_len) format but x is (batch, seq_len, channels)
        x_local_feats = self.local_conv(x.permute(0, 2, 1)).permute(0, 2, 1)

        # augment time ticks to global features
        time_ticks = torch.arange(0, x.size(1) * (1 / 12), 1 / 12).to(x.device)
        x_global_feats = torch.cat(
            [
                time_ticks[None, :, None].expand(x_global_feats.size(0), -1, -1),
                x_global_feats,
            ],
            dim=-1,
        )

        x_expanding_logsig = self.expanding_logsig_map(
            x_global_feats, basepoint=x_global_feats[:, 0, :]
        )  # (batch, seq_len, logsig_dim)
        x_expanding_logsig = max_abs_norm(x_expanding_logsig, -1)

        combined_features = torch.cat(
            [
                x_expanding_logsig,  # (batch, seq_len, logsig_dim)
                x_local_feats,  # (batch, seq_len, hidden_size)
            ],
            dim=-1,
        )  # (batch, seq_len, logsig_dim + hidden_size + 1)
        actions = self.fc(combined_features)  # (batch, seq_len, 2)
        grid_actions, pv_actions = actions[..., 0], actions[..., 1]
        grid_actions = F.tanh(grid_actions)
        pv_actions = F.sigmoid(pv_actions)

        return grid_actions, pv_actions

    def step_forward(
        self,
        battery_state: torch.Tensor,
        x_t: torch.Tensor,
        path: Optional[signatory.Path] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x_global_feats = self.global_conv(x_t)  # (batch, seq_len[1], gc_dim)
        x_local_feats = self.local_conv(x_t.permute(0, 2, 1)).permute(0, 2, 1)

        if path is None:
            current_time_tick = torch.zeros(1).to(x_t.device)
            x_global_feats = torch.cat(
                [
                    current_time_tick[None, :, None].expand(
                        x_global_feats.size(0), -1, -1
                    ),
                    x_global_feats,
                ],
                dim=-1,
            )
            path = signatory.Path(x_global_feats, self.sig_depth, basepoint=True)
        else:
            # time augmentation
            current_time_tick = torch.tensor([(path.size(1) - 2) * (1 / 12)]).to(
                x_t.device
            )
            x_global_feats = torch.cat(
                [
                    current_time_tick[None, :, None].expand(
                        x_global_feats.size(0), -1, -1
                    ),
                    x_global_feats,
                ],
                dim=-1,
            )
            # update path with new global features
            path.update(x_global_feats)
        x_current_logsig = max_abs_norm(path.logsignature(), -1)
        x_t = torch.cat(
            [
                x_current_logsig[:, None, :],
                x_local_feats,
                battery_state[:, None, :] / self.battery.capacity_kWh,
            ],
            dim=-1,
        )  # (batch, seq_len[1], logsig_dim + hidden_size + 1)
        z_t = self.fc(x_t[:, 0, :])[:, None, :]

        grid_action, pv_action = z_t[..., 0], z_t[..., 1]
        grid_action = F.tanh(grid_action)
        pv_action = F.sigmoid(pv_action)

        return grid_action, pv_action, path

    def configure_optimizers(self):
        return optim.Adam(self.parameters())

    def training_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int,
    ):
        grid_actions, pv_actions, battery_states, costs = self(*batch)
        loss = costs.sum(1).mean()
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
        return self(*batch) if len(batch) == 4 else self.predict(batch)

    def on_before_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        if (self.training) & (self.augmenter is not None):
            state, pv_power, price, peak_ind = batch
            state = self.augmenter(state)
            batch = (state, pv_power, price, peak_ind)
        return batch
