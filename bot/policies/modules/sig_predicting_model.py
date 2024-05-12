from typing import Any, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import signatory
import pytorch_lightning as pl
from .transforms import time_aug, to_leadlag, max_abs_norm, unorm


class SigPred(pl.LightningModule):
    def __init__(
        self,
        num_features: int,
        target_features: int,
        augment_channels: int,
        sig_depth: int,
        target_sig_depth: int,
        conv_channels: int,
        hidden_size: int,
    ):
        super(SigPred, self).__init__()
        self.save_hyperparameters()
        self.hidden_size = hidden_size
        self.conv_channels = conv_channels
        self.target_sig_depth = target_sig_depth
        self.num_features = num_features
        self.target_features = target_features
        self.sig_channels = signatory.signature_channels(
            num_features + augment_channels + 1, sig_depth
        )
        self.target_sig_channels = signatory.signature_channels(
            target_features, target_sig_depth
        )

        self.rolling_sig = nn.Sequential(
            signatory.Augment(num_features, (conv_channels, augment_channels), 6),
            signatory.Signature(depth=sig_depth, stream=True),
        )
        self.conv_layers = nn.Sequential(
            nn.Conv1d(self.sig_channels, conv_channels, 6),
            nn.ReLU(),
            nn.Conv1d(conv_channels, hidden_size * 2, 4),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
        )

        self.output_layers = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.target_sig_channels),
        )

        self.signature_map = signatory.Signature(target_sig_depth)

    def forward(self, x_prev: torch.Tensor) -> torch.Tensor:
        x_rolling_sig = self.rolling_sig(x_prev)  # (batch, time, sig_channels)
        x_conv = self.conv_layers(
            x_rolling_sig.permute(0, 2, 1)
        )  # (batch, conv_channels, 1)
        sig_hat = self.output_layers(x_conv.squeeze(-1))  # (batch, target_sig_channels)
        return sig_hat

    def training_step(self, batch, batch_idx):
        x_prev, x_next = batch
        sig_hat = self(x_prev[..., 1:])
        x_next_path = to_leadlag(x_next[..., [1]])
        assert x_next_path.shape[-1] == self.target_features
        x_next_sig = unorm(self.signature_map(x_next_path))
        loss = F.mse_loss(sig_hat, x_next_sig, reduction="sum")

        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x_prev, x_next = batch
        sig_hat = self(x_prev[..., 1:])
        x_next_path = to_leadlag(x_next[..., [1]])
        assert x_next_path.shape[-1] == self.target_features
        x_next_sig = unorm(self.signature_map(x_next_path))
        loss = F.mse_loss(sig_hat, x_next_sig, reduction="sum")

        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=1e-3)
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer=opt,
                    mode="min",
                    factor=0.5,
                    patience=5,
                ),
                "monitor": "train_loss",
            },
        }

    def predict(self, x_prev: torch.Tensor) -> torch.Tensor:
        return self(x_prev)

    def predict_step(self, batch, batch_idx):
        x_prev, x_next = batch
        sig_hat = self(x_prev[..., 1:])
        return sig_hat
