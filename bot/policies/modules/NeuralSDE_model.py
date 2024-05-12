from typing import Any, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import signatory
import sigkernel
import pytorch_lightning as pl
import torchsde
from .transforms import time_aug


class SDE(torch.nn.Module):
    noise_type = "diagonal"
    sde_type = "stratonovich"

    def __init__(self, state_size: int, hidden_size: int, g_scale: float = 0.3):
        super(SDE, self).__init__()
        self.state_size = state_size
        self.hidden_size = hidden_size
        self.g_scale = g_scale

        self.drift = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, state_size),
            nn.Tanh(),
        )
        self.diffusion = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, state_size),
            nn.Tanh(),
        )

    def f(self, t, y):
        # day_encoding = torch.cat(
        #     [
        #         torch.sin(2 * torch.pi * t),
        #         torch.cos(2 * torch.pi * t),
        #     ],
        #     dim=-1,
        # )  # (batch, time, 2)
        # week_encoding = torch.cat(
        #     [
        #         torch.sin(2 * torch.pi * t / 7),
        #         torch.cos(2 * torch.pi * t / 7),
        #     ],
        #     dim=-1,
        # )
        # time_encoding = torch.cat(
        #     [
        #         day_encoding,
        #         week_encoding,
        #     ],
        #     dim=-1,
        # )
        # y = torch.cat([y, time_encoding], dim=-1)
        return self.drift(y)

    def g(self, t, y):
        return self.diffusion(y) * self.g_scale


class NeuralSDE(pl.LightningModule):
    def __init__(
        self,
        augment_channels: int,
        sig_depth: int,
        state_size: int,
        hidden_size: int,
        num_features: Optional[int] = None,
        g_scale: float = 1,
        sig_loss_weight: float = 0.1,
    ):
        super(NeuralSDE, self).__init__()
        if num_features is None:
            num_features = state_size

        self.save_hyperparameters()
        self.sig_summarise = nn.Sequential(
            signatory.Augment(num_features, (hidden_size, augment_channels), 6),
            signatory.Signature(depth=sig_depth),
            nn.LazyLinear(hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.LazyLinear(hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, hidden_size),
        )
        self.x1_summarise = nn.Sequential(
            nn.LazyLinear(state_size),
            nn.ReLU(),
            nn.Linear(state_size, state_size),
        )
        self.joint_summarise = nn.Sequential(
            nn.LazyLinear(hidden_size + state_size),
            nn.ReLU(),
            nn.Linear(hidden_size + state_size, state_size),
        )
        self.sde = SDE(state_size, hidden_size, g_scale)
        self.output = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_features),
        )
        self.state_size = state_size
        self.hidden_size = hidden_size
        self.g_scale = g_scale
        self.sig_loss_weight = sig_loss_weight
        # use sig kernel
        self.static_kernel = sigkernel.RBFKernel(sigma=0.5)
        self.signature_kernel = sigkernel.SigKernel(
            static_kernel=self.static_kernel, dyadic_order=1
        )
        self.signature_map = signatory.Signature(depth=sig_depth)

    def summarise(self, x: torch.Tensor) -> torch.Tensor:
        # x dims: (batch, time, features)
        sig_features = self.sig_summarise(x)
        x1_features = self.x1_summarise(x[:, -1, :])
        state = self.joint_summarise(torch.cat([sig_features, x1_features], dim=-1))
        return state

    def forward(self, x_prev: torch.Tensor, ts: torch.Tensor) -> torch.Tensor:
        z0 = self.summarise(x_prev)
        z1 = torchsde.sdeint_adjoint(
            self.sde,
            z0,
            ts,
            method="reversible_heun",
            adjoint_method="adjoint_reversible_heun",
            dt=(ts[1] - ts[0]) / 10,
        ).permute(
            1, 0, 2
        )  # (batch, time, state)
        x_hat = self.output(z1)
        return x_hat

    def training_step(self, batch, batch_idx):
        x_prev, x_next = batch
        ts = torch.linspace(0, 1, x_next.shape[1])
        # ts = x_next[:, :, 0]
        x_hat = self(x_prev[..., 1:], ts)  # (batch, time, features)
        # loss = F.mse_loss(x_hat, x_next)
        x_hat_w_time = torch.cat(
            [
                x_next[..., [0]],
                x_hat,
            ],
            dim=-1,
        )
        match_loss = F.mse_loss(x_hat, x_next[..., 1:])
        # match_loss = self.signature_kernel.compute_distance(
        #     x_hat_w_time, x_next
        # ).mean()
        # match_loss = F.mse_loss(self.signature_map(x_hat_w_time), self.signature_map(x_next))
        if self.sig_loss_weight == 0:
            loss = match_loss
        else:
            loss = match_loss + self.sig_loss_weight * F.mse_loss(
                self.signature_map(x_hat_w_time),
                self.signature_map(x_next).mean(-1, keepdim=True),
            ) / signatory.signature_channels(
                x_hat_w_time.size(-1), self.signature_map.depth
            )
            loss = match_loss + self.sig_loss_weight * F.mse_loss(
                self.signature_map(x_hat_w_time),
                self.signature_map(x_next).mean(-1, keepdim=True),
            ) / signatory.signature_channels(
                x_hat_w_time.size(-1), self.signature_map.depth
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

    def validation_step(self, batch, batch_idx):
        x_prev, x_next = batch
        ts = torch.linspace(0, 1, x_next.shape[1])
        # ts = x_next[:, :, 0]
        x_hat = self(x_prev[..., 1:], ts)
        # loss = F.mse_loss(x_hat, x_next)
        x_hat_w_time = torch.cat(
            [
                x_next[..., [0]],
                x_hat,
            ],
            dim=-1,
        )
        match_loss = F.mse_loss(x_hat, x_next[..., 1:])
        # match_loss = self.signature_kernel.compute_distance(x_hat_w_time, x_next).mean()
        # match_loss = F.mse_loss(self.signature_map(x_hat_w_time), self.signature_map(x_next))
        if self.sig_loss_weight == 0:
            loss = match_loss
        else:
            sig_loss = (
                self.sig_loss_weight
                * F.mse_loss(
                    self.signature_map(x_hat_w_time),
                    self.signature_map(x_next).mean(-1, keepdim=True),
                )
                / signatory.signature_channels(
                    x_hat_w_time.size(-1), self.signature_map.depth
                )
            )
            loss = match_loss + sig_loss * self.sig_loss_weight
            self.log(
                "val_sig_loss",
                sig_loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
        self.log(
            "val_match_loss",
            match_loss,
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

    def predict(self, x_prev: torch.Tensor, ts: torch.Tensor) -> torch.Tensor:
        return self(x_prev, ts)

    def predict_step(self, batch, batch_idx):
        x_prev, x_next = batch
        ts = torch.linspace(0, 1, x_next.shape[1])
        x_hat = self(x_prev[..., 1:], ts)
        return x_hat
