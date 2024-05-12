from typing import Any, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import signatory
from .transforms import time_aug
import sigkernel
import pytorch_lightning as pl
import torchcde


class SigSeqPredictionModel(pl.LightningModule):
    def __init__(
        self,
        num_features: int,
        num_layers: int,
        hidden_size: int,
        input_len: int,
        output_len: int,
        rbf_sigma: float = 0.5,
        dyadic_order: int = 1,
        max_batch: int = 100,
    ):
        self.summary_network = nn.GRU(
            input_size=num_features - 1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.output_network = nn.GRU(
            input_size=num_features - 1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
