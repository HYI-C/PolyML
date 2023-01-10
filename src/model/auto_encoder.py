# This is the main file for the autoencoder model

from typing import Any, Dict, List, Union

import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn
from torch.optim import Adam
from torchmetrics import Accuracy, MeanAbsoluteError, MetricCollection

from losses. vae_losses import VolumeLoss, CompositeLoss, SumLoss

from nets.embedder import ContinuousEmbedder
from nets.enc_dec import EncDec
from nets.encoder import Encoder
from nets.heads import CompositeHead, VolumeHead

class ClassicModel(pl.LightningModule):  
    """Initialises a ClassicModel instance.

    Args:
        emb (CombinedEmbedder): a module to embed each data point.
        rnn_dim (int): dimensions of the recursive neural net feature vector.
        drop_rate (float): drop out rate in the model. 0 <= drop_rate < 1
        bottleneck_dim (int): dimensions of information bottleneck.
        lr (float): learning rate.
        **kwargs: Additional arguments for the pl.LighteningModule constructor
    """

    def __init__(
        self,
        emb: ContinuousEmbedder,
        rnn_dim: int,
        drop_rate: float,
        bottleneck_dim: int,
        lr: float,
        target_cols: List[str],
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.rnn_dim = rnn_dim
        self.drop_rate = drop_rate
        self.bottleneck_dim = bottleneck_dim
        self.lr = lr
        self.target_cols = target_cols
        self.emb = emb
        self.encoder = Encoder(emb, rnn_dim, drop_rate)
    
        self.head = self.configure_heads(emb)
        self.net = EncDec(
            self.encoder,
            self.head,
        )

        self.criterion = self.configure_criterion()
        self.save_hyperparameters(self.build_parameter_dict())
        self.configure_metrics()

    def build_parameter_dict(self) -> Dict[str, Any]:
        """Return a dictionary of parameters.

        Returns:
            Dict[str, Any]: Parameters of the ClassicModel instance
        """
        return {
            "rnn_dim": self.rnn_dim,
            "drop_rate": self.drop_rate,
            "bottleneck_dim": self.bottleneck_dim,
            "lr": self.lr,
            "target_cols": self.target_cols,
            **self.emb.build_parameter_dict(),
        }

    def configure_optimizers(self):
        """Pytorch lightning automatically calls this to configure the optimizers."""
        return Adam(self.parameters(), lr=self.lr)

    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Conducts a forward pass of this model.

        Args:
            x (Dict[str, torch.Tensor]): batch input

        Returns:
            Dict[str, torch.Tensor]: model output
        """
        pred = self.net.forward(x)
        return pred

    def configure_criterion(self) -> nn.Module:
        """Configures a loss function. This might be a composite function.

        Returns:
            nn.Module: Module that returns loss function on __call__
        """
        pre_loss_ = self.head.loss_function()

        pre_loss = SumLoss(
            {
                "composite": pre_loss_ #...Add other types of losses below here
            }
        )
        return pre_loss

    def configure_heads(self, emb: nn.Module) -> nn.Module:
        head_continuous = {
            f"next_{f}": VolumeHead(self.bottleneck_dim, self.drop_rate) for f in emb.continuous_features
        }

        self.head_map = {**head_continuous} #..add more heads here

        return CompositeHead(self.head_map)

    def configure_metrics(self) -> nn.ModuleDict:
        """Configures metrics for this model.

        Torchmetrics implements metric classes that will be called each step and then automatically compute
        compute metrics on certain events triggered by pytorch lightning. Here we configure these metrics as
        classes, to be called later.

        Returns:
            nn.ModuleDict[str, torchmetrics.Metric]: dictionary with metric instances
        """
        # add MAE for continuous features
        metrics_cont_feat = {n: MeanAbsoluteError() for n in self.emb.continuous_features}


        metrics = MetricCollection({**metrics_cont_feat})

        self.metrics = nn.ModuleDict(
            {
                "train_metrics": metrics.clone(prefix="train_metrics/"),
                "val_metrics": metrics.clone(prefix="val_metrics/"),
            }
        )

    def get_and_log_loss(self, y_pred: Dict[str, torch.Tensor], y_true: Dict[str, Any], split: str) -> torch.Tensor:
        """Calculates and logs loss.

        Args:
            y_pred (Dict[str, torch.Tensor]): model output
            y_true (Dict[str, Any]): batch
            split (str): the split of the step: (i.e. train, val, test)

        Returns:
            torch.Tensor: loss value
        """
        loss, loss_components = self.criterion(y_pred, y_true)
        loss_components = {f"{split}_{name}": loss for name, loss in loss_components.items()}
        self.log_dict(loss_components)
        return loss

    def get_and_log_metrics(
        self, y_pred: Dict[str, torch.Tensor], y_true: Dict[str, Any], split: str
    ) -> Dict[str, torch.Tensor]:
        """Update metric instances with each batch predictions and ground truths.

        Further, log them.

        Args:
            y_pred (Dict[str, torch.Tensor]): model output
            y_true (Dict[str, Any]): the batch
            split (str): the split of the step. (i.e. train, val or test)

        Returns:
            Dict[str, torch.Tensor]: dictionary of metrics for this batch.
        """
        metric_values = {}

        for name in self.emb.continuous_features:
            cont_pred = y_pred[f"next_{name}"][:, 0]
            metric_values[name] = self.metrics[f"{split}_metrics"][name](cont_pred, y_true[name])

        metric_values = {f"{split}_metrics/{k}": v for k, v in metric_values.items()}

        self.log_dict(metric_values)
        return metric_values

    def training_step(self, *args, **kwargs) -> torch.Tensor:
        return self.all_split_step(split="train", *args, **kwargs)

    def validation_step(self, *args, **kwargs) -> torch.Tensor:
        return self.all_split_step(split="val", *args, **kwargs)

    def test_step(self, *args, **kwargs) -> torch.Tensor:
        return self.all_split_step(split="test", *args, **kwargs)

    def all_split_step(
        self, batch: Dict[str, Any], batch_idx: int, split: str
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """As steps in all splits have more in common than divides them, this is a universal step function.

        Args:
            batch (Dict[str, Any]): batch from dataloader
            batch_idx (int): index of the batch per epoch
            split (str): split of the step. ie. train, val or test

        Returns:
            Union[torch.Tensor, Dict[str, torch.Tensor]]: returns loss for training steps and metrics else.
                These values will be fed into callbacks and backward propagation.
        """
        output = self.forward(batch)

        # loss = self.get_and_log_loss(output, target, split)
        loss = self.get_and_log_loss(output, batch, split)
        self.get_and_log_metrics(output, batch, split)

        if split == "train":
            return loss
        else:
            return output

    def load_state_from_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load the model state from a checkpoint.

        Args:
            checkpoint_path (str): Path to the model checkpoint file.
        """
        checkpoint = torch.load(checkpoint_path)
        model_dict = self.state_dict()
        checkpoint_dict = checkpoint["state_dict"]
        pretrained_dict = {k: v for k, v in checkpoint_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)
