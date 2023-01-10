import math
from typing import Callable, Dict

import torch
import torch.distributions as d
from torch import nn
from torch.nn import functional as F

from losses.vae_losses import VolumeLoss, CompositeLoss


class BasicHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        drop_rate: float,
        output_dim: int,
        transform: Callable,
        init_norm=False,
    ):
        super().__init__()
        self.drop_rate = drop_rate
        hidden_dim = int(math.sqrt(input_dim * output_dim))
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.transform = transform
        self.input_shape = [None, input_dim]
        self.output_shape = [None, output_dim]

        self.init_norm = init_norm
        self.init_bias = None
        self.init_std = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        This is the decoder, all the heads underneath inherit this
        '''
        eps = 0.00001
        out = F.dropout(F.relu(self.fc1(x)), self.drop_rate, self.training)
        tmp = self.fc2(out)

        if self.init_norm:
            if self.init_bias is None:
                self.init_bias = -tmp.mean(dim=0, keepdims=True).detach()
                self.init_std = tmp.std(dim=0, keepdims=True).detach() + eps
            tmp = (tmp + self.init_bias) / self.init_std
        out = self.transform(tmp)
        return out

class VolumeHead(BasicHead):
    """
    Output is the network for matching Volume
    """

    def __init__(self, input_dim: int, drop_rate: float):
        def normalize(x: torch.Tensor):
            # x[:, 1] = torch.exp(x[:, 1]/10)
            # x[:, 1] = x[:, 1]**2
            return x

        super().__init__(input_dim, drop_rate, 2, normalize)

    def loss_function(self):
        return VolumeLoss()

class CompositeDistribution:
    def __init__(self, distrs: Dict[str, d.Distribution]):
        self.distrs = distrs

    def sample(self, shape) -> Dict[str, torch.Tensor]:
        return {k: d.sample(shape) for k, d in self.distrs}


class CompositeHead(nn.Module):
    def __init__(self, d: Dict[str, nn.Module]):
        super().__init__()
        self.heads = nn.ModuleDict(d)
        self.input_shape = next(iter(d.values())).input_shape

    def forward(self, x):
        return {key: h(x) for key, h in self.heads.items()}

    def loss_function(self, preprocess: Callable) -> CompositeLoss:
        return CompositeLoss(
            {k: h.loss_function() for k, h in self.heads.items() if h.loss_function() is not None},
            preprocess,
        )

    def distribution(self, params: Dict[str, torch.Tensor]) -> CompositeDistribution:
        gens = {k: h.distribution(params[k]) for k, h in self.heads.items()}
        return CompositeDistribution(gens)
