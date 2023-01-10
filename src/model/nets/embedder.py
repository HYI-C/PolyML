from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

class ContinuousEmbedder(nn.Module):
    """
    An embedder for continuous features. Is a nn.Module.

    This embeds the continuous features into a latent space, which
    we can then interrogate. 

    Args:
        continuous_features (List[str]): list of continuous features
        category_dict (Dict[str, List]): dictionary of discrete features
        embed_dim (int): embedding dimension
        drop_rate (float): dropout rate. Defaults to ``0.0``.
        pre_encoded (bool): whether to use the input data as is. Defaults to ``False``.
    """

    def __init__(
        self,
        continuous_features: List[str],
        category_dict: Dict[str, List], #..This is the dictionary of discrete features
        embed_dim: int,
        drop_rate: float = 0.0,
        pre_encoded: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.drop_rate = drop_rate
        self.continuous_features = continuous_features
        self.output_shape = [None, embed_dim]
        self.pre_encoded = pre_encoded

        # create the continuous feature encoding, with one hidden layer for good measure
        num_cf = len(self.continuous_features)
        self.c1 = nn.Linear(num_cf, 2 * num_cf) #...The number of continous features should be the number of matrices
        self.c2 = nn.Linear(2 * num_cf, embed_dim) # We want the embed_dim to be the length of the plucker coordinates


    def build_parameter_dict(self) -> Dict[str, Any]:
        """Return a dict of parameters.

        Returns:
            Dict[str, Any]: Parameters of the embedder
        """
        return {
            "embed_dim": self.embed_dim,
            "embedder_drop_rate": self.drop_rate,
            "pre_encoded_features": self.pre_encoded,
        }

    def forward(self, x: Dict[str, torch.Tensor]):
        # batch x num_cont_features
        cf = torch.stack([x[f] for f in self.continuous_features], dim=1)
        cf[cf.isnan()] = 0

        out = F.dropout(F.relu(self.c1(cf)), self.drop_rate, self.training)
        # batch x embed_dim
        out = F.dropout(F.relu(self.c2(out)), self.drop_rate, self.training)
        assert not torch.isnan(out.sum())

        return out
