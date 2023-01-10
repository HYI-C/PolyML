import datetime
from typing import Any, Dict, List, Sequence, Union

import numpy as np
import torch
from torch import nn

# from event_model.utils import OrdinalEncoderWithUnknown #..This is for encoding discrete events
#.. I wonder if we actually still need this module because this is to create
#targets for the sequence of continuous features.
from model.nets.embedder import ContinuousEmbedder 


class DummyTransform:
    def __call__(self, x, *args, **kwargs):
        return x

    def output_len(self, input_len: int):
        return input_len


class TargetCreator(nn.Module):
    """
    A class to create targets for a sequence of vertices. Is an instance of nn.Module.

    Args:
        cols (List[str]): The list of columns to use as features.
        enc (Dict[str, OrdinalEncoderWithUnknown]): The encoders to use for each feature.
        max_item_len (int): The maximum length of the sequence. Defaults to ``100``.
        pre_encode_features (bool): Whether to pre-encode the features. Defaults to ``False``.
        start_token_discr (str): The token to use for the start of a discrete feature. Defaults to ``"StartToken"``.
        start_token_cont (int): The token to use for the start of a continuous feature. Defaults to ``-1e6``.

    Attributes:
        enc (Dict[str, OrdinalEncoderWithUnknown]): The encoders to use for each feature.
        cols (List[str]): The list of columns to use as features.
        t_name (str): The name of the time feature. Is always ``"t"``.
        max_item_len (int): The maximum length of the sequence.
        start_token_discr (str): The token to use for the start of a discrete feature.
        start_token_cont (int): The token to use for the start of a continuous feature.
    """

    def __init__(
        self,
        cols: List[str],
        emb: ContinuousEmbedder,
        max_item_len: int = 100,
        start_token_cont: int = -1e6,
    ):
        super().__init__()
        # self.enc = emb.enc #...This is for continuous feature embedding, we don't need this
        self.cols = cols
        self.t_name = "t"
        self.max_item_len = max_item_len
        self.pre_encode_features = emb.pre_encoded
        self.start_token_cont = start_token_cont

    def build_parameter_dict(self) -> Dict[str, Any]:
        """Return a dictionary of parameters.

        Returns:
            Dict[str, Any]: A dictionary of the target transform parameters
        """
        return {
            "columns": self.cols,
            "time_column": self.t_name,
            "max_item_len": self.max_item_len,
            "pre_encode_features": self.pre_encode_features,
            "start_token_continuous": self.start_token_cont,
        }

    def output_len(self, input_dim: int) -> int:
        """
        Returns the length of the output of the model.

        Args:
            input_dim (int): The length of the input.

        Returns:
            int: The length of the output.
        """
        return min(self.max_item_len, input_dim)


    def __call__(
        self,
        x: Dict[str, np.ndarray],
        asof_time: datetime.date,
    ) -> Dict[str, Union[torch.Tensor, Sequence[str]]]:

        # trim the too long sequences
        x = {k: v[-(self.max_item_len) :] for k, v in x.items()}

        x_token = x.copy()
        x_token["dt"] = np.append([self.start_token_cont, 0], x_token["t"][1:] - x_token["t"][:-1]).astype(np.float32)
        x_token["t"] = x_token["t"].astype(np.float32)

        for k, v in x_token.items():
            if k == "dt":
                continue
            if k in self.cols:
                x_token[k] = np.append([self.start_token_cont], v)
            else:
                x_token[k] = np.append([None], v)

        x_out = x_token.copy()
        x_out["t_to_now"] = asof_time - x_token["t"][-1]
        for c in self.cols + ["dt"]:  #...what is this next... stuff? Originally it was meant to 
            x_out[f"next_{c}"] = x_out[c][1:]
            assert len(x_out[f"next_{c}"]) == len(x_out["t"]) - 1

        return x_out