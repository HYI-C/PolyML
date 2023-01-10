import numpy as np
from typing import Callable, Dict, Sequence
import torch
import torch.distributions as d
import torch.nn.functional as f
from torch import nn 


class CompositeLoss(nn.Module):
    '''This gives the sum of all the losses'''
    def __init__(self, dl):
        super().__init__()
        self.losses = nn.ModuleDict(dl)
        self.preprocess = None #..We might want to add some stuff here to preprocess the loss
    def forward(self, y_pred: Dict[str, torch.Tensor], target: Dict[str, torch.Tensor]):
        #...preproccess here
        losses = {name: loss(y_pred[name], target[name]) for name, loss in self.losses.items()} #...is self.losses defined?
        return sum(list(losses.values())), losses

class SumLoss(nn.Module):
    def __init__(self, s: Sequence[nn.Module]):
        super().__init__()
        self.losses = nn.ModuleDict(s)

    def forward(self, y_pred: Dict[str, torch.Tensor], target: Dict[str, torch.Tensor]):
        losses = {name: loss(y_pred, target) for name, loss in self.losses.items()}
        loss_sum = sum([loss[0] for loss in losses.values()])
        loss_dict = {name: loss[0] for name, loss in losses.items()}
        [loss_dict.update(loss[1]) for loss in losses.values()]
        # loss_dict = dict(ChainMap([loss[1] for loss in losses.values()]))
        # [loss_dict.update(d) for name, value
        return loss_sum, loss_dict

class VolumeLoss(nn.Module):
    '''In this loss function we want to minimise the volume difference between
    the input and output shape.

    the input and output is the matrix of the vertices
    '''
    def __init__(self):
        super().__init__()

    def forward(self, y_pred: torch.Tensor, target: torch.Tensor):
        target[target.isnan()] = 0
        
        losses = {name: loss(y_pred, target[name]) for name, loss in self.losses.items()}
        volume_loss = None
        loss_dict = {name: loss[0] for name, loss in losses.items()}
        return volume_loss, loss_dict
    
class VAEloss(nn.Module):
    #...Adds a variational term to any final loss
    def __init__(self, final_loss: nn.Module, reg_weight=1):
        super().__init__()
        self.final_loss = final_loss
        self.reg_weight = reg_weight

    def forward(self, model_out, target_x) -> torch.Tensor:
        fit_loss, losses_dict = self.final_loss(model_out, target_x)
        mu = model_out["mu"]
        std = model_out["std"]

        if self.reg_weight is not None:
            '''This is the KLD loss, https://arxiv.org/abs/1312.6114'''
            KLD_element = 1 + torch.log(std**2) - mu**2 - std**2
            KLD = -0.5 * torch.mean(KLD_element)
            my_loss = fit_loss + self.reg_weight * KLD
        else:
            my_loss = fit_loss
            KLD = 0.0

        losses_dict["kl_div"] = KLD
        losses_dict["model_fit"] = fit_loss
        losses_dict["total"] = my_loss
        losses_dict = {f"loss/{name}": loss for name, loss in losses_dict.items()}

        assert my_loss not in [-np.inf, np.inf], "Loss not finite!"
        assert not torch.isnan(my_loss), "Got an NaN loss"

        assert sum(losses_dict.value()) not in [-np.inf, np.inf], "Loss not finite!"
        assert not torch.isnan(sum(losses_dict.values())), "Got a NaN loss"

        return my_loss, losses_dict