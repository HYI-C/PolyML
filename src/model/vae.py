import numpy as np
import pytorch_lightning as pl
import torch 
from torch import nn
from torch.optim import Adam
from torchmetrics import Accuracy, MeanAbsoluteError, MetricCollection

from losses.vae_losses import * #...This imports all the losses, including volume loss, accuracy

'''
This is the Variational auto-encoder module, not sure why we want to use a VAE
instead of an AE---which makes more intuitive sense.?
'''