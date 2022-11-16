#import libraries
from settings.settings import Settings
from utils.data_utils import *
from itertools import chain, combinations
from ast import literal_eval
import sqlite3
import pandas as pd
import numpy as np

#configure settings
ms = Settings()
ms.configure(quickstart_settings="custom")

# Do algorithm

# Output the average testing metrics and losses
with open('./MLResults.txt','a') as myfile:
    myfile.write('Accuracies [\pm 0.5, \pm 0.025*range, \pm 0.05*range]: '+str(np.sum(acc_list,axis=0)/k)+'\nLosses [MAE, Log(cosh), MAPE, MSE]: '+str(np.sum(metric_loss_data,axis=0)/k))
    