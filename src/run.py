#import libraries
from settings.settings import Settings
from utils.data_utils import *
from itertools import chain, combinations
from ast import literal_eval
import sqlite3
import pandas as pd
import numpy as np

#import modules
from utils.data_utils import *
from utils.data_wrangling import *

from model.general import *



#configure settings
ms = Settings()
ms.configure(quickstart_settings="custom")

# Do algorithm

data, headings = ImportPolyData().get_data(path = "./data/dim_2_plucker.db", file_name = "dim_2_plucker")
data = ImportPolyData().dual_to_float(data)
Y_Ranges = ImportPolyData().get_range(data)

# Create input and target
polygons_X, polygons_Y, Y_choice_range, number_polygon = DataWrangling(data = data, Y_Ranges=Y_Ranges).create_targets()

# Create training and test data
Training_data, Training_values, Testing_data, Testing_values = DataWrangling().tts(polygons_X=polygons_X, polygons_Y=polygons_Y)
#print(Training_data)
# Do the ML
GeneralML(Y_choice_range = Y_choice_range).seq_model_train(Training_data=Training_data, Training_values=Training_values, Testing_data=Testing_data, Testing_values=Testing_values)
'''
# Output the average testing metrics and losses
with open('./MLResults.txt','a') as myfile:
    myfile.write('Accuracies [\pm 0.5, \pm 0.025*range, \pm 0.05*range]: '+str(np.sum(acc_list,axis=0)/k)+'\nLosses [MAE, Log(cosh), MAPE, MSE]: '+str(np.sum(metric_loss_data,axis=0)/k))
'''