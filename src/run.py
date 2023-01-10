#import libraries
from settings.settings import Settings
from utils.data_utils import *
from itertools import chain, combinations
from ast import literal_eval
import sqlite3
import pandas as pd
import numpy as np
import os 

#import modules
from utils.data_utils import *
from utils.data_wrangling import *

from model.general import *

# Data generation
os.system("sage data/data_gen.py")

#configure settings
ms = Settings()
#ms.configure(quickstart_settings="4d_PV") #...use None for 3d and "4d_PV" for 4d training
ms.configure(quickstart_settings=None)
# Do algorithm

data, headings = ImportPolyData().get_db_data(path = ms.path, file_name = ms.table_name)
data = ImportPolyData().dual_to_float(data=data, float_inx = ms.dual_volume)
# #...This is for if there is dual volume

Y_Ranges = ImportPolyData().get_range(data)

# Create input and target, the input is of the form [[plucker1], [plucker2], etc.]
polygons_X, polygons_Y, Y_choice_range, number_polygon = DataWrangling(data = data, Y_Ranges=Y_Ranges).create_targets(input = ms.pluckers, target = ms.volume)
# Create training and test data
Training_data, Training_values, Testing_data, Testing_values = DataWrangling().tts(polygons_X=polygons_X, polygons_Y=polygons_Y)

# Do the ML
Model = GeneralML(Y_choice_range = Y_choice_range)
Model.seq_model_train(Training_data=Training_data, Training_values=Training_values, Testing_data=Testing_data, Testing_values=Testing_values)
print("Testing accurary:", Model.acc_list)

'''
# Output the average testing metrics and losses
with open('./MLResults.txt','a') as myfile:
    myfile.write('Accuracies [\pm 0.5, \pm 0.025*range, \pm 0.05*range]: '+str(np.sum(acc_list,axis=0)/k)+'\nLosses [MAE, Log(cosh), MAPE, MSE]: '+str(np.sum(metric_loss_data,axis=0)/k))
'''