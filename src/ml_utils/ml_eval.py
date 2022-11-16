from itertools import chain, combinations
from ast import literal_eval
from settings.settings import *
from copy import deepcopy
from math import floor
import numpy as np

class MLEval:
    def __init__(
        self,
    ):
        pass

    def k_cv(self, k_cv=5, split=0.8, polygons_X=None, polygons_Y=None):
        k = k_cv #...number of cross-validations to perform
        tt_split = split #... only relevant if no cross-validation, sets the size of the train:test ratio split
        num_polygon = len(polygons_Y)
        ML_data = [[polygons_X[index],polygons_Y[index]] for index in range(num_polygon)]

        np.random.shuffle(ML_data)

        Training_data, Training_values, Testing_data, Testing_values = [], [], [], []
        if k > 1:
            s = int(floor(len(ML_data)/k))        #...number of datapoints in each validation split
            for i in range(k):
                Training_data.append([HS[0] for HS in ML_data[:i*s]]+[HS[0] for HS in ML_data[(i+1)*s:]])
                Training_values.append([HS[1] for HS in ML_data[:i*s]]+[HS[1] for HS in ML_data[(i+1)*s:]])
                Testing_data.append([HS[0] for HS in ML_data[i*s:(i+1)*s]])
                Testing_values.append([HS[1] for HS in ML_data[i*s:(i+1)*s]])
        elif k == 1:
            s = int(floor(len(ML_data)*tt_split)) #...number of datapoints in train split
            Training_data.append([HS[0] for HS in ML_data[:s]])
            Training_values.append([HS[1] for HS in ML_data[:s]])
            Testing_data.append([HS[0] for HS in ML_data[s:]])
            Testing_values.append([HS[1] for HS in ML_data[s:]])

        return Training_data, Training_values, Testing_data, Testing_values