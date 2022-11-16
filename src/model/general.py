from itertools import chain, combinations
from ast import literal_eval
from settings.settings import *
from copy import deepcopy
from math import floor
from tensorflow import keras
import numpy as np


class GeneralML:
    def __init__(
        self,
        data = None, 
        Y_Ranges = None
    ):
        self.data = data
        self.Y_Ranges = Y_Ranges
        ms = Settings()
        ms.configure()
        self.num_epochs = ms.num_epochs
        self.batch_size = ms.batch_size
        self.layers_size = ms.layer_size
        self.k = ms.k

    def act_fn(self, x): 
        return keras.activations.relu(x,alpha=0.01)

    def model(self):
        model = keras.Sequential()
        for layer_size in self.layer_sizes:
            model.add(keras.layers.Dense(layer_size, activation = self.act_fn))
            model.add(keras.layers.Dropout(0.1))
        model.add(keras.layers.Dense(1))
        model.compile(optimizer='adam', loss='logcosh') #...choose from: [MAE,MAPE,MSE,logcosh]

        return model

    def train(self, model, Training_data=None, Training_values=None):
        hist_data = []        
        for i in range(self.k):
            hist_data.append(model.fit(Training_data[i], Training_values[i], batch_size=self.batch_size, epochs=self.num_epochs, shuffle=True, validation_split=0., verbose=0))
        return model

    def test(self, model, Testing_data = None, Testing_values=None):
        '''This function counts the number of values that are within certain accs.'''
        loss_data = []
        acc_list = []
        predictions = np.ndarray.flatten(model.predict(Testing_data[i]))
        loss_data.append([float(keras.losses.MAE(Testing_values[i],predictions)),float(keras.losses.logcosh(Testing_values[i],predictions)),float(keras.losses.MAPE(Testing_values[i],predictions)),float(keras.losses.MSE(Testing_values[i],predictions))])
        

        count_A, count_B, count_C = 0, 0, 0
        for test in range(len(predictions)):
            if Testing_values[i][test]-0.5 <= predictions[test] <= Testing_values[i][test]+0.5: #...if they are within 0.5 of predictions
                count_A += 1
                count_B += 1
                count_C += 1
            elif Testing_values[i][test]-Y_choice_range*0.025 <= predictions[test] <= Testing_values[i][test]+Y_choice_range*0.025:
                #...if they are within 2.5% of the total range
                count_B += 1
                count_C += 1
            elif Testing_values[i][test]-Y_choice_range*0.05 <= predictions[test] <= Testing_values[i][test]+Y_choice_range*0.05:
                count_C += 1
        acc_list.append([count_A/len(predictions),count_B/len(predictions),count_C/len(predictions)])

        return