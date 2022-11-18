from itertools import chain, combinations
from ast import literal_eval
from settings.settings import *
from copy import deepcopy
from math import floor
from tensorflow import keras
import tensorflow as tf
import numpy as np


class GeneralML:
    def __init__(
        self,
        Y_choice_range = None
    ):
        self.Y_choice_range = Y_choice_range
        ms = Settings()
        ms.configure()
        self.num_epochs = ms.num_epochs
        self.batch_size = ms.batch_size
        self.layer_sizes = ms.layer_sizes
        self.k = ms.k
        self.loss_data = []
        self.hist_data = []
        self.acc_list = []
        self.model = None


    def act_fn(self, x): 
        return keras.activations.relu(x,alpha=0.01)

    def seq_model(self):
        model = keras.Sequential()
        for layer_size in self.layer_sizes:
            model.add(keras.layers.Dense(layer_size, activation = self.act_fn))
            #model.add(keras.layers.Dropout(0.1))
        model.add(keras.layers.Dense(1))
        model.compile(optimizer='adam', loss='logcosh') #...choose from: [MAE,MAPE,MSE,logcosh]
        self.model = model
        return 

    def seq_model_train(self, Training_data=None, Training_values=None, Testing_data=None, Testing_values=None):
        # Training the model   

        for i in range(self.k):
            self.seq_model()
            
            Y = Training_data[i]
            X = Training_values[i]
            #Y = tf.convert_to_tensor(Y, dtype=tf.int64)
            #X = tf.convert_to_tensor(X, dtype=tf.int64)

            self.hist_data.append(self.model.fit(Y, X, batch_size=self.batch_size, epochs=self.num_epochs, shuffle=True, validation_split=0., verbose=0))
            self.seq_model_test(Testing_data=Testing_data, Testing_values=Testing_values, i=i)
        # Testing the model

    def seq_model_test(self, Testing_data= None, Testing_values = None, i=0):
        predictions = np.ndarray.flatten(self.model.predict(Testing_data[i]))
        self.loss_data.append([float(keras.losses.MAE(Testing_values[i],predictions)),float(keras.losses.logcosh(Testing_values[i],predictions)),float(keras.losses.MAPE(Testing_values[i],predictions)),float(keras.losses.MSE(Testing_values[i],predictions))])
        count_A, count_B, count_C = 0, 0, 0
        for test in range(len(predictions)):
            if Testing_values[i][test]-0.5 <= predictions[test] <= Testing_values[i][test]+0.5: #...if they are within 0.5 of predictions
                count_A += 1
                count_B += 1
                count_C += 1
            elif Testing_values[i][test]-self.Y_choice_range*0.025 <= predictions[test] <= Testing_values[i][test]+self.Y_choice_range*0.025:
                #...if they are within 2.5% of the total range
                count_B += 1
                count_C += 1
            elif Testing_values[i][test]-self.Y_choice_range*0.05 <= predictions[test] <= Testing_values[i][test]+self.Y_choice_range*0.05:
                count_C += 1
        self.acc_list.append([count_A/len(predictions),count_B/len(predictions),count_C/len(predictions)])
        return