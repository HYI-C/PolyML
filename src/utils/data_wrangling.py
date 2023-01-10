from itertools import chain, combinations
from ast import literal_eval
from settings.settings import *
from copy import deepcopy
from math import floor
import numpy as np

class DataWrangling:
    def __init__(
        self,
        data = None, 
        Y_Ranges = None,
        config = None,
    ):
        self.data = data
        self.Y_Ranges = Y_Ranges
        ms = Settings()
        self.num_vertices = ms.num_vertices
        self.vertices = ms.vertices
        self.pluckers = ms.pluckers
        self.inversion_check = ms.inversion_check
        self.volume = ms.volume
        self.k = ms.k
        try:
            if ms.n:
                self.n = ms.n
        except:
            pass
    #TODO the main problem here is that the pluckers are of different lengths,
    # which is something that our neural network can't handle. 
    def create_targets(self, input = 2, target = 7, poly_n=0,Pad=1,Pchoice=0,gcd=0,k_cv=5,split=0.8):
            '''This is the general function for all investigations. This
            defaults to an input of pluckers and target of volume'''
            #Extract relevant parts of data to ML 
            #Data selection hyper-params
            #X_choice, Y_choice = X, Y   #...choose what to ML (use 'headings': id, vertices, plucker, plucker_len, num_vertices, num_points, num_interior_points, volume, dual_volume, gorenstein_idx, codimension)
            try:
                n = self.n
            except:
                n = poly_n                  #...select number of vertices to ML, use '0' to mean all
            Pad_check = Pad             #...true to perform padding, false to select according to length
            Pad_choice = Pchoice        #...number to pad onto end of vectors, only relevant when padding
            GCD_scheme = gcd             #...whether to augment plucker coords by: (0) nothing, (1) pairwise gcds, (2) (n-1)-gcds
            try:
                Y_choice_range = self.Y_Ranges[target-3][1] - self.Y_Ranges[target-3][0] #... (max - min) for the selected varible to ML. Note that Y_range starts at 3
            except:
                Y_choice_range = self.Y_Ranges[1] - self.Y_Ranges[0]
            #Extract relevant X & Y data. Here we add the input properties to
            #polygons_X and the output properties to polygons_Y
            polygons_X, polygons_Y, last_poly_pts = [], [], []
            for idx, poly in enumerate(self.data): # for each polygon in the data 
                if int(poly[self.num_vertices]) == n or n == 0: #...extract only polygons with n vertices, or all polygons if n==0 (inefficient extra 'ifs' but this section is not the time bottleneck)
                    if input == self.vertices and poly[self.vertices]!=last_poly_pts: #...skip repeated lines in dataset (where plucker coords permuted)
                        last_poly_pts = poly[self.vertices] #...keep track of last polygon, so know when moved onto next one
                        polygons_X.append(list(chain(*literal_eval(poly[input])))) #...if using vertices need to flatten to a vector                    
                        polygons_Y.append(literal_eval(poly[target]))
                    elif input == self.pluckers: #...usually we use this
                        if GCD_scheme == 1:   #...augment vectors with pairwise gcds: pairwise between plucker coordinates.
                            polygons_X.append(literal_eval(poly[input])+[np.gcd(*np.absolute(x)) for x in combinations(literal_eval(poly[input]),2)]) # need to use literal_eval because data is saved as string
                        elif GCD_scheme == 2: #...augment vectors with (n-1)-gcds
                            polygons_X.append(literal_eval(poly[input])+[np.gcd.reduce(np.absolute(x)) for x in combinations(literal_eval(poly[2]),poly[3]-1)])
                        else:
                            polygons_X.append(literal_eval(poly[input]))
                            polygons_Y.append(literal_eval(str(poly[target])))
            number_polygon = len(polygons_Y)

                
            #Run inversion, editing data accordingly swapping param with last entry of the input vector
            if self.inversion_check:
                params = deepcopy(polygons_Y) #...We now want the Y to be the input and X to be the target
                for poly in range(len(polygons_X)):
                    polygons_Y[poly] = polygons_X[poly][-1]
                    polygons_X[poly][-1] = params[poly]
                Y_choice_range = max(polygons_Y)-min(polygons_Y)
                del(params)
            
            #Pad plucker coordinates if desired (only needed for n == 0)
            if Pad_check and n != 0:
                max_length = max(map(len,polygons_X))
                for polygon in polygons_X:
                    while len(polygon) < max_length: #...pad all X vectors to the maximum length
                        polygon += [Pad_choice]
                del(polygon,poly)

            return polygons_X, polygons_Y, Y_choice_range, number_polygon


    def tts(self, split=0.8, polygons_X=None, polygons_Y=None):
        k = self.k #...number of cross-validations to perform
        tt_split = split #... only relevant if no cross-validation, sets the size of the train:test ratio split
        num_polygon = len(polygons_Y)
        ML_data = [[polygons_X[index],polygons_Y[index]] for index in range(num_polygon)]

        np.random.shuffle(ML_data)

        Training_data, Training_values, Testing_data, Testing_values = [],[],[],[]
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