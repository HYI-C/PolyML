from itertools import chain, combinations
from ast import literal_eval
from settings.settings import *
from copy import deepcopy
from math import floor
import numpy as np

class data_wrangling:
    def __init__(
        self,
        data = None, 
        Y_Ranges = None,
    ):
        self.data = data
        self.Y_Ranges = Y_Ranges
        ms = Settings()
        ms.configure()
        self.num_vertices = ms.num_vertices
        self.vertices = ms.vertices
        self.pluckers = ms.pluckers
        self.inversion_check = ms.inversion_check
        
    def ML(self, input = "plucker", target = "volume", poly_n=0,Pad=1,Pchoice=0,gcd=0,k_cv=5,split=0.8):
            '''This is the general function for all investigations'''
            #Extract relevant parts of data to ML 
            #Data selection hyper-params
            #X_choice, Y_choice = X, Y   #...choose what to ML (use 'headings': id, vertices, plucker, plucker_len, num_vertices, num_points, num_interior_points, volume, dual_volume, gorenstein_idx, codimension)
            n = poly_n                  #...select number of vertices to ML, use '0' to mean all
            Pad_check = Pad             #...true to perform padding, false to select according to length
            Pad_choice = Pchoice        #...number to pad onto end of vectors, only relevant when padding
            GCD_scheme = gcd             #...whether to augment plucker coords by: (0) nothing, (1) pairwise gcds, (2) (n-1)-gcds
            Y_choice_range = self.Y_Ranges[target-3][1] - self.Y_Ranges[target-3][0] #... (max - min) for the selected varible to ML. Note that Y_range starts at 3

            #Extract relevant X & Y data. Here we add the input properties to
            #polygons_X and the output properties to polygons_Y
            polygons_X, polygons_Y, last_poly_pts = [], [], []
            for idx, poly in enumerate(self.data): # for each polygon in the data 
                if int(poly[self.num_vertices]) == n or n == 0: #...extract only polygons with n vertices, or all polygons if n==0 (inefficient extra 'ifs' but this section is not the time bottleneck)
                    if input == self.vertices and poly[self.vertices]!=last_poly_pts: #...skip repeated lines in dataset (where plucker coords permuted)
                        last_poly_pts = poly[self.vertices] #...keep track of last polygon, so know when moved onto next one
                        polygons_X.append(list(chain(*literal_eval(poly[input])))) #...if using vertices need to flatten to a vector                    
                        polygons_Y.append(poly[target])
                    elif input == self.pluckers: 
                        if GCD_scheme == 1:   #...augment vectors with pairwise gcds: pairwise between plucker coordinates.
                            polygons_X.append(literal_eval(poly[input])+[np.gcd(*np.absolute(x)) for x in combinations(literal_eval(poly[input]),2)]) # need to use literal_eval because data is saved as string
                        elif GCD_scheme == 2: #...augment vectors with (n-1)-gcds
                            polygons_X.append(literal_eval(poly[input])+[np.gcd.reduce(np.absolute(x)) for x in combinations(literal_eval(poly[2]),poly[3]-1)])
                        else: polygons_X.append(literal_eval(poly[input]))
                        polygons_Y.append(poly[target])
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
        