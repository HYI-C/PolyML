import os
import scipy.constants as spc

path = "./data/dim_4_plucker.db"
table_name = "poly_vol_4"

# Select investigations to run
input_set = [1]       #...select 1 --> vertices, or 2 --> pluckers, or both
output_set = [2] #...select 7 --> volume, 8 --> dual volume, 9 --> gorenstein index, 10 --> codimension, or any combination thereof
number_vertex_set = [3,4,5,6] #...select which datasets of polygons with this many vertices to consider, note including '0' will run ML on all polygons using vector padding (can edit padding functionality in function call: what choice of number to pad with)
crossval_check = True   #...select whether to perform cross-validation, default is 5-fold (can edit in function directly)
tt_ratio_set = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]    #...if running over varying train/test split ratios (where no cross-validation, st k=1) select the train set proportions to consider from the interval (0,1)
gcd_scheme = 0          #...select which gcd augmentaion scheme to use: 0 --> none, 1 --> pairwise gcds, 2 --> (n-1)-gcds
inversion_check = False #...choose whether to run the inversion investigation, swapping the final vector entry with the learned parameter

# Input and output set
vertices = 0
pluckers = 1
volume = 2
num_vertices = 3

# ML settings
num_epochs = 20
batch_size = 32
layer_sizes = [64, 64, 64, 64]

k = 5 #...k-fold vallidation