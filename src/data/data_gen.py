#!/opt/sage/sage -python

import numpy as np
import sqlite3
from sage.all import *

# Read in the data
with open('./data/v05.txt', 'r') as f:
    L = f.readlines()
L=[l.strip() for l in L]
L = [l for l in L if "N" not in l] #...select the matrices only
L = L[:-1]
L = [l.split() for l in L]         #...use delimiter
L = [list(map(int, l)) for l in L] #...convert to int


# Define functions 
def plk(vec): #...plucker 
    vert = matrix(vec)
    ker=vert.kernel().matrix()
    return ker.minors(ker.nrows())

def polyvol(vec): #...volume
    vert = matrix(vec)
    vol=Polyhedron(vertices=vert).volume()
    return vol

# Build the plücker coordinates and volumes
plklist=[]
for i in range(0,len(L)):
    plklist.append(plk(L[i].T.tolist()))
volist=[]
for i in range(0,len(L)):
    volist.append(24*polyvol(L[i].T))

# Save to database
con = sqlite3.connect("dim_4_plucker.db")
cur = con.cursor()

cur.execute("CREATE TABLE poly_vol_4(vertices, plucker, volume, num_vertices)")
con.commit()