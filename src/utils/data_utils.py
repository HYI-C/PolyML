import sqlite3
import pandas as pd

#Import Polygon Data
class ImportPolyData:
    def __init__(
        self,
    ):
        self
    
    def get_data(self): 
        '''This function gets data from the database and returns an np array.'''
        with sqlite3.connect("./dim_2_plucker.db") as db: #...ensure this is the correct path to the datafile
            df = pd.read_sql_query("SELECT * FROM dim_2_plucker", db)   #...read database into a pandas dataframe
            headings = df.columns.values                                #...save the headings of each column in the table
            data = df.values                                            #...convert pandas dataframe to np.array
        del(df) 
        return data, headings

    def dual_to_float(self, data):
        '''This function reformats dual volume data to float'''
        for polygon in data:
            if isinstance(polygon[8],str): #...where dual volume interpreted as string convert to a float
                polygon[8] = float(polygon[8].split('/')[0])/float(polygon[8].split('/')[1])
        del(polygon)
        return polygon
#Extract the ranges of the polygon parameters
    def get_range(self, data): 
        Y_Ranges = [[min([poly[i] for poly in data]),max([poly[i] for poly in data])] for i in [3,4,5,6,7,8,9,10]]
        return Y_Ranges