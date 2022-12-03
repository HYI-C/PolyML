import sqlite3
import pandas as pd

#Import Polygon Data
class ImportPolyData:
    def __init__(
        self,
    ):
        self
    
    def get_db_data(self, path = None, file_name=None): 
        '''This function gets data from the database and returns an np array.'''
        with sqlite3.connect(path) as db: #...ensure this is the correct path to the datafile
            df = pd.read_sql_query("SELECT * FROM {}".format(file_name), db)   #...read database into a pandas dataframe
            headings = df.columns.values                                #...save the headings of each column in the table
            data = df.values                                            #...convert pandas dataframe to np.array
        del(df) 
        return data, headings

    def dual_to_float(self, data= None, float_inx= None):
        '''This function reformats dual volume data to float'''
        for polygon in data:
            if isinstance(polygon[8],str): #...where dual volume interpreted as string convert to a float
                polygon[8] = float(polygon[8].split('/')[0])/float(polygon[8].split('/')[1])
        del(polygon)
        return data
#Extract the ranges of the polygon parameters
    def get_range(self, data): # refactor this to handle target
        try:
            Y_Ranges = [[min([poly[i] for poly in data]),max([poly[i] for poly in data])] for i in [3,4,5,6,7,8,9,10]]
        except:
            #Y_Ranges = [[min([float(poly[i]) for poly in data]),max([float(poly[i]) for poly in data])] for i in [0,1,2,3]]
            Y_Ranges = [min([float(poly[2]) for poly in data]),max([float(poly[2]) for poly in data])]
        return Y_Ranges