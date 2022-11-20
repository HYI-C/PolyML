import pandas as pd
import sqlite3

def get_db_data(): 
    '''This function gets data from the database and returns an np array.'''
    with sqlite3.connect("./notebooks/dim_4_plucker.db") as db: #...ensure this is the correct path to the datafile
        cur = db.cursor()
        df = pd.read_sql_query("SELECT * FROM poly_vol_4 ", db)   #...read database into a pandas dataframe
        headings = df.columns.values                                #...save the headings of each column in the table
        data = df.values                                            #...convert pandas dataframe to np.array
    del(df) 
    return data, headings


data, headings = get_db_data()
i=500
print("Testing \n", "Vertices", data[i][0], "\n plücker", data[i][1], "\n Volume", data[i][2], "\n num_vertices", data[i][3])