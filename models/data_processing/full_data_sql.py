import sqlite3
from import_data import import_training_data
import pandas as pd

data = import_training_data()
conn = sqlite3.connect('../SQL_Data/full_training_data/full_data.sqlite3')
data.to_sql('Training_Data', con=conn, if_exists='replace')
num_laps = data['finalLapTime'].nunique()
print(f'{num_laps} laps added to the database')
