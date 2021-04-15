from import_data import import_training_data
import pandas as pd

df = import_training_data()

df.to_csv('training_data.csv')
